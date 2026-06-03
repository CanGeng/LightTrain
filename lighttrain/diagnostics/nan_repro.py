"""NaN repro writer.

When a NaN/Inf is detected (by ``NanHunterCallback`` or by a manual call
from a diagnostics path) we drop a self-contained reproduction kit to
``runs/<...>/diagnostics/repro_nan_<ts>/`` so the user can reproduce the
crash with a single ``python repro.py`` outside the framework.

The kit:

```
repro_nan_<ts>/
  repro.py              # 80-line stand-alone script
  batch.pt              # weights-only torch.save of the offending batch
  model_state.safetensors
  model_spec.json       # registry short-name + params for build_minimal_model
  README.md             # how to run
```

The repro script depends only on torch + safetensors + lighttrain.minimal
(core layer). No OmegaConf / Pydantic / Trainer.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_model as _save_model

from ..minimal import dump_spec

_REPRO_TEMPLATE = '''\
"""Auto-generated NaN reproduction script.

Run with:

    python repro.py

Reproduces the NaN/Inf that aborted the parent run at:

    run    = {run}
    step   = {step}
    module = {module}
    error  = {error!r}

Only depends on torch + safetensors + lighttrain.minimal (~80 LoC each).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Re-populate the registry — pull in adapters for whatever model the parent
# run used. Adjust the import below if you used a model outside tiny_lm.
import lighttrain.builtin_plugins.models.adapters  # noqa: F401

from lighttrain.minimal import build_minimal_model, load_state


HERE = Path(__file__).parent
spec = json.loads((HERE / "model_spec.json").read_text(encoding="utf-8"))
model = build_minimal_model(spec)
load_state(model, HERE / "model_state.safetensors", strict=False)
batch = torch.load(HERE / "batch.pt", weights_only=True)
model.eval()

with torch.autograd.detect_anomaly():
    out = model(**batch)
    logits = out.outputs.get("logits") if hasattr(out, "outputs") else None
    if logits is not None:
        bad = (~torch.isfinite(logits)).sum().item()
        print(f"non-finite logits: {{bad}}")
    if hasattr(out, "loss") and out.loss is not None:
        print(f"loss: {{out.loss.item()}}")
'''


def write_nan_repro(
    run_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    exception: BaseException | None = None,
    module_name: str = "",
    model_spec: Mapping[str, Any] | None = None,
) -> Path:
    """Write a NaN repro kit; return the kit directory.

    ``model_spec`` should be a JSON-safe dict like
    ``{"name": "tiny_lm", "params": {"vocab_size": 260, ...}}``. If omitted
    we try to infer it from ``type(model).__name__`` lowercased — this works
    for the framework's own ``tiny_lm`` because registry short names match
    the class slug. For HF / custom models the caller must pass ``model_spec``.
    """
    run_dir = Path(run_dir)
    diag = run_dir / "diagnostics" / f"repro_nan_{int(time.time())}"
    diag.mkdir(parents=True, exist_ok=True)

    # 1) model state — ``safetensors.torch.save_model`` handles tied weights
    # (e.g. tiny_lm's ``tok_emb.weight is lm_head.weight``) by dropping the
    # alias instead of refusing to save. ``load_state(..., strict=False)``
    # picks the surviving copy back up.
    _save_model(model, str(diag / "model_state.safetensors"))

    # 2) batch — torch.save with weights_only-compatible tensors.
    safe_batch = {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    torch.save(safe_batch, str(diag / "batch.pt"))

    # 3) model_spec.json — short name + params, used by build_minimal_model.
    if model_spec is None:
        model_spec = _infer_spec(model)
    (diag / "model_spec.json").write_text(
        json.dumps(dump_spec(model_spec["name"], model_spec.get("params", {})), indent=2),
        encoding="utf-8",
    )

    # 4) repro.py.
    tb = ""
    if exception is not None:
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    repro = _REPRO_TEMPLATE.format(
        run=str(run_dir),
        step=int(step),
        module=module_name or "<unknown>",
        error=str(exception) if exception else "",
    )
    (diag / "repro.py").write_text(repro, encoding="utf-8")

    # 5) README.md.
    readme = [
        f"# NaN repro — step {step}",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Module: `{module_name or '<unknown>'}`",
        f"- Exception: `{type(exception).__name__ if exception else 'NaN detected (no exception)'}`",
        "",
        "## Run it",
        "",
        "```bash",
        "python repro.py",
        "```",
    ]
    if tb:
        readme += ["", "## Traceback", "", "```", tb.strip(), "```"]
    (diag / "README.md").write_text("\n".join(readme), encoding="utf-8")

    return diag


def _infer_spec(model: torch.nn.Module) -> dict[str, Any]:
    """Best-effort spec inference for built-in models.

    We look up the class name on the registry to verify the inferred slug
    actually resolves; if it doesn't we fall back to a ``_target_`` form
    (the repro.py template handles both).
    """
    cls = type(model)
    candidates = (
        cls.__name__.lower(),
        cls.__name__.lower().replace("model", ""),
        cls.__name__.lower().replace("causallm", "_lm"),
    )
    from ..registry import contains

    for name in candidates:
        if contains("model", name):
            return {"name": name, "params": _extract_init_params(model)}
    return {
        "_target_": f"{cls.__module__}:{cls.__name__}",
        "params": _extract_init_params(model),
    }


def _extract_init_params(model: torch.nn.Module) -> dict[str, Any]:
    """Pull architecture-ish ints/floats off the model for repro.

    We don't try to be exhaustive — the repro script only needs enough to
    build a shape-compatible architecture; load_state(strict=False) handles
    any name mismatches.
    """
    out: dict[str, Any] = {}
    for k in (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "max_seq_len",
        "dropout",
        "tie_weights",
    ):
        if hasattr(model, k):
            v = getattr(model, k)
            if isinstance(v, (int, float, bool, str)):
                out[k] = v
    return out


__all__ = ["write_nan_repro"]
