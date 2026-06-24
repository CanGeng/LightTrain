"""Crash bundle writer.

When the training loop raises an unhandled exception we drop a
``runs/<...>/diagnostics/crash_<ts>/`` directory with everything needed
to ``lighttrain replay --run <run>``:

```
crash_<ts>/
  batch.pt
  decoded.txt
  model_state.safetensors
  optimizer_state.pt
  rng.pt
  env.json
  metrics_recent.jsonl
  traceback.txt
  logs_tail.txt
  model_spec.json
```

Shares the underlying packing logic with :mod:`frozen_step` (crash and
frozen bundles use the same format) by leaning on the
in-memory :class:`FrozenStepWriter.snapshot` already kept by
``FrozenStepCallback``. When that callback isn't installed we still
produce as much as we can — model + batch + traceback always.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_model as _save_model

from ..minimal import dump_spec
from ..utils.seed import rng_state

_log = logging.getLogger(__name__)


def write_crash_bundle(
    run_dir: str | Path,
    *,
    exception: BaseException,
    step: int,
    model: torch.nn.Module | None = None,
    batch: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    metrics: Mapping[str, Any] | None = None,
    recent_logs: str = "",
    tokenizer: Any | None = None,
) -> Path:
    """Write a crash bundle and return the directory."""
    run_dir = Path(run_dir)
    bundle = run_dir / "diagnostics" / f"crash_{int(time.time())}"
    bundle.mkdir(parents=True, exist_ok=True)

    # traceback.
    tb_str = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    (bundle / "traceback.txt").write_text(tb_str, encoding="utf-8")

    # env capture (lightweight).
    env: dict[str, Any] = {
        "exception_type": type(exception).__name__,
        "exception_str": str(exception),
        "step": int(step),
        "ts": time.time(),
    }
    try:
        from ..utils.env_capture import capture_env

        env.update(capture_env())
    except Exception:  # noqa: BLE001
        _log.warning(
            "crash_bundle: env capture failed; env.json omits extended environment info",
            exc_info=True,
        )
    (bundle / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")

    # batch + decoded.
    if isinstance(batch, dict):
        safe = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        torch.save(safe, str(bundle / "batch.pt"))
        decoded_lines: list[str] = []
        ids = batch.get("input_ids")
        if isinstance(ids, torch.Tensor) and tokenizer is not None and hasattr(
            tokenizer, "decode"
        ):
            for i in range(ids.shape[0]):
                try:
                    decoded_lines.append(f"# sample[{i}]")
                    decoded_lines.append(str(tokenizer.decode(ids[i].tolist())))
                    decoded_lines.append("")
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "crash_bundle: tokenizer.decode failed for sample %d; decoded.txt records a placeholder",
                        i,
                        exc_info=True,
                    )
                    decoded_lines.append("<decode error>")
        (bundle / "decoded.txt").write_text(
            "\n".join(decoded_lines), encoding="utf-8"
        )

    # model state + spec.
    if model is not None:
        try:
            _save_model(model, str(bundle / "model_state.safetensors"))
        except Exception:  # noqa: BLE001
            _log.warning(
                "crash_bundle: model state save failed; bundle omits model_state.safetensors",
                exc_info=True,
            )
        try:
            from .nan_repro import _infer_spec  # private helper, reused

            spec = _infer_spec(model)
            (bundle / "model_spec.json").write_text(
                json.dumps(dump_spec(spec.get("name", "<unknown>"), spec.get("params", {})), indent=2)
                if "name" in spec
                else json.dumps(spec, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            _log.warning(
                "crash_bundle: model spec inference failed; bundle omits model_spec.json",
                exc_info=True,
            )

    # optimizer state.
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        try:
            torch.save(optimizer.state_dict(), str(bundle / "optimizer_state.pt"))
        except Exception:  # noqa: BLE001
            _log.warning(
                "crash_bundle: optimizer state capture failed; bundle omits optimizer_state.pt",
                exc_info=True,
            )

    # rng.
    try:
        torch.save(rng_state(), str(bundle / "rng.pt"))
    except Exception:  # noqa: BLE001
        _log.warning(
            "crash_bundle: RNG state capture failed; bundle omits rng.pt (replay won't be bit-exact)",
            exc_info=True,
        )

    # metrics_recent.jsonl — one line.
    if metrics:
        try:
            with (bundle / "metrics_recent.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"step": int(step), **{k: _scalar(v) for k, v in metrics.items()}}) + "\n")
        except Exception:  # noqa: BLE001
            _log.warning(
                "crash_bundle: recent metrics write failed; bundle omits metrics_recent.jsonl",
                exc_info=True,
            )

    # tail of logs (if caller supplied any).
    if recent_logs:
        (bundle / "logs_tail.txt").write_text(str(recent_logs), encoding="utf-8")

    return bundle


def _scalar(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        try:
            return float(v.detach().item())
        except Exception:  # noqa: BLE001
            _log.warning(
                "crash_bundle: tensor metric could not be coerced to float; recording None",
                exc_info=True,
            )
            return None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


__all__ = ["write_crash_bundle"]
