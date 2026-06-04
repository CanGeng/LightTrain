"""Loss attribution.

Three-level loss attribution:

* **sample**: per-sample loss (reduce='none' summed over token dim).
* **token**:  per-(sample, position) loss matrix.
* **module**: ``torch.autograd.grad`` of the loss with respect to each
  named module's output (top-K modules by gradient norm).

Sample + token levels are cheap and run on demand. Module level requires
a second backward pass through the captured module outputs and is only
auto-triggered on NaN / invariant violation.

The :class:`LossAttributionCallback` wraps the function as a callback
that fires periodically (``every_n_steps``) and also on demand via
``compute_loss_attribution_now(...)``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lighttrain.registry import register


def compute_loss_attribution(
    *,
    model: Any,
    batch: Any,
    outputs: Any,
    loss: Any,
    levels: Iterable[str] = ("sample", "token"),
    ignore_index: int = -100,
    top_k_modules: int = 10,
) -> dict[str, Any]:
    """Compute one or more attribution levels. Returns JSON-safe dict.

    For ``sample`` / ``token`` we recompute per-element loss from
    ``outputs.outputs['logits']`` + ``batch['labels']`` (next-token shift
    matching :class:`CrossEntropyLoss`). For ``module`` we ask
    ``torch.autograd.grad`` for the gradient of the scalar ``loss`` with
    respect to each captured module's output.
    """
    levels = list(levels)
    out: dict[str, Any] = {"levels": levels}
    logits = None
    labels = None
    if isinstance(batch, dict):
        labels = batch.get("labels")
    if hasattr(outputs, "outputs"):
        logits = outputs.outputs.get("logits")

    if ("sample" in levels or "token" in levels) and isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
        # Same shift as CrossEntropyLoss in lighttrain.builtin_plugins.losses.core.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, T, V = shift_logits.shape
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=int(ignore_index),
            reduction="none",
        ).view(B, T)
        mask = (shift_labels != int(ignore_index)).float()
        valid = mask.sum(dim=1).clamp(min=1.0)
        per_sample = (per_token * mask).sum(dim=1) / valid
        if "sample" in levels:
            order = torch.argsort(per_sample, descending=True)
            out["sample"] = {
                "loss_per_sample": per_sample.detach().cpu().tolist(),
                "topk_indices": order[: max(1, B // 2)].detach().cpu().tolist(),
            }
        if "token" in levels:
            out["token"] = {
                "loss_per_token": per_token.detach().cpu().tolist(),
                "valid_mask": mask.detach().cpu().tolist(),
            }

    if "module" in levels and isinstance(loss, torch.Tensor) and model is not None:
        # Re-run forward with hooks to capture module outputs, then take
        # grad of loss w.r.t. each captured output. This is best-effort —
        # captured tensors must require grad and be part of the graph.
        captured: dict[str, torch.Tensor] = {}
        handles: list[Any] = []

        def _capture(name: str):
            def _hook(_mod: Any, _in: Any, output: Any) -> None:
                if isinstance(output, torch.Tensor) and output.requires_grad:
                    captured[name] = output

            return _hook

        for name, mod in model.named_modules():
            if mod is model:
                continue
            handles.append(mod.register_forward_hook(_capture(name)))
        try:
            # The provided ``outputs`` was produced under no-grad in some
            # paths; recompute with grad enabled.
            with torch.enable_grad():
                _ = model(**batch)
            module_norms: dict[str, float] = {}
            if captured:
                grads = torch.autograd.grad(
                    loss,
                    list(captured.values()),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                for (name, _), g in zip(captured.items(), grads, strict=False):
                    if g is None:
                        continue
                    module_norms[name] = float(g.detach().data.norm(2).item())
            top = sorted(module_norms.items(), key=lambda kv: kv[1], reverse=True)
            out["module"] = {"top_k": top[: int(top_k_modules)]}
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass

    return out


def render_attribution_markdown(report: dict[str, Any], *, step: int) -> str:
    lines = [f"# Loss attribution — step {step}", ""]
    if "sample" in report:
        sample = report["sample"]
        lines += ["## Per-sample loss", ""]
        for i, v in enumerate(sample.get("loss_per_sample", [])):
            lines.append(f"- sample[{i}] = {v:.4f}")
        lines.append("")
    if "module" in report:
        lines += ["## Top modules by ∂loss/∂out norm", ""]
        for name, g in report["module"].get("top_k", []):
            lines.append(f"- `{name}` :: {g:.4f}")
        lines.append("")
    if "token" in report:
        lines += [
            "## Per-token loss matrix",
            "",
            "(omitted from markdown — see JSON dump for full B×T matrix)",
            "",
        ]
    return "\n".join(lines)


@register("callback", "loss_attribution")
class LossAttributionCallback:
    """Drop a loss-attribution snapshot every ``every_n_steps`` steps.

    Always runs ``sample`` + ``token``; runs ``module`` only on
    ``on_invariant_fail`` / ``on_nan_detected`` (or when explicitly
    asked via :func:`compute_loss_attribution_now`).
    """

    def __init__(
        self,
        *,
        every_n_steps: int = 500,
        levels: Iterable[str] = ("sample", "token"),
        on_nan: bool = True,
    ) -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.levels = tuple(levels)
        self.on_nan = bool(on_nan)
        self._run_dir: Path | None = None
        self._latest_outputs: Any = None
        self._latest_batch: Any = None
        self._latest_loss: Any = None
        self._model: Any = None

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        rd = getattr(ctx, "run_dir", None) if ctx is not None else None
        if rd is None and trainer is not None:
            rd = getattr(trainer, "_run_dir", None)
        self._run_dir = Path(rd) if rd is not None else None
        self._model = (
            getattr(ctx, "model", None)
            if ctx is not None
            else getattr(trainer, "model", None)
        )

    def on_loss_computed(
        self,
        *,
        step: int = 0,
        loss: Any = None,
        outputs: Any = None,
        batch: Any = None,
        model: Any = None,
        **_: Any,
    ) -> None:
        self._latest_outputs = outputs
        self._latest_batch = batch
        self._latest_loss = loss
        if model is not None:
            self._model = model

    def on_step_end(self, *, step: int = 0, **_: Any) -> None:
        if self._run_dir is None or step <= 0 or step % self.every_n_steps != 0:
            return
        self._dump(step=step, force_module=False)

    def on_nan_detected(self, *, step: int = 0, **_: Any) -> None:
        if self._run_dir is None or not self.on_nan:
            return
        self._dump(step=step, force_module=True)

    def _dump(self, *, step: int, force_module: bool) -> None:
        if self._latest_outputs is None or self._latest_batch is None or self._run_dir is None:
            return
        run_dir = self._run_dir
        levels = list(self.levels)
        if force_module and "module" not in levels:
            levels.append("module")
        try:
            report = compute_loss_attribution(
                model=self._model,
                batch=self._latest_batch,
                outputs=self._latest_outputs,
                loss=self._latest_loss,
                levels=levels,
            )
        except Exception:  # noqa: BLE001
            return
        out_dir = run_dir / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"loss_attribution_{int(step)}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (out_dir / f"loss_attribution_{int(step)}.md").write_text(
            render_attribution_markdown(report, step=int(step)),
            encoding="utf-8",
        )


__all__ = [
    "LossAttributionCallback",
    "compute_loss_attribution",
    "render_attribution_markdown",
]
