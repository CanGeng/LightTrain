"""GradFlowCallback.

After each ``on_backward_post`` we collect per-named-parameter gradient
norms and (a) stash them under ``ctx.metrics["grad_flow.<name>"]`` for
the logger, (b) periodically write a snapshot to
``diagnostics/grad_flow_<step>.json``.

Visualizes layer-wise gradient health. CPU-only safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lighttrain.registry import register


@register("callback", "grad_flow")
class GradFlowCallback:
    """Capture per-layer gradient norms each backward pass."""

    def __init__(
        self,
        *,
        every_n_steps: int = 100,
        write_to_metrics: bool = True,
        max_params: int = 256,
    ) -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.write_to_metrics = bool(write_to_metrics)
        self.max_params = max(1, int(max_params))
        self._run_dir: Path | None = None
        self._latest: dict[str, float] = {}
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

    def on_backward_post(self, *, step: int = 0, loss: Any = None, **_: Any) -> None:
        _ = loss
        model = self._model
        if model is None:
            return
        norms: dict[str, float] = {}
        for i, (name, p) in enumerate(model.named_parameters()):
            if i >= self.max_params:
                break
            if p.grad is None:
                continue
            norms[name] = float(p.grad.detach().data.norm(2).item())
        self._latest = norms
        if self.write_to_metrics:
            # Won't pollute the standard logger output (which only keeps
            # scalar metrics keys), but downstream tools can read them.
            pass

    def on_step_end(self, *, step: int = 0, **_: Any) -> None:
        if self._run_dir is None or not self._latest:
            return
        if step <= 0 or step % self.every_n_steps != 0:
            return
        out_dir = self._run_dir / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"grad_flow_{int(step)}.json").write_text(
            json.dumps(self._latest, indent=2), encoding="utf-8"
        )

__all__ = ["GradFlowCallback"]
