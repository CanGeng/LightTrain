"""StandardEngine — thin orchestrator that delegates to an UpdateRule.

The engine owns the ``Accelerator`` (mixed precision, device placement). For
the actual forward / backward / clip / step sequence it asks the configured
``update_rule`` so research code can swap in alternative training math
(reverse KL, RPO, …) without touching the engine.
"""

from __future__ import annotations

from typing import Any, Mapping

from lighttrain.registry import register
from lighttrain.engine._context import StepContext


@register("engine", "standard")
class StandardEngine:
    def __init__(
        self,
        *,
        update_rule: Any,
        loss_fn: Any | None = None,
        accelerator: Any | None = None,
    ) -> None:
        self.update_rule = update_rule
        self.loss_fn = loss_fn
        self.accelerator = accelerator

    def step(self, batch: Mapping[str, Any], ctx: StepContext) -> dict[str, Any]:
        if ctx.loss_fn is None:
            ctx.loss_fn = self.loss_fn
        if ctx.accelerator is None:
            ctx.accelerator = self.accelerator
        return self.update_rule.step(ctx.model, batch, ctx)


__all__ = ["StandardEngine"]
