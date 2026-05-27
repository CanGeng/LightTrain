"""SequenceParallelStrategy — split the sequence dimension across TP ranks.

SP is always combined with TP: it shards the sequence dimension so that
attention's QKV computation sees a local sub-sequence, then reassembles
via AllGather before the output projection.

Requires PyTorch >= 2.2 (sequence_parallel kwarg in ColwiseParallel).
"""

from __future__ import annotations

import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


@register("model_parallel_strategy", "sequence_parallel")
class SequenceParallelStrategy:
    """Sequence parallelism: split seq dim across TP ranks alongside TP surgery."""

    def __init__(
        self,
        *,
        auto_plan_for: str | None = None,
        plan: list[dict] | None = None,
    ) -> None:
        self.auto_plan_for = auto_plan_for
        self.plan = plan

    def apply(self, model: nn.Module, parallel_ctx: ParallelContext) -> nn.Module:
        from lighttrain.registry import get as _reg_get
        # Delegate to TensorParallelStrategy with sequence_parallel=True.
        tp_cls = _reg_get("model_parallel_strategy", "tensor_parallel")
        tp = tp_cls(
            auto_plan_for=self.auto_plan_for,
            plan=self.plan,
            sequence_parallel=True,
        )
        return tp.apply(model, parallel_ctx)

    def is_stateless(self) -> bool:
        return True


__all__ = ["SequenceParallelStrategy"]
