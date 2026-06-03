"""Per-step LR schedulers (concrete impls).

All schedulers subclass ``SchedulerBase`` (``lighttrain.optim.base``, core) and
implement ``_factor(step)``. They expose ``step() / state_dict() /
load_state_dict()`` and the ``step_per_batch`` flag (always ``True`` — they tick
once per optimizer step).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from lighttrain.optim.base import SchedulerBase
from lighttrain.registry import register


@register("scheduler", "constant")
class ConstantScheduler(SchedulerBase):
    def __init__(self, optimizer: torch.optim.Optimizer | None = None, **_: Any) -> None:
        super().__init__(optimizer)

    def _factor(self, step: int) -> float:  # noqa: ARG002
        return 1.0


@register("scheduler", "linear")
class LinearScheduler(SchedulerBase):
    """Linear decay from 1.0 → ``end_factor`` over ``total_steps``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        total_steps: int = 1000,
        end_factor: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__(optimizer)
        self.total_steps = max(1, int(total_steps))
        self.end_factor = float(end_factor)
        self.warmup_steps = max(0, int(warmup_steps))

    def _factor(self, step: int) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 + (self.end_factor - 1.0) * progress


@register("scheduler", "warmup_cosine")
class WarmupCosineScheduler(SchedulerBase):
    """Linear warmup, then cosine decay to ``min_lr_ratio``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__(optimizer)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)

    def _factor(self, step: int) -> float:
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


@register("scheduler", "wsd")
class WSDScheduler(SchedulerBase):
    """Warmup → Stable → linear Decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        warmup_steps: int = 100,
        stable_steps: int = 700,
        decay_steps: int = 200,
        min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__(optimizer)
        self.warmup_steps = max(0, int(warmup_steps))
        self.stable_steps = max(0, int(stable_steps))
        self.decay_steps = max(1, int(decay_steps))
        self.min_lr_ratio = float(min_lr_ratio)

    def _factor(self, step: int) -> float:
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return step / self.warmup_steps
        s = step - self.warmup_steps
        if s <= self.stable_steps:
            return 1.0
        d = s - self.stable_steps
        progress = min(d / self.decay_steps, 1.0)
        return 1.0 + (self.min_lr_ratio - 1.0) * progress


__all__ = [
    "ConstantScheduler",
    "LinearScheduler",
    "WSDScheduler",
    "WarmupCosineScheduler",
]
