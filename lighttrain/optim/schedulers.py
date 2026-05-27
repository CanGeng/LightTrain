"""Per-step LR schedulers.

All schedulers expose ``step()``, ``state_dict()``, ``load_state_dict()``, and
the boolean ``step_per_batch`` flag (always ``True`` — they tick once per
optimizer step).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..registry import register


class _SchedulerBase:
    step_per_batch: bool = True

    def __init__(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        self.optimizer: torch.optim.Optimizer | None = optimizer
        self.last_step: int = 0
        self._base_lrs: list[float] = []
        if optimizer is not None:
            self.attach(optimizer)

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._base_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]

    def _set_lrs(self, factor: float) -> None:
        if self.optimizer is None or not self._base_lrs:
            return
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base * factor

    def step(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.last_step += 1
        self._set_lrs(self._factor(self.last_step))

    def _factor(self, step: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_lr(self) -> list[float]:
        if self.optimizer is None:
            return []
        return [g.get("lr", 0.0) for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "base_lrs": list(self._base_lrs)}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.last_step = int(sd.get("last_step", 0))
        if "base_lrs" in sd:
            self._base_lrs = list(sd["base_lrs"])


@register("scheduler", "constant")
class ConstantScheduler(_SchedulerBase):
    def __init__(self, optimizer: torch.optim.Optimizer | None = None, **_: Any) -> None:
        super().__init__(optimizer)

    def _factor(self, step: int) -> float:  # noqa: ARG002
        return 1.0


@register("scheduler", "linear")
class LinearScheduler(_SchedulerBase):
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
class WarmupCosineScheduler(_SchedulerBase):
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
class WSDScheduler(_SchedulerBase):
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
