"""Best-checkpoint callback — tracks the best monitored metric & flags it."""

from __future__ import annotations

import math
from typing import Any

from ...registry import register


@register("callback", "best_ckpt")
class BestCheckpointCallback:
    """When a new best ``monitor`` value is seen, sets a flag the trainer reads.

    The trainer checks ``cb.should_save`` after dispatching ``on_eval_end``
    and, if true, calls ``ckpt_mgr.save(... kind='best')``. Keeping the
    actual save in the trainer avoids passing the manager through events.
    """

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float = math.inf if mode == "min" else -math.inf
        self.should_save: bool = False
        self.last_value: float | None = None

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def _consume(self, metrics: dict | None) -> None:
        self.should_save = False
        if not metrics or self.monitor not in metrics:
            return
        try:
            value = float(metrics[self.monitor])
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.last_value = value
        if self._improved(value):
            self.best = value
            self.should_save = True

    def on_eval_end(self, *, metrics: dict | None = None, **_: Any) -> None:
        self._consume(metrics)

    def on_step_end(self, *, metrics: dict | None = None, **_: Any) -> None:
        # Allow tracking a streaming train metric when no eval is configured.
        if self.monitor and metrics and self.monitor in metrics:
            self._consume(metrics)


__all__ = ["BestCheckpointCallback"]
