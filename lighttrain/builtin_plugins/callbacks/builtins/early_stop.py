"""Early-stop on a monitored metric."""

from __future__ import annotations

import math
from typing import Any

from lighttrain.callbacks.base import Signal
from lighttrain.registry import register


@register("callback", "early_stop")
class EarlyStopCallback:
    """Returns ``STOP_TRAINING`` after ``patience`` non-improvements."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float = math.inf if mode == "min" else -math.inf
        self.bad_epochs = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def _check(self, metrics: dict | None) -> Any:
        if not metrics or self.monitor not in metrics:
            return None
        try:
            value = float(metrics[self.monitor])
        except (TypeError, ValueError):
            return None
        if self._improved(value):
            self.best = value
            self.bad_epochs = 0
            return None
        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            return Signal.STOP_TRAINING
        return None

    def on_eval_end(self, *, metrics: dict | None = None, **_: Any) -> Any:
        return self._check(metrics)

    def on_epoch_end(self, *, metrics: dict | None = None, **_: Any) -> Any:
        return self._check(metrics)


__all__ = ["EarlyStopCallback"]
