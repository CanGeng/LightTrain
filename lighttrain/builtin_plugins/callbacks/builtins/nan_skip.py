"""Skip / abort on NaN/Inf loss."""

from __future__ import annotations

import math
from typing import Any

from lighttrain.callbacks.base import Signal
from lighttrain.registry import register


@register("callback", "nan_skip")
class NaNSkipCallback:
    """Returns ``SKIP_STEP`` when the loss is non-finite; aborts after N skips."""

    def __init__(self, max_skips: int = 10) -> None:
        self.max_skips = int(max_skips)
        self.skipped = 0

    def on_loss_computed(self, *, loss: Any = None, **_: Any) -> Any:
        if loss is None:
            return None
        try:
            value = float(loss.item()) if hasattr(loss, "item") else float(loss)
        except (TypeError, ValueError):
            return None
        if math.isfinite(value):
            return None
        self.skipped += 1
        if self.skipped > self.max_skips:
            return Signal.STOP_TRAINING
        return Signal.SKIP_STEP


__all__ = ["NaNSkipCallback"]
