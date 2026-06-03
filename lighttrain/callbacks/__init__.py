"""Callbacks — the ``CallbackProtocol`` + ``EventBus`` core seam.

Concrete builtin callbacks (ema / best_ckpt / early_stop / throughput / ...) and
the invariants callback are registered impls living in
``lighttrain.builtin_plugins.callbacks`` (DESIGN §3.3).
"""

from __future__ import annotations

from .base import CALLBACK_EVENTS, EventBus, Signal

__all__ = ["CALLBACK_EVENTS", "EventBus", "Signal"]
