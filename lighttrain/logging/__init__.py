"""Logging framework — ``LoggerBus`` aggregator (the ``LoggerProtocol`` is in
``lighttrain.protocols``).

Concrete logger backends (console / jsonl / tensorboard) are registered impls
living in ``lighttrain.builtin_plugins.logging_backends`` (DESIGN §3.3).
"""

from __future__ import annotations

from ._bus import LoggerBus

__all__ = ["LoggerBus"]
