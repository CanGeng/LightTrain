"""Logger Protocol + backends.

Public surface:
    - LoggerBus: aggregator with backend isolation.
    - ConsoleLogger / JSONLLogger / TensorBoardLogger: registered under
      category ``logger`` (short names: ``console`` / ``jsonl`` /
      ``tensorboard`` or ``tb``).
"""

from __future__ import annotations

from ._bus import LoggerBus
from .backends import ConsoleLogger, JSONLLogger, TensorBoardLogger

__all__ = ["ConsoleLogger", "JSONLLogger", "LoggerBus", "TensorBoardLogger"]
