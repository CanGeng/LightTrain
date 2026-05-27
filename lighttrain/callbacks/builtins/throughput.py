"""Throughput tracker (tokens/s, samples/s, step time)."""

from __future__ import annotations

import collections
import time
from typing import Any

from ...registry import register


@register("callback", "throughput")
class ThroughputCallback:
    """Reports rolling-window throughput to the engine context's logger.

    Hooks ``on_step_begin`` to mark t0, ``on_step_end`` to record duration
    and emit ``tokens_per_sec`` / ``samples_per_sec`` / ``step_time_ms`` into
    the metrics dict (the trainer is responsible for forwarding these to the
    LoggerBus when it calls ``log_dict``).
    """

    def __init__(self, window: int = 50) -> None:
        self.window = max(1, int(window))
        self._times: collections.deque[float] = collections.deque(maxlen=self.window)
        self._tokens: collections.deque[int] = collections.deque(maxlen=self.window)
        self._samples: collections.deque[int] = collections.deque(maxlen=self.window)
        self._t0: float | None = None

    def on_step_begin(self, **_: Any) -> None:
        self._t0 = time.perf_counter()

    def on_step_end(self, *, batch: dict | None = None, metrics: dict | None = None,
                    **_: Any) -> None:
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        self._t0 = None
        self._times.append(dt)
        n_samples = 0
        n_tokens = 0
        if batch is not None:
            ids = batch.get("input_ids") if isinstance(batch, dict) else None
            if ids is not None and hasattr(ids, "shape"):
                n_samples = int(ids.shape[0]) if ids.ndim >= 1 else 0
                n_tokens = int(ids.numel()) if hasattr(ids, "numel") else 0
        self._samples.append(n_samples)
        self._tokens.append(n_tokens)
        if metrics is None:
            return
        total_t = sum(self._times) or 1e-9
        metrics["step_time_ms"] = (self._times[-1] * 1000.0)
        metrics["samples_per_sec"] = sum(self._samples) / total_t
        metrics["tokens_per_sec"] = sum(self._tokens) / total_t


__all__ = ["ThroughputCallback"]
