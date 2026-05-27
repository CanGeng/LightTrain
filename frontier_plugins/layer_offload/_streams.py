"""CUDA stream helpers.

On a real CUDA device we keep two streams: ``compute`` (the default stream)
and ``transfer`` (a dedicated copy stream so layer pre-fetches don't block
the running matmuls). On CPU runs we degrade to single-threaded no-ops so
the same engine code path runs everywhere.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class StreamManager:
    def __init__(self, device: torch.device, *, num_streams: int = 2) -> None:
        self.device = device
        self.num_streams = max(1, int(num_streams))
        self._compute = None
        self._transfer = None
        if device.type == "cuda" and torch.cuda.is_available():
            self._compute = torch.cuda.current_stream(device)
            self._transfer = torch.cuda.Stream(device=device) if num_streams > 1 else self._compute

    @property
    def compute(self):
        return self._compute

    @property
    def transfer(self):
        return self._transfer

    @contextmanager
    def on_transfer(self) -> Iterator[None]:
        if self._transfer is not None and self._compute is not None and self._compute is not self._transfer:
            with torch.cuda.stream(self._transfer):
                yield
        else:
            yield

    def sync(self) -> None:
        if self._compute is not None:
            self._compute.synchronize()
        if self._transfer is not None and self._transfer is not self._compute:
            self._transfer.synchronize()


__all__ = ["StreamManager"]
