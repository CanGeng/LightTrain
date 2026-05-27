"""Stateful resumable sampler.

Iterates indices in sub-chunks so that resuming mid-epoch is precise: state
records ``epoch``, ``chunk_idx``, and ``consumed_in_chunk``. After
``load_state_dict``, ``__iter__`` skips that many indices in the current chunk
before continuing.

Also serves as the base for stateful-architecture (RWKV/Mamba) samplers where
chunk boundaries are where the model can call ``state_reset()``.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Iterator, Sized


class StatefulResumableSampler:
    """Yield indices in fixed-size chunks; resumable down to an offset."""

    def __init__(
        self,
        dataset: Sized,
        *,
        chunk_size: int = 1024,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.chunk_size = max(1, int(chunk_size))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._n = len(dataset)
        self._epoch = 0
        self._chunk_idx = 0
        self._consumed_in_chunk = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        order = list(range(self._n))
        if self.shuffle:
            rng.shuffle(order)
        chunks = [
            order[i : i + self.chunk_size]
            for i in range(0, self._n, self.chunk_size)
        ]
        return self._walk(chunks)

    def _walk(self, chunks: list[list[int]]) -> Iterator[int]:
        # Resume into the saved chunk + offset.
        for ci in range(self._chunk_idx, len(chunks)):
            chunk = chunks[ci]
            start = self._consumed_in_chunk if ci == self._chunk_idx else 0
            for j in range(start, len(chunk)):
                self._chunk_idx = ci
                self._consumed_in_chunk = j + 1
                yield chunk[j]
            self._chunk_idx = ci + 1
            self._consumed_in_chunk = 0
        # End of epoch.
        self._epoch += 1
        self._chunk_idx = 0
        self._consumed_in_chunk = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "chunk_idx": self._chunk_idx,
            "consumed_in_chunk": self._consumed_in_chunk,
            "seed": self.seed,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self._chunk_idx = int(sd.get("chunk_idx", 0))
        self._consumed_in_chunk = int(sd.get("consumed_in_chunk", 0))
        self.seed = int(sd.get("seed", self.seed))

    def chunk_boundaries(self) -> Iterable[int]:
        """Indices where state_reset() should be called (chunk starts)."""
        return range(0, self._n, self.chunk_size)


__all__ = ["StatefulResumableSampler"]
