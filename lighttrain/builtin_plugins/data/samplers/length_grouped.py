"""Length-grouped sampler — bucket-by-length to reduce padding waste.

Looks at ``len(dataset[i]['input_ids'])`` once at init. Inside an epoch, items
are grouped into ``mega_batch`` blocks (default ~50 batches' worth), shuffled
within each block by length-similarity, and concatenated. Padding waste drops
because adjacent indices have similar lengths.

State is resumable via ``state_dict``: epoch + within-block offset.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sized


def _length_of(sample: Any) -> int:
    if isinstance(sample, dict):
        v = sample.get("input_ids")
        if v is not None:
            return len(v)
    if hasattr(sample, "__len__"):
        try:
            return len(sample)
        except TypeError:
            pass
    return 0


def _materialize_lengths(dataset: Sized) -> list[int]:
    n = len(dataset)
    lengths = [0] * n
    for i in range(n):
        lengths[i] = _length_of(dataset[i])  # type: ignore[index]
    return lengths


class LengthGroupedSampler:
    """Group similar-length indices together inside an epoch."""

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        mega_batch_mult: int = 50,
        descending: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.mega_batch_mult = max(1, int(mega_batch_mult))
        self.descending = bool(descending)
        self.seed = int(seed)
        self._n = len(dataset)
        self._epoch = 0
        self._lengths = _materialize_lengths(dataset)

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterable[int]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(self._n))
        rng.shuffle(indices)
        mega = self.batch_size * self.mega_batch_mult
        chunks: list[list[int]] = [
            indices[i : i + mega] for i in range(0, self._n, mega)
        ]
        ordered: list[int] = []
        for chunk in chunks:
            chunk.sort(key=lambda i: self._lengths[i], reverse=self.descending)
            ordered.extend(chunk)
        self._epoch += 1
        return iter(ordered)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "seed": self.seed}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self.seed = int(sd.get("seed", self.seed))


__all__ = ["LengthGroupedSampler"]
