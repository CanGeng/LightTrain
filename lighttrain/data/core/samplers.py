"""Index samplers."""

from __future__ import annotations

import random
from typing import Any, Iterator, Sized

from ...registry import register


@register("sampler", "sequential")
class SequentialSampler:
    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset
        self._n = len(dataset)
        self._epoch = 0
        # Number of indices already consumed in the current epoch; the next
        # __iter__ skips this many before yielding (mid-epoch resume, BUG-1).
        self._skip = 0

    def __len__(self) -> int:
        return self._n

    def _order(self) -> list[int]:
        return list(range(self._n))

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        skip = self._skip
        self._skip = 0
        for idx in order[skip:]:
            yield idx
        # Advance only when the epoch is fully consumed.
        self._epoch += 1

    def seek(self, epoch: int, consumed_indices: int) -> None:
        """Position for mid-epoch resume: epoch + #indices already consumed."""
        self._epoch = int(epoch)
        self._skip = max(0, int(consumed_indices))

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "skip": self._skip}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self._skip = int(sd.get("skip", 0))


@register("sampler", "shuffle")
class ShuffleSampler:
    """Shuffle indices once per epoch with a deterministic seeded RNG."""

    def __init__(self, dataset: Sized, *, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self._n = len(dataset)
        self._epoch = 0
        self._skip = 0

    def __len__(self) -> int:
        return self._n

    def _order(self) -> list[int]:
        rng = random.Random(self.seed + self._epoch)
        order = list(range(self._n))
        rng.shuffle(order)
        return order

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        skip = self._skip
        self._skip = 0
        for idx in order[skip:]:
            yield idx
        self._epoch += 1

    def seek(self, epoch: int, consumed_indices: int) -> None:
        """Position for mid-epoch resume: epoch + #indices already consumed.

        The order is rebuilt deterministically from ``seed + epoch``, so
        skipping ``consumed_indices`` lands on the exact next index."""
        self._epoch = int(epoch)
        self._skip = max(0, int(consumed_indices))

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "seed": self.seed, "skip": self._skip}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self.seed = int(sd.get("seed", self.seed))
        self._skip = int(sd.get("skip", 0))


__all__ = ["SequentialSampler", "ShuffleSampler"]
