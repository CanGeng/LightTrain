"""Index samplers."""

from __future__ import annotations

import random
from typing import Any, Sized

from ...registry import register


@register("sampler", "sequential")
class SequentialSampler:
    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset
        self._n = len(dataset)
        self._epoch = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return iter(range(self._n))

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))


@register("sampler", "shuffle")
class ShuffleSampler:
    """Shuffle indices once per epoch with a deterministic seeded RNG."""

    def __init__(self, dataset: Sized, *, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self._n = len(dataset)
        self._epoch = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        order = list(range(self._n))
        rng.shuffle(order)
        self._epoch += 1
        return iter(order)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "seed": self.seed}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self.seed = int(sd.get("seed", self.seed))


__all__ = ["SequentialSampler", "ShuffleSampler"]
