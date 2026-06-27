"""Curriculum sampler — schedule difficulty over training.

Difficulty here is the integer ``len(input_ids)`` (a cheap proxy that works
without per-sample annotations). Three schedules:

* ``linear``  — start at percentile p_start, ramp to p_end
* ``step``    — staircase over fixed step boundaries
* ``constant``— always within [p_lo, p_hi]

The sampler maintains an external ``step`` counter set by the trainer (or
re-derived from epoch when no signal is available).
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

from .length_grouped import _materialize_lengths


class CurriculumSampler:
    """Iterate indices whose length lies within an evolving percentile band."""

    def __init__(
        self,
        dataset: Sized,
        *,
        schedule: str = "linear",
        p_start: float = 0.25,
        p_end: float = 1.0,
        steps: int = 1_000,
        p_lo: float = 0.0,
        p_hi: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.schedule = schedule
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.steps = max(1, int(steps))
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.seed = int(seed)
        self._n = len(dataset)
        self._epoch = 0
        self._step_hint = 0
        self._lengths = _materialize_lengths(dataset)
        self._sorted = sorted(range(self._n), key=lambda i: self._lengths[i])

    def __len__(self) -> int:
        return self._n

    def set_step(self, step: int) -> None:
        self._step_hint = max(0, int(step))

    def _band(self) -> tuple[float, float]:
        if self.schedule == "constant":
            return self.p_lo, self.p_hi
        progress = min(1.0, self._step_hint / max(1, self.steps))
        if self.schedule == "linear":
            hi = self.p_start + (self.p_end - self.p_start) * progress
            return 0.0, max(self.p_start, min(1.0, hi))
        if self.schedule == "step":
            buckets = 4
            level = min(buckets - 1, int(progress * buckets))
            hi = self.p_start + (self.p_end - self.p_start) * (level / (buckets - 1))
            return 0.0, max(self.p_start, min(1.0, hi))
        raise ValueError(f"unknown curriculum schedule: {self.schedule!r}")

    def __iter__(self) -> Iterator[int]:
        lo, hi = self._band()
        n = self._n
        i_lo = max(0, int(lo * n))
        i_hi = min(n, max(i_lo + 1, int(hi * n)))
        keep = self._sorted[i_lo:i_hi]
        # Permute within the band for stochasticity.
        import random

        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(keep)
        self._epoch += 1
        return iter(keep)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "step_hint": self._step_hint,
            "seed": self.seed,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._epoch = int(sd.get("epoch", 0))
        self._step_hint = int(sd.get("step_hint", 0))
        self.seed = int(sd.get("seed", self.seed))


__all__ = ["CurriculumSampler"]
