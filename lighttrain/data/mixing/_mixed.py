"""Multi-source dataset mixer.

Three interleave strategies:

* ``round_robin``           — cycle through sources, one row each
* ``weighted``              — sample sources proportional to ``weights``
* ``exhaust_then_resample`` — drain each source in order, restart smaller ones

Both an iterator (``mix_rows``) and a torch-style dataset (``MixedDataset``)
are exposed. The PrepGraph ``mix`` node wraps ``mix_rows`` for offline
materialization.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence


def mix_rows(
    sources: Sequence[Iterable[dict]],
    *,
    strategy: str = "weighted",
    weights: Sequence[float] | None = None,
    temperature: float = 1.0,
    max_samples_per_source: int | None = None,
    max_samples_total: int | None = None,
    seed: int = 0,
) -> Iterator[dict]:
    if not sources:
        return
    n = len(sources)
    weights = list(weights) if weights is not None else [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must match sources length")

    # Apply temperature to weights (T<1 sharpens; T>1 flattens).
    if temperature != 1.0 and temperature > 0:
        weights = [w ** (1.0 / temperature) for w in weights]
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = [w / total_w for w in weights]

    iters: list[Iterator[dict]] = [iter(s) for s in sources]
    counts = [0] * n
    yielded = 0
    rng = random.Random(seed)

    def _take(i: int) -> dict | None:
        if (
            max_samples_per_source is not None
            and counts[i] >= max_samples_per_source
        ):
            return None
        try:
            row = next(iters[i])
            counts[i] += 1
            return row
        except StopIteration:
            return None

    if strategy == "round_robin":
        active = list(range(n))
        while active:
            for i in list(active):
                row = _take(i)
                if row is None:
                    active.remove(i)
                    continue
                yield row
                yielded += 1
                if max_samples_total is not None and yielded >= max_samples_total:
                    return
        return

    if strategy == "exhaust_then_resample":
        for i in range(n):
            while True:
                row = _take(i)
                if row is None:
                    break
                yield row
                yielded += 1
                if max_samples_total is not None and yielded >= max_samples_total:
                    return
        return

    # Default: weighted sampling without replacement of exhausted sources.
    if strategy != "weighted":
        raise ValueError(f"unknown mix strategy: {strategy!r}")
    active = list(range(n))
    active_weights = list(weights)
    while active:
        i = rng.choices(active, weights=active_weights, k=1)[0]
        row = _take(i)
        if row is None:
            idx = active.index(i)
            active.pop(idx)
            active_weights.pop(idx)
            if not active:
                break
            s = sum(active_weights)
            if s <= 0:
                break
            active_weights = [w / s for w in active_weights]
            continue
        yield row
        yielded += 1
        if max_samples_total is not None and yielded >= max_samples_total:
            return


class MixedDataset:
    """Map-style mix dataset — caches a finite materialization of ``mix_rows``."""

    def __init__(
        self,
        sources: Sequence[object],
        *,
        strategy: str = "weighted",
        weights: Sequence[float] | None = None,
        temperature: float = 1.0,
        max_samples_per_source: int | None = None,
        max_samples_total: int | None = None,
        seed: int = 0,
    ) -> None:
        # Each "source" is an iterable yielding sample dicts. To support map
        # access we materialize once at construction.
        iters = [list(_iter_source(s)) for s in sources]
        self._rows = list(
            mix_rows(
                iters,
                strategy=strategy,
                weights=weights,
                temperature=temperature,
                max_samples_per_source=max_samples_per_source,
                max_samples_total=max_samples_total,
                seed=seed,
            )
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[int(idx)]


def _iter_source(s: object) -> Iterable[dict]:
    if hasattr(s, "__iter__"):
        return iter(s)
    raise TypeError(f"Mix source must be iterable, got {type(s).__name__}")


__all__ = ["MixedDataset", "mix_rows"]
