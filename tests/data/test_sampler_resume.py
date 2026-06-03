"""BUG-1: mid-epoch resume must be step-exact and prefetch-independent.

The default samplers (`sequential`, `shuffle`) and the `stateful_resumable`
sampler all resume from an authoritative *consumed-index* count via ``seek``.
Because the count comes from the training loop (batches actually consumed), not
from the sampler's yield position, resume stays exact under DataLoader prefetch
(``num_workers > 0``) — the v0.1.8 epoch-granularity drift is gone.
"""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, Dataset

from lighttrain.builtin_plugins.data.core.samplers import (
    SequentialSampler,
    ShuffleSampler,
)
from lighttrain.builtin_plugins.data.samplers.stateful_resumable import (
    StatefulResumableSampler,
)


def _samplers(n: int):
    return [
        ("sequential", SequentialSampler(list(range(n)))),
        ("shuffle", ShuffleSampler(list(range(n)), seed=7)),
        ("stateful_noshuffle", StatefulResumableSampler(list(range(n)), chunk_size=4, shuffle=False)),
        ("stateful_shuffle", StatefulResumableSampler(list(range(n)), chunk_size=4, seed=7, shuffle=True)),
    ]


@pytest.mark.parametrize("name,_factory_idx", [(n, i) for i, (n, _) in enumerate(_samplers(20))])
@pytest.mark.parametrize("consumed", [0, 1, 5, 7, 13])
def test_seek_resumes_exact_suffix_of_epoch_order(name, _factory_idx, consumed):
    """seek(epoch=0, consumed) → next iter yields exactly the order's suffix
    after the first ``consumed`` indices. This is the core resume invariant,
    independent of how the sampler is driven."""
    n = 20
    sampler = _samplers(n)[_factory_idx][1]

    # The full epoch-0 order (a fresh, identically-seeded sampler).
    reference = _samplers(n)[_factory_idx][1]
    full = list(iter(reference))

    sampler.seek(0, consumed)
    resumed = list(iter(sampler))
    assert resumed == full[consumed:]


@pytest.mark.parametrize("name,idx", [(n, i) for i, (n, _) in enumerate(_samplers(20))])
def test_split_run_equals_single_run_via_seek(name, idx):
    """A run split at an arbitrary mid-epoch point and resumed via ``seek``
    yields the same total sequence as an uninterrupted run (step-exact)."""
    n = 20
    boundary = 7

    single = list(iter(_samplers(n)[idx][1]))

    s = _samplers(n)[idx][1]
    phase1 = []
    for k, i in enumerate(iter(s)):
        phase1.append(i)
        if k + 1 == boundary:
            break
    # Resume a fresh sampler from the authoritative consumed count.
    s2 = _samplers(n)[idx][1]
    s2.seek(0, boundary)
    phase2 = list(iter(s2))

    assert phase1 + phase2 == single


class _RangeDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


@pytest.mark.parametrize("num_workers", [0, 2])
def test_seek_is_prefetch_independent_with_real_dataloader(num_workers):
    """With ``num_workers>0`` the DataLoader prefetches, so the sampler yields
    further than the consumer has taken. Resuming from the *consumed* count
    (not the yielded count) still produces the exact continuation — proving the
    fix is prefetch-independent, not just correct at num_workers=0.
    """
    n = 40
    batch_size = 2
    consumed_batches = 5  # consumer stops here; prefetch has run further

    # Uninterrupted reference sequence of indices.
    ref_sampler = SequentialSampler(list(range(n)))
    ref_loader = DataLoader(
        _RangeDataset(n), batch_size=batch_size, sampler=ref_sampler, num_workers=num_workers
    )
    reference = [int(x) for batch in ref_loader for x in batch]

    # Phase 1: consume only `consumed_batches` batches, then stop (leaving the
    # prefetch buffer ahead of us).
    p1_sampler = SequentialSampler(list(range(n)))
    p1_loader = DataLoader(
        _RangeDataset(n), batch_size=batch_size, sampler=p1_sampler, num_workers=num_workers
    )
    taken: list[int] = []
    for k, batch in enumerate(p1_loader):
        taken.extend(int(x) for x in batch)
        if k + 1 == consumed_batches:
            break
    del p1_loader  # tear down workers

    # Phase 2: resume a fresh sampler from the authoritative consumed count.
    p2_sampler = SequentialSampler(list(range(n)))
    p2_sampler.seek(0, consumed_batches * batch_size)
    p2_loader = DataLoader(
        _RangeDataset(n), batch_size=batch_size, sampler=p2_sampler, num_workers=num_workers
    )
    rest = [int(x) for batch in p2_loader for x in batch]

    assert taken + rest == reference
    assert len(taken) == consumed_batches * batch_size


def test_stateful_seek_matches_yield_position_at_num_workers_0():
    """R7 guard: at num_workers=0, seek(epoch, consumed) lands on the same
    chunk position the stateful sampler tracks via its own yield-time counters
    — so the existing bit-exact resume path is preserved, not regressed."""
    n = 20
    s = StatefulResumableSampler(list(range(n)), chunk_size=4, seed=7, shuffle=True)
    # Consume 6 indices via natural iteration (yield-time tracking).
    it = iter(s)
    for _ in range(6):
        next(it)
    yield_state = s.state_dict()

    # A fresh sampler positioned purely by seek(6).
    s2 = StatefulResumableSampler(list(range(n)), chunk_size=4, seed=7, shuffle=True)
    s2.seek(0, 6)
    seek_state = s2.state_dict()

    assert (yield_state["chunk_idx"], yield_state["consumed_in_chunk"]) == (
        seek_state["chunk_idx"],
        seek_state["consumed_in_chunk"],
    )
