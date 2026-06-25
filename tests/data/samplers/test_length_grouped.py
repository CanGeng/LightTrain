"""Edge-case unit tests for ``lighttrain.builtin_plugins.data.samplers.length_grouped``.

Coverage targets (uncovered lines driving to 100 %):
  23-28  _length_of: non-dict fallback branches — __len__, TypeError raise, return 0
  61     LengthGroupedSampler.__len__
  79     LengthGroupedSampler.state_dict
  82-83  LengthGroupedSampler.load_state_dict (epoch + seed restore)

General edge cases covered beyond the uncovered lines:
  - Permutation completeness (every index yielded exactly once per epoch)
  - Sort order within mega-batch blocks (descending and ascending)
  - Multi-epoch determinism (seed + epoch combine correctly)
  - Clamping: batch_size and mega_batch_mult < 1 clamp to 1
  - state_dict/load_state_dict round-trip
  - Empty dataset (n=0)
  - Single-element dataset
  - mega_batch larger than dataset (one chunk)
  - _length_of: dict with input_ids, dict without input_ids, bare list, object with
    __len__, object whose __len__ raises TypeError, plain int (no __len__)
"""

from __future__ import annotations

import pytest

from lighttrain.builtin_plugins.data.samplers.length_grouped import (
    LengthGroupedSampler,
    _length_of,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _ListLike:
    """Has __len__; represents a non-dict sequence sample."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


class _BadLen:
    """Has __len__ but raises TypeError — exercises the except branch."""

    def __len__(self) -> int:  # noqa: PYI034
        raise TypeError("deliberate")


class _NoLen:
    """No __len__ and not a dict — _length_of must return 0."""

    pass


class _SimpleDataset:
    """Minimal Sized dataset whose items are dicts with ``input_ids``."""

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = lengths

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": [0] * self._lengths[i]}


class _RawDataset:
    """Dataset whose items are plain lists (not dicts)."""

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = lengths

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, i: int) -> list:
        return [0] * self._lengths[i]


# ---------------------------------------------------------------------------
# _length_of — unit-level coverage of lines 18-28
# ---------------------------------------------------------------------------


def test_invariant_length_of_dict_with_input_ids():
    """dict with 'input_ids' → length of the list (lines 19-22)."""
    assert _length_of({"input_ids": [1, 2, 3]}) == 3


def test_invariant_length_of_dict_without_input_ids_falls_through():
    """dict missing 'input_ids' key falls through to the __len__ branch (line 23)."""
    d = {"tokens": [1, 2]}
    # dict has no __len__ that returns semantic length here — but dict itself
    # has __len__ (number of keys), so we get 1 (one key: 'tokens').
    assert _length_of(d) == 1


def test_invariant_length_of_list_like_object():
    """Non-dict with __len__ → len() result (lines 23-25)."""
    obj = _ListLike(7)
    assert _length_of(obj) == 7


def test_invariant_length_of_bad_len_returns_zero():
    """Non-dict whose __len__ raises TypeError → 0 (lines 23-28, except branch)."""
    obj = _BadLen()
    assert _length_of(obj) == 0


def test_invariant_length_of_no_len_returns_zero():
    """Object without __len__ and not a dict → 0 (line 28, fallback)."""
    obj = _NoLen()
    assert _length_of(obj) == 0


def test_invariant_length_of_plain_list():
    """Plain list falls through dict check; its __len__ gives correct length."""
    assert _length_of([10, 20, 30]) == 3


def test_invariant_length_of_empty_list():
    """Empty list → 0."""
    assert _length_of([]) == 0


def test_invariant_length_of_empty_input_ids():
    """dict with empty 'input_ids' → 0."""
    assert _length_of({"input_ids": []}) == 0


def test_invariant_length_of_none_input_ids_falls_through():
    """dict where input_ids is None: get() returns None, branch skipped (line 21)."""
    # sample.get("input_ids") is None → falls through to __len__ on the dict.
    # dict{"input_ids": None} has 1 key.
    assert _length_of({"input_ids": None}) == 1


# ---------------------------------------------------------------------------
# LengthGroupedSampler.__len__ — line 61
# ---------------------------------------------------------------------------


def test_invariant_len_equals_dataset_size():
    """``len(sampler)`` returns the number of dataset items (line 61)."""
    ds = _SimpleDataset([1, 2, 3, 4, 5])
    s = LengthGroupedSampler(ds, batch_size=2)
    assert len(s) == 5


def test_invariant_len_empty_dataset():
    """``len(sampler)`` is 0 for an empty dataset."""
    ds = _SimpleDataset([])
    s = LengthGroupedSampler(ds, batch_size=2)
    assert len(s) == 0


# ---------------------------------------------------------------------------
# state_dict — line 79
# ---------------------------------------------------------------------------


def test_invariant_state_dict_initial():
    """state_dict returns epoch=0 and the configured seed before any iteration (line 79)."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=42)
    sd = s.state_dict()
    assert sd == {"epoch": 0, "seed": 42}


def test_invariant_state_dict_after_one_epoch():
    """After one iteration epoch advances to 1; state_dict reflects that."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=7)
    list(iter(s))
    sd = s.state_dict()
    assert sd["epoch"] == 1
    assert sd["seed"] == 7


def test_invariant_state_dict_after_multiple_epochs():
    """state_dict epoch increments with every iteration."""
    ds = _SimpleDataset([10, 20])
    s = LengthGroupedSampler(ds, batch_size=1, seed=0)
    for _ in range(3):
        list(iter(s))
    assert s.state_dict()["epoch"] == 3


# ---------------------------------------------------------------------------
# load_state_dict — lines 82-83
# ---------------------------------------------------------------------------


def test_invariant_load_state_dict_restores_epoch(  ):
    """load_state_dict sets _epoch from the saved dict (line 82)."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=0)
    s.load_state_dict({"epoch": 5, "seed": 0})
    assert s._epoch == 5


def test_invariant_load_state_dict_restores_seed():
    """load_state_dict sets seed from the saved dict (line 83)."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=0)
    s.load_state_dict({"epoch": 0, "seed": 99})
    assert s.seed == 99


def test_invariant_load_state_dict_missing_epoch_defaults_to_zero():
    """load_state_dict with missing 'epoch' key uses default 0 (line 82)."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=5)
    s._epoch = 7  # simulate mid-run state
    s.load_state_dict({"seed": 5})  # epoch absent
    assert s._epoch == 0


def test_invariant_load_state_dict_missing_seed_keeps_current():
    """load_state_dict with missing 'seed' key preserves the current seed (line 83)."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=1, seed=11)
    s.load_state_dict({"epoch": 2})  # seed absent
    assert s.seed == 11


def test_invariant_state_dict_round_trip():
    """state_dict + load_state_dict restores exact sampler state."""
    ds = _SimpleDataset([3, 1, 4, 1, 5])
    s = LengthGroupedSampler(ds, batch_size=2, seed=17)
    list(iter(s))  # advance epoch to 1
    sd = s.state_dict()

    s2 = LengthGroupedSampler(ds, batch_size=2, seed=0)
    s2.load_state_dict(sd)
    assert s2._epoch == sd["epoch"]
    assert s2.seed == sd["seed"]
    # Both should now produce identical epoch-1 outputs.
    assert list(iter(s)) == list(iter(s2))


# ---------------------------------------------------------------------------
# Permutation completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,batch_size,mega_mult", [
    (8, 2, 2),
    (10, 3, 1),
    (1, 1, 1),
    (20, 5, 3),
])
def test_invariant_permutation_complete(n, batch_size, mega_mult):
    """Every index is yielded exactly once per epoch."""
    lengths = list(range(n))
    ds = _SimpleDataset(lengths)
    s = LengthGroupedSampler(ds, batch_size=batch_size, mega_batch_mult=mega_mult, seed=0)
    out = list(iter(s))
    assert sorted(out) == list(range(n))


# ---------------------------------------------------------------------------
# Sort order within mega-batch blocks
# ---------------------------------------------------------------------------


def test_invariant_descending_order_within_block():
    """Items within each mega-batch block are sorted descending by length."""
    lengths = [1, 8, 3, 5, 2, 9, 4, 6]
    ds = _SimpleDataset(lengths)
    # mega = 2 * 2 = 4 items per block
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=2, descending=True, seed=0)
    out = list(iter(s))
    # Verify each block of 4 is descending by length.
    for start in range(0, len(lengths), 4):
        block = out[start : start + 4]
        block_lens = [lengths[i] for i in block]
        assert block_lens == sorted(block_lens, reverse=True)


def test_invariant_ascending_order_within_block():
    """When descending=False, items within each block are sorted ascending."""
    lengths = [1, 8, 3, 5, 2, 9, 4, 6]
    ds = _SimpleDataset(lengths)
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=2, descending=False, seed=0)
    out = list(iter(s))
    for start in range(0, len(lengths), 4):
        block = out[start : start + 4]
        block_lens = [lengths[i] for i in block]
        assert block_lens == sorted(block_lens)


# ---------------------------------------------------------------------------
# Multi-epoch determinism
# ---------------------------------------------------------------------------


def test_invariant_different_epochs_different_orders():
    """Two successive epochs produce different orderings (seed+epoch combine).

    Uses multiple small mega-batches so that the within-epoch shuffle of chunk
    order actually varies between epochs.  With mega_batch_mult=1, mega=2 per
    epoch, so 10 chunks of size 2 each — the shuffle reorders those chunks
    differently each epoch.
    """
    lengths = list(range(1, 21))
    ds = _SimpleDataset(lengths)
    # mega_batch_mult=1, batch_size=2 → mega=2 → 10 chunks of size 2
    # epoch seed changes each iteration, so chunk assignment varies.
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=1, seed=42)
    epoch0 = list(iter(s))
    epoch1 = list(iter(s))
    # With 10 independently-shuffled chunks across two different RNG seeds,
    # the probability of an identical ordering is negligible.
    assert epoch0 != epoch1


def test_invariant_same_seed_same_epoch_reproducible():
    """Two identically-configured samplers produce the same epoch-0 sequence."""
    lengths = [5, 1, 9, 2, 7]
    ds = _SimpleDataset(lengths)
    s1 = LengthGroupedSampler(ds, batch_size=2, seed=3)
    s2 = LengthGroupedSampler(ds, batch_size=2, seed=3)
    assert list(iter(s1)) == list(iter(s2))


# ---------------------------------------------------------------------------
# Clamping of batch_size and mega_batch_mult
# ---------------------------------------------------------------------------


def test_invariant_batch_size_clamped_to_one():
    """batch_size <= 0 is clamped to 1; sampler still yields all indices."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=0)
    assert s.batch_size == 1
    assert sorted(list(iter(s))) == [0, 1, 2]


def test_invariant_mega_batch_mult_clamped_to_one():
    """mega_batch_mult <= 0 is clamped to 1; sampler still yields all indices."""
    ds = _SimpleDataset([1, 2, 3])
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=0)
    assert s.mega_batch_mult == 1
    assert sorted(list(iter(s))) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Edge: mega_batch larger than entire dataset (single chunk)
# ---------------------------------------------------------------------------


def test_invariant_single_chunk_when_mega_exceeds_dataset():
    """When mega_batch > n, the whole dataset is one chunk sorted by length."""
    lengths = [3, 1, 4, 1, 5, 9, 2, 6]
    ds = _SimpleDataset(lengths)
    # batch_size=100, mega_batch_mult=10 → mega=1000 >> 8
    s = LengthGroupedSampler(ds, batch_size=100, mega_batch_mult=10, descending=True, seed=0)
    out = list(iter(s))
    out_lens = [lengths[i] for i in out]
    assert out_lens == sorted(out_lens, reverse=True)


# ---------------------------------------------------------------------------
# Non-dict dataset items (raw lists)
# ---------------------------------------------------------------------------


def test_invariant_raw_list_items_measured_correctly():
    """Dataset items that are plain lists use __len__ for length measurement."""
    ds = _RawDataset([3, 7, 1, 5])
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=2, descending=True, seed=0)
    out = list(iter(s))
    assert sorted(out) == [0, 1, 2, 3]
