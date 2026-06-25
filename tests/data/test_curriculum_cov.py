"""Edge-case tests for ``lighttrain.builtin_plugins.data.samplers.curriculum``.

Coverage targets (previously uncovered):
* line 52  — ``__len__`` returns ``self._n``
* line 59  — ``_band()`` constant-schedule branch
* lines 64-69 — ``_band()`` step-schedule branch (all four bucket levels + clamp)
* line 69  — ``_band()`` unknown-schedule raises ValueError
* line 86  — ``state_dict()`` structure and values
* lines 93-95 — ``load_state_dict()`` restores epoch / step_hint / seed; missing-key defaults

General edge cases also covered:
* linear schedule: band widens monotonically with progress
* band clamping to [p_start, 1.0]
* ``set_step`` clamps negative inputs to 0
* ``__iter__`` returns a shuffled permutation within the band (deterministic with seed)
* epoch counter increments each ``iter()`` call (changes shuffle order)
* round-trip state_dict / load_state_dict preserves sampler behaviour
* tiny dataset (n=1) does not crash
* dataset whose items lack ``input_ids`` (lengths fall back to 0)
"""

from __future__ import annotations

import pytest

from lighttrain.builtin_plugins.data.samplers.curriculum import CurriculumSampler

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal Sized dataset whose items are dicts with ``input_ids``."""

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = list(lengths)

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, i: int):
        return {"input_ids": [0] * self._lengths[i]}


class _NoInputIdsDataset:
    """Dataset whose items are plain ints (no ``input_ids``), lengths → 0."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        return i


# ---------------------------------------------------------------------------
# __len__  (line 52)
# ---------------------------------------------------------------------------

def test_invariant_len_equals_dataset_size():
    """``__len__`` must equal the number of items in the dataset (line 52)."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(ds, schedule="linear")
    assert len(sampler) == 10


def test_invariant_len_single_item():
    """``__len__`` works for a size-1 dataset."""
    ds = _FakeDataset([5])
    assert len(CurriculumSampler(ds)) == 1


# ---------------------------------------------------------------------------
# constant schedule  (line 59)
# ---------------------------------------------------------------------------

def test_invariant_constant_band_ignores_step():
    """Constant schedule always returns (p_lo, p_hi) regardless of step."""
    ds = _FakeDataset(list(range(1, 21)))
    sampler = CurriculumSampler(ds, schedule="constant", p_lo=0.2, p_hi=0.8)

    for step in (0, 500, 1000):
        sampler.set_step(step)
        assert sampler._band() == (0.2, 0.8)


def test_invariant_constant_band_full_range():
    """Constant schedule with defaults (p_lo=0.0, p_hi=1.0) spans the whole dataset."""
    ds = _FakeDataset(list(range(10)))
    sampler = CurriculumSampler(ds, schedule="constant")
    assert sampler._band() == (0.0, 1.0)


# ---------------------------------------------------------------------------
# step schedule  (lines 64-69)
# ---------------------------------------------------------------------------

def test_invariant_step_schedule_level0_at_start():
    """Step schedule at progress=0 returns band clamped to at least p_start (level 0)."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.25, p_end=1.0, steps=100
    )
    sampler.set_step(0)
    lo, hi = sampler._band()
    # level=0, hi = p_start + 0 = 0.25; clamp: max(0.25, ...) = 0.25
    assert lo == 0.0
    assert pytest.approx(hi, abs=1e-9) == 0.25


def test_invariant_step_schedule_level1():
    """Step schedule at progress ≥ 1/4 advances to level 1."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.0, p_end=1.0, steps=100
    )
    # progress=0.25 → int(0.25 * 4)=1, level=1
    sampler.set_step(25)
    lo, hi = sampler._band()
    assert lo == 0.0
    # hi = 0.0 + 1.0 * (1/3) ≈ 0.3333
    assert pytest.approx(hi, abs=1e-6) == 1 / 3


def test_invariant_step_schedule_level2():
    """Step schedule at progress ≥ 2/4 advances to level 2."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.0, p_end=1.0, steps=100
    )
    sampler.set_step(50)
    lo, hi = sampler._band()
    assert lo == 0.0
    assert pytest.approx(hi, abs=1e-6) == 2 / 3


def test_invariant_step_schedule_max_level_at_completion():
    """Step schedule at progress=1.0 clamps to final level (hi capped at 1.0)."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.0, p_end=1.0, steps=100
    )
    sampler.set_step(100)
    lo, hi = sampler._band()
    assert lo == 0.0
    assert hi == 1.0


def test_invariant_step_schedule_hi_clamped_to_p_start():
    """When level produces hi < p_start, hi is raised to p_start (the max(p_start, hi) clamp)."""
    ds = _FakeDataset(list(range(1, 101)))
    # p_start=0.5, p_end=1.0; at level 0, raw_hi = 0.5 + 0.5*(0/3)=0.5 → no clamp needed
    # Use a weird p_start=0.8 so that even level 0 raw hi would be 0.8, clamp holds
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.8, p_end=1.0, steps=100
    )
    sampler.set_step(0)
    lo, hi = sampler._band()
    assert lo == 0.0
    assert hi == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# unknown schedule raises ValueError  (line 69)
# ---------------------------------------------------------------------------

def test_invariant_unknown_schedule_raises():
    """Unknown schedule string must raise ``ValueError`` with the schedule name."""
    ds = _FakeDataset([1, 2, 3])
    sampler = CurriculumSampler(ds, schedule="cosine")
    with pytest.raises(ValueError, match="cosine"):
        sampler._band()


# ---------------------------------------------------------------------------
# linear schedule band properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("progress_frac,expected_hi", [
    (0.0, 0.25),
    (0.5, 0.625),
    (1.0, 1.0),
])
def test_invariant_linear_band_hi_ramps(progress_frac, expected_hi):
    """Linear schedule hi = p_start + (p_end - p_start) * progress."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="linear", p_start=0.25, p_end=1.0, steps=100
    )
    sampler.set_step(int(progress_frac * 100))
    lo, hi = sampler._band()
    assert lo == 0.0
    assert hi == pytest.approx(expected_hi, abs=1e-6)


def test_invariant_linear_band_clamped_at_1():
    """Linear schedule clamps hi at 1.0 even if p_end > 1.0 were possible."""
    ds = _FakeDataset(list(range(10)))
    # progress > 1.0 → clamped to 1.0 → hi = 1.0
    sampler = CurriculumSampler(
        ds, schedule="linear", p_start=0.1, p_end=1.0, steps=10
    )
    sampler.set_step(9999)
    _, hi = sampler._band()
    assert hi == 1.0


# ---------------------------------------------------------------------------
# set_step clamps negative values
# ---------------------------------------------------------------------------

def test_invariant_set_step_clamps_negative():
    """Negative step values are clamped to 0 by set_step."""
    ds = _FakeDataset([1, 2, 3])
    sampler = CurriculumSampler(ds)
    sampler.set_step(-10)
    assert sampler._step_hint == 0


# ---------------------------------------------------------------------------
# __iter__: returns indices, determinism, epoch increments
# ---------------------------------------------------------------------------

def test_invariant_iter_returns_valid_indices():
    """All returned indices are valid and within the band slice."""
    ds = _FakeDataset(list(range(1, 21)))
    sampler = CurriculumSampler(
        ds, schedule="linear", p_start=0.5, p_end=1.0, steps=10, seed=42
    )
    sampler.set_step(10)  # progress=1.0, band covers top 50%
    indices = list(sampler)
    assert all(0 <= i < 20 for i in indices)
    # with p_start=0.5, at progress=1.0 lo=0 i_lo=0, hi=1.0 i_hi=20 → all 20
    assert len(indices) == 20


def test_invariant_iter_deterministic_same_epoch():
    """Two samplers with the same seed and epoch produce the same order."""
    ds = _FakeDataset(list(range(1, 51)))
    s1 = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=7)
    s2 = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=7)
    assert list(s1) == list(s2)


def test_invariant_iter_epoch_increments():
    """Each call to ``__iter__`` bumps ``_epoch``, changing the shuffle."""
    ds = _FakeDataset(list(range(1, 101)))
    sampler = CurriculumSampler(
        ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=0
    )
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    assert sampler._epoch == 2
    # Different shuffle in epoch 1 (with 100 items this is virtually certain)
    assert epoch0 != epoch1


def test_invariant_iter_is_permutation_of_band():
    """Indices yielded are exactly a permutation of the sorted-slice indices."""
    lengths = list(range(1, 11))   # lengths 1..10
    ds = _FakeDataset(lengths)
    sampler = CurriculumSampler(
        ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=3
    )
    out = list(sampler)
    assert sorted(out) == list(range(10))


def test_invariant_no_input_ids_lengths_zero():
    """Dataset without ``input_ids`` materialises lengths=0; sampler still iterates."""
    ds = _NoInputIdsDataset(5)
    sampler = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0)
    out = list(sampler)
    assert sorted(out) == list(range(5))


def test_invariant_tiny_single_item_dataset():
    """Single-item dataset: iter always yields exactly one index [0]."""
    ds = _FakeDataset([7])
    sampler = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0)
    assert list(sampler) == [0]


# ---------------------------------------------------------------------------
# state_dict  (line 86)
# ---------------------------------------------------------------------------

def test_invariant_state_dict_keys_and_types():
    """state_dict contains exactly 'epoch', 'step_hint', 'seed' with correct values."""
    ds = _FakeDataset(list(range(1, 6)))
    sampler = CurriculumSampler(ds, seed=99)
    sd = sampler.state_dict()
    assert set(sd.keys()) == {"epoch", "step_hint", "seed"}
    assert sd["epoch"] == 0
    assert sd["step_hint"] == 0
    assert sd["seed"] == 99


def test_invariant_state_dict_updates_after_iter_and_set_step():
    """state_dict reflects epoch increments and step_hint updates."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(ds, seed=0)
    list(sampler)          # epoch → 1
    list(sampler)          # epoch → 2
    sampler.set_step(42)
    sd = sampler.state_dict()
    assert sd["epoch"] == 2
    assert sd["step_hint"] == 42
    assert sd["seed"] == 0


# ---------------------------------------------------------------------------
# load_state_dict  (lines 93-95)
# ---------------------------------------------------------------------------

def test_invariant_load_state_dict_restores_epoch_and_step():
    """load_state_dict sets _epoch and _step_hint from the dict."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(ds, seed=5)
    sampler.load_state_dict({"epoch": 7, "step_hint": 100, "seed": 5})
    assert sampler._epoch == 7
    assert sampler._step_hint == 100


def test_invariant_load_state_dict_restores_seed():
    """load_state_dict overrides the sampler's seed (line 95)."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(ds, seed=0)
    sampler.load_state_dict({"epoch": 0, "step_hint": 0, "seed": 42})
    assert sampler.seed == 42


def test_invariant_load_state_dict_missing_keys_use_defaults():
    """Missing keys in state dict use defaults: epoch→0, step_hint→0, seed→current."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(ds, seed=17)
    # load empty dict
    sampler.load_state_dict({})
    assert sampler._epoch == 0
    assert sampler._step_hint == 0
    # seed should fall back to current value (17) since key is absent
    assert sampler.seed == 17


def test_invariant_state_dict_round_trip_yields_same_sequence():
    """Saving and restoring state_dict produces identical iteration output."""
    ds = _FakeDataset(list(range(1, 31)))
    s1 = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=8)
    list(s1)               # advance epoch by 1
    s1.set_step(50)
    sd = s1.state_dict()

    s2 = CurriculumSampler(ds, schedule="constant", p_lo=0.0, p_hi=1.0, seed=999)
    s2.load_state_dict(sd)
    assert list(s1) == list(s2)


# ---------------------------------------------------------------------------
# Parametrized: all schedules produce non-empty output with default settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schedule,kwargs", [
    ("linear",   {"p_start": 0.0, "p_end": 1.0, "steps": 10}),
    ("constant", {"p_lo": 0.0, "p_hi": 1.0}),
    ("step",     {"p_start": 0.0, "p_end": 1.0, "steps": 10}),
])
def test_invariant_all_schedules_produce_indices(schedule, kwargs):
    """Every schedule name yields at least one index for a non-empty dataset."""
    ds = _FakeDataset(list(range(1, 21)))
    sampler = CurriculumSampler(ds, schedule=schedule, **kwargs)
    sampler.set_step(5)
    out = list(sampler)
    assert len(out) > 0
    assert all(0 <= i < 20 for i in out)


# ---------------------------------------------------------------------------
# step schedule: progress > 1.0 is clamped via min(1.0, ...)
# ---------------------------------------------------------------------------

def test_invariant_step_schedule_progress_clamped_past_steps():
    """Step at a value far beyond `steps` still returns a valid (clamped) band."""
    ds = _FakeDataset(list(range(1, 11)))
    sampler = CurriculumSampler(
        ds, schedule="step", p_start=0.0, p_end=1.0, steps=10
    )
    sampler.set_step(99999)
    lo, hi = sampler._band()
    assert lo == 0.0
    assert hi == 1.0
