"""Edge-case tests for ``lighttrain.data.mixing`` (``mix_rows`` + ``MixedDataset``).

Covers every branch of the multi-source mixer:

* **Guards**: empty ``sources`` early-return; ``weights`` length mismatch;
  zero/negative total weight; unknown strategy; non-iterable source.
* **Temperature**: applied when ``!= 1.0 and > 0``; skipped for ``T == 1`` and
  ``T <= 0``.
* **Per-source / total caps**: ``max_samples_per_source`` and
  ``max_samples_total`` enforced across all strategies.
* **round_robin**: cycle one row each, drop exhausted sources.
* **exhaust_then_resample**: drain sources in order.
* **weighted**: seeded sampling, exhausted-source eviction + renormalization,
  zero-remaining-weight break, last-source break.
* **MixedDataset**: eager materialization, ``__len__``, int-coerced
  ``__getitem__``.

``weighted`` is RNG-driven, so order assertions there use a fixed ``seed=`` or
assert the preserved multiset (mixing without caps drains every source).
"""

from __future__ import annotations

import pytest

from lighttrain.data.mixing import MixedDataset, mix_rows


def _src(*ids: str) -> list[dict]:
    """A source = list of ``{"id": ...}`` rows."""
    return [{"id": x} for x in ids]


def _ids(rows) -> list[str]:
    return [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_invariant_empty_sources_yields_nothing():
    """No sources → empty iterator (early return)."""
    assert list(mix_rows([])) == []


def test_invariant_weights_length_mismatch_raises():
    """``weights`` length must equal ``sources`` length."""
    with pytest.raises(ValueError, match="weights length must match"):
        list(mix_rows([_src("a"), _src("b")], weights=[1.0]))


def test_invariant_zero_total_weight_raises():
    """All-zero weights → positive-sum guard fires."""
    with pytest.raises(ValueError, match="must sum to a positive value"):
        list(mix_rows([_src("a"), _src("b")], weights=[0.0, 0.0]))


def test_invariant_unknown_strategy_raises():
    """An unrecognized strategy name is rejected."""
    with pytest.raises(ValueError, match="unknown mix strategy"):
        list(mix_rows([_src("a")], strategy="bogus"))


def test_invariant_non_iterable_source_raises_typeerror():
    """``MixedDataset`` rejects a non-iterable source via ``_iter_source``."""
    with pytest.raises(TypeError, match="must be iterable"):
        MixedDataset([123])


# ---------------------------------------------------------------------------
# round_robin
# ---------------------------------------------------------------------------

def test_invariant_round_robin_interleaves_and_drains_exhausted():
    """One row per source per cycle; a drained source drops out, others
    continue."""
    out = mix_rows([_src("a1", "a2", "a3"), _src("b1")], strategy="round_robin")
    assert _ids(out) == ["a1", "b1", "a2", "a3"]


def test_invariant_round_robin_respects_max_samples_total():
    """``max_samples_total`` caps the round-robin stream."""
    out = mix_rows(
        [_src("a1", "a2"), _src("b1", "b2")],
        strategy="round_robin",
        max_samples_total=2,
    )
    assert _ids(out) == ["a1", "b1"]


def test_invariant_max_samples_per_source_caps_each_source():
    """``max_samples_per_source`` limits how many rows each source contributes."""
    out = mix_rows(
        [_src("a1", "a2", "a3"), _src("b1", "b2")],
        strategy="round_robin",
        max_samples_per_source=1,
    )
    assert _ids(out) == ["a1", "b1"]


# ---------------------------------------------------------------------------
# exhaust_then_resample
# ---------------------------------------------------------------------------

def test_invariant_exhaust_then_resample_drains_in_source_order():
    """Sources are drained fully, in order."""
    out = mix_rows(
        [_src("a1", "a2"), _src("b1")], strategy="exhaust_then_resample"
    )
    assert _ids(out) == ["a1", "a2", "b1"]


def test_invariant_exhaust_then_resample_respects_max_total():
    """``max_samples_total`` short-circuits the in-order drain."""
    out = mix_rows(
        [_src("a1", "a2"), _src("b1")],
        strategy="exhaust_then_resample",
        max_samples_total=1,
    )
    assert _ids(out) == ["a1"]


# ---------------------------------------------------------------------------
# weighted (default strategy)
# ---------------------------------------------------------------------------

def test_invariant_weighted_default_weights_preserve_all_rows():
    """Default (uniform) weights, no caps → every row appears exactly once
    (the loop drains and evicts each source, hitting the last-source break)."""
    sources = [_src("a1", "a2"), _src("b1", "b2")]
    out = mix_rows(sources, strategy="weighted", seed=0)
    assert sorted(_ids(out)) == ["a1", "a2", "b1", "b2"]


def test_invariant_weighted_is_deterministic_for_a_fixed_seed():
    """Same seed → identical interleaving; different seed → (here) different."""
    sources = [_src(*[f"a{i}" for i in range(8)]), _src(*[f"b{i}" for i in range(8)])]
    a = _ids(mix_rows(sources, strategy="weighted", weights=[1.0, 1.0], seed=7))
    b = _ids(mix_rows(sources, strategy="weighted", weights=[1.0, 1.0], seed=7))
    c = _ids(mix_rows(sources, strategy="weighted", weights=[1.0, 1.0], seed=123))
    assert a == b  # determinism
    assert a != c  # the two seeds produce different orders for this input


def test_invariant_weighted_respects_max_samples_total():
    """``max_samples_total`` caps the weighted stream."""
    out = mix_rows(
        [_src("a1", "a2", "a3"), _src("b1", "b2", "b3")],
        strategy="weighted",
        max_samples_total=2,
        seed=0,
    )
    assert len(_ids(out)) == 2


def test_invariant_weighted_single_source_drains_then_breaks():
    """A lone source is fully drained, then the ``not active`` break ends it."""
    out = mix_rows([_src("a1", "a2", "a3")], strategy="weighted", seed=0)
    assert _ids(out) == ["a1", "a2", "a3"]


def test_invariant_weighted_zero_remaining_weight_breaks_early():
    """With weights ``[1, 0]``, once the weighted source drains, the remaining
    active weight sums to 0 → the loop breaks and the zero-weight source's rows
    are never yielded."""
    out = mix_rows(
        [_src("a1", "a2"), _src("b1", "b2")],
        strategy="weighted",
        weights=[1.0, 0.0],
        seed=0,
    )
    assert _ids(out) == ["a1", "a2"]  # b* never sampled (weight 0, then break)


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def test_invariant_temperature_applied_preserves_all_rows():
    """``temperature != 1`` (and > 0) rescales weights but still drains all
    sources when uncapped."""
    sources = [_src("a1", "a2"), _src("b1", "b2")]
    out = mix_rows(sources, strategy="weighted", weights=[3.0, 1.0], temperature=0.5, seed=0)
    assert sorted(_ids(out)) == ["a1", "a2", "b1", "b2"]


def test_invariant_nonpositive_temperature_is_not_applied():
    """``temperature <= 0`` skips the rescale (guarded by ``temperature > 0``);
    uniform weights still yield a valid full drain."""
    sources = [_src("a1"), _src("b1")]
    out = mix_rows(sources, strategy="weighted", weights=[1.0, 1.0], temperature=0.0, seed=0)
    assert sorted(_ids(out)) == ["a1", "b1"]


# ---------------------------------------------------------------------------
# MixedDataset
# ---------------------------------------------------------------------------

def test_invariant_mixed_dataset_materializes_len_and_getitem():
    """``MixedDataset`` eagerly materializes the mix; ``__len__`` / ``__getitem__``
    expose it as a map-style dataset."""
    ds = MixedDataset(
        [_src("a1", "a2"), _src("b1")], strategy="round_robin"
    )
    assert len(ds) == 3
    assert ds[0] == {"id": "a1"}
    assert ds[1] == {"id": "b1"}
    assert ds[2] == {"id": "a2"}


def test_invariant_mixed_dataset_getitem_coerces_float_index_to_int():
    """``__getitem__`` coerces its index to int (``self._rows[int(idx)]``)."""
    ds = MixedDataset([_src("a1", "a2")], strategy="round_robin")
    assert ds[1.0] == {"id": "a2"}


def test_invariant_mixed_dataset_accepts_generator_sources():
    """A generator source is consumed via ``_iter_source`` (has ``__iter__``)."""
    def gen():
        yield {"id": "g1"}
        yield {"id": "g2"}

    ds = MixedDataset([gen()], strategy="exhaust_then_resample")
    assert len(ds) == 2
    assert _ids([ds[0], ds[1]]) == ["g1", "g2"]
