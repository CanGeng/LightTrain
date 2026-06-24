"""Adversarial tests for ``lighttrain.data.prepgraph._fp``.

Targets that the legacy ``tests/test_prepgraph_fingerprint.py`` misses:
  * The historical FP_NONE_01 fix (docs/changelog/v0.1.3): ``None`` values
    must be preserved by ``canonical_config`` so explicit nulls invalidate
    caches.
  * Pinned-but-mutable choices (float quantization at 9 decimals, input_fps
    normalization via ``sorted(str(...))``) live in dedicated ``test_pin_*``
    tests with the standard pin warning.
  * Sensitivity to every field of the fingerprint payload (kind /
    schema_version / code_version) — change one field, the digest changes.
  * Trap cases: ``True != 1`` (Python's bool-is-int collision), ``-0.0 == 0.0``,
    Unicode keys, non-string ``input_fps``, ``getsource`` caching.
"""
from __future__ import annotations

import hashlib
import inspect

from lighttrain.data.prepgraph._fp import (
    SCHEMA_VERSION,
    canonical_config,
    code_version_for,
    compose_fingerprint,
)

# --------------------------------------------------------------------------- #
# canonical_config: None preservation (FP_NONE_01 regression)                 #
# --------------------------------------------------------------------------- #


def test_canonical_config_preserves_none_values() -> None:
    """Post-fix behavior: ``None`` is preserved, not dropped.

    Input: ``{"a": 1, "b": None}``. Analytical: ``canonical_config`` returns
    the same key set; ``None`` survives as ``None`` so JSON serialization
    will produce ``null`` (vs. omitting the key, which would collapse with
    ``{"a": 1}``).

    Pre-fix had ``if v is None: continue`` which dropped the key — see
    docs/changelog/v0.1.3.
    """
    out = canonical_config({"a": 1, "b": None})
    assert out == {"a": 1, "b": None}
    assert "b" in out  # explicit: the key SURVIVES


def test_regression_FP_NONE_01_explicit_none_changes_fingerprint() -> None:
    """Pre-fix bug: ``canonical_config`` had ``if v is None: continue``,
    dropping None values. ``{"lr": 0.001, "warmup": None}`` and
    ``{"lr": 0.001}`` collapsed to the same fingerprint, causing false
    cache hits when configs were explicitly nulled to override defaults
    (see docs/changelog/v0.1.3: 'canonical_config None 指纹碰撞').

    Input: the exact two configs from the changelog. Analytical: post-fix,
    the explicit ``None`` participates in the JSON payload as ``null``, so
    the SHA-256 over the canonicalized payload differs from the
    ``warmup``-absent version.

    Also pins ``{"a": None}`` ≠ ``{"a": {}}`` — a separate trap that the
    same dropping behavior would have collapsed.
    """
    base = dict(
        kind="dummy",
        schema_kind="rows",
        code_version="cv",
        input_fps=[],
    )
    fp_with_none = compose_fingerprint(
        config={"lr": 0.001, "warmup": None}, **base
    )
    fp_without = compose_fingerprint(config={"lr": 0.001}, **base)
    assert fp_with_none != fp_without, (
        "Pre-fix: None dropped, configs collapse to identical fingerprint"
    )

    fp_none = compose_fingerprint(config={"a": None}, **base)
    fp_empty = compose_fingerprint(config={"a": {}}, **base)
    assert fp_none != fp_empty


# --------------------------------------------------------------------------- #
# canonical_config: ordering, collections, types                              #
# --------------------------------------------------------------------------- #


def test_canonical_config_nested_dict_key_order_invariant() -> None:
    """Deeply nested dicts with permuted keys produce identical canonical form.

    Input: a 5-level nested dict and a key-permuted equivalent. Analytical:
    every level sorts keys (``for k in sorted(value)`` in the function body),
    so any permutation collapses to one canonical ordering.
    """
    a = {
        "z": 1,
        "a": {"y": 2, "b": {"x": 3, "c": {"w": 4, "d": {"v": 5, "e": 6}}}},
    }
    b = {
        "a": {"b": {"c": {"d": {"e": 6, "v": 5}, "w": 4}, "x": 3}, "y": 2},
        "z": 1,
    }
    assert canonical_config(a) == canonical_config(b)


def test_canonical_config_tuple_list_equivalent() -> None:
    """``(1, 2, 3)`` and ``[1, 2, 3]`` canonicalize to identical lists.

    Pin: JSON has no tuple type so the function converts tuples to lists;
    YAML/JSON round-trips would otherwise mutate fingerprints.
    """
    assert canonical_config((1, 2, 3)) == canonical_config([1, 2, 3])
    # Nested tuple inside dict.
    assert canonical_config({"x": (1, 2)}) == canonical_config({"x": [1, 2]})


def test_canonical_config_negative_zero_and_subnormal_floats() -> None:
    """``-0.0`` canonicalizes to ``0.0`` and tiny subnormals round to ``0.0``;
    ``-1e-9`` retains its sign.

    Pin: the 9-decimal round (``round(value, 9)``) flips ``-0.0 → 0.0`` and
    crushes ``1e-12 → 0.0``, but ``-1e-9`` remains ``-1e-9``.
    """
    assert canonical_config(-0.0) == 0.0
    assert canonical_config(1e-12) == 0.0
    assert canonical_config(-1e-9) == -1e-9


def test_canonical_config_bool_not_int() -> None:
    """``True`` stays ``True``, not coerced to ``1``.

    Trap: Python's ``isinstance(True, int) is True`` is a common source of
    collisions. The function checks ``isinstance(value, bool)`` BEFORE the
    int branch, so booleans survive as bools.

    Two configs ``{"flag": True}`` and ``{"flag": 1}`` must therefore produce
    different fingerprints.
    """
    assert canonical_config(True) is True
    assert canonical_config(False) is False
    base = dict(
        kind="dummy",
        schema_kind="rows",
        code_version="cv",
        input_fps=[],
    )
    fp_bool = compose_fingerprint(config={"flag": True}, **base)
    fp_int = compose_fingerprint(config={"flag": 1}, **base)
    assert fp_bool != fp_int


def test_canonical_config_repr_fallback_for_unknown_types() -> None:
    """Unknown types fall back to ``repr`` deterministically.

    Input: a custom ``object()``. Two configs containing the same object
    instance must canonicalize identically (``repr`` of the same object is
    stable within one process).
    """
    sentinel = object()
    a = canonical_config({"obj": sentinel})
    b = canonical_config({"obj": sentinel})
    assert a == b
    # And it's a string (the repr), not the original object.
    assert isinstance(a["obj"], str)


# --------------------------------------------------------------------------- #
# pin: float quantization                                                     #
# --------------------------------------------------------------------------- #


def test_pin_canonical_config_float_quantization_at_nine_decimals() -> None:
    """Floats are rounded to 9 decimals: values agreeing to that precision
    collapse, values differing at the 9th decimal split.

    Input pairs:
        within  9 dec: ``0.1234567891`` ≈ ``0.1234567892``  →  equal
        across  9 dec: ``0.123456789`` ≠ ``0.123456788``    →  distinct

    If this behavior is intentionally changed, update this test AND bump
    SCHEMA_VERSION (or document the breaking change).
    """
    # Within rounding granularity (10th decimal differs) → equal.
    assert canonical_config(0.1234567891) == canonical_config(0.1234567892)
    # Across 9th decimal → distinct.
    assert canonical_config(0.123456789) != canonical_config(0.123456788)


# --------------------------------------------------------------------------- #
# compose_fingerprint: determinism + sensitivity to every payload field       #
# --------------------------------------------------------------------------- #


def _base_kwargs():
    return dict(
        kind="tokenize",
        schema_kind="rows",
        code_version="cv0",
        config={"lr": 0.1},
        input_fps=["u0"],
    )


def test_compose_fingerprint_deterministic() -> None:
    """Same args → identical 64-char SHA-256 hex digest.

    Invariant: ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
    plus ``hashlib.sha256(...).hexdigest()`` is deterministic by construction.
    """
    fp1 = compose_fingerprint(**_base_kwargs())
    fp2 = compose_fingerprint(**_base_kwargs())
    assert fp1 == fp2
    assert len(fp1) == 64
    int(fp1, 16)  # raises if not valid hex


def test_compose_fingerprint_sensitive_to_kind() -> None:
    """Changing ``kind`` changes the digest."""
    base = _base_kwargs()
    other = dict(base, kind="validate")
    assert compose_fingerprint(**base) != compose_fingerprint(**other)


def test_compose_fingerprint_sensitive_to_code_version() -> None:
    """Changing ``code_version`` changes the digest."""
    base = _base_kwargs()
    other = dict(base, code_version="cv1")
    assert compose_fingerprint(**base) != compose_fingerprint(**other)


def test_compose_fingerprint_sensitive_to_schema_version(monkeypatch) -> None:
    """Bumping ``SCHEMA_VERSION[schema_kind]`` changes the digest.

    Input: monkeypatch ``SCHEMA_VERSION["rows"]`` from its current value to
    a new value; the fingerprint must change. Analytical: the payload
    includes ``"schema_version": SCHEMA_VERSION.get(schema_kind, "0.0")``.
    """
    fp_before = compose_fingerprint(**_base_kwargs())
    new_version = SCHEMA_VERSION["rows"] + "-bumped"
    monkeypatch.setitem(SCHEMA_VERSION, "rows", new_version)
    fp_after = compose_fingerprint(**_base_kwargs())
    assert fp_before != fp_after


# --------------------------------------------------------------------------- #
# compose_fingerprint: input_fps normalization                                #
# --------------------------------------------------------------------------- #


def test_compose_fingerprint_input_order_invariant() -> None:
    """Permuting ``input_fps`` does not change the digest.

    Input: ``["u0", "u1", "u2"]`` and ``["u2", "u0", "u1"]``.
    Analytical: the payload uses ``sorted(str(x) for x in input_fps)``.
    """
    a = dict(_base_kwargs(), input_fps=["u0", "u1", "u2"])
    b = dict(_base_kwargs(), input_fps=["u2", "u0", "u1"])
    assert compose_fingerprint(**a) == compose_fingerprint(**b)


def test_pin_compose_fingerprint_input_fps_normalized_via_sorted_str() -> None:
    """Non-string ``input_fps`` are normalized through ``str(...)``.

    Input: ``[1, 2]`` and ``[2, 1]`` (ints) → identical digest, because they
    get coerced to ``["1", "2"]`` and then sorted.

    Stronger pin: ``[1, 2]`` and ``["1", "2"]`` also collapse to the same
    digest — the ``str(x)`` coercion does the cross-type normalization.

    If this behavior is intentionally changed (e.g. requiring strict string
    input), update this test AND bump SCHEMA_VERSION (or document the
    breaking change).
    """
    a = dict(_base_kwargs(), input_fps=[1, 2])
    b = dict(_base_kwargs(), input_fps=[2, 1])
    c = dict(_base_kwargs(), input_fps=["1", "2"])
    fp_a = compose_fingerprint(**a)
    fp_b = compose_fingerprint(**b)
    fp_c = compose_fingerprint(**c)
    assert fp_a == fp_b == fp_c


def test_compose_fingerprint_dict_key_unicode() -> None:
    """Non-ASCII dict keys hash deterministically.

    Pin: ``sort_keys=True`` in ``json.dumps`` and the explicit ``str(k)`` in
    ``canonical_config`` ensure that Unicode keys order consistently.
    """
    a = dict(_base_kwargs(), config={"键α": 1, "键β": 2})
    b = dict(_base_kwargs(), config={"键β": 2, "键α": 1})
    assert compose_fingerprint(**a) == compose_fingerprint(**b)


# --------------------------------------------------------------------------- #
# code_version_for: source hashing + cache + fallback                         #
# --------------------------------------------------------------------------- #


class _CodeA:
    """A class whose source is X."""

    def f(self) -> int:
        return 1


class _CodeB:
    """A class whose source is Y — different from _CodeA."""

    def f(self) -> int:
        return 2


def test_code_version_for_class_uses_source_hash() -> None:
    """Distinct class sources produce distinct code_version digests.

    Analytical: ``code_version_for(cls)`` returns
    ``sha256(inspect.getsource(cls))`` when possible. _CodeA and _CodeB have
    different sources, so their hex digests must differ.
    """
    cv_a = code_version_for(_CodeA)
    cv_b = code_version_for(_CodeB)
    assert cv_a != cv_b
    # Sanity: each is a SHA-256 hex digest of the source.
    expected_a = hashlib.sha256(inspect.getsource(_CodeA).encode("utf-8")).hexdigest()
    assert cv_a == expected_a


def test_code_version_for_class_cached(monkeypatch) -> None:
    """Repeated lookups for the same class do not re-read source from disk.

    Invariant: ``code_version_for`` is wrapped in ``functools.lru_cache``;
    the second call must NOT trigger ``inspect.getsource``.
    """
    # Clear the LRU cache to ensure a fresh look-up sequence.
    code_version_for.cache_clear()

    call_count = {"n": 0}
    real_getsource = inspect.getsource

    def _counting_getsource(obj):
        call_count["n"] += 1
        return real_getsource(obj)

    monkeypatch.setattr(inspect, "getsource", _counting_getsource)

    code_version_for(_CodeA)
    after_first = call_count["n"]
    code_version_for(_CodeA)
    after_second = call_count["n"]

    assert after_first == 1
    assert after_second == 1, "second call must hit the LRU cache, not re-read source"


def test_code_version_for_getsource_failure_falls_back(monkeypatch) -> None:
    """When ``inspect.getsource`` raises, fall back to a deterministic hash.

    Input: monkeypatch ``inspect.getsource`` to raise ``OSError``; the
    function must still return a hex digest (from the module/mtime fallback).
    Two calls with the same argument under the same patched conditions
    must return the same digest.
    """
    code_version_for.cache_clear()

    def _boom(_obj):
        raise OSError("simulated")

    monkeypatch.setattr(inspect, "getsource", _boom)
    cv1 = code_version_for(_CodeA)
    cv2 = code_version_for(_CodeA)
    assert cv1 == cv2
    assert len(cv1) == 64
    int(cv1, 16)  # valid hex digest


# --------------------------------------------------------------------------- #
# Cross-field interaction: payload assembly is the source of truth            #
# --------------------------------------------------------------------------- #


def test_compose_fingerprint_payload_is_canonicalized() -> None:
    """A config tuple yields the same digest as the equivalent list.

    Cross-pins canonical_config → compose_fingerprint plumbing.
    """
    a = dict(_base_kwargs(), config={"x": (1, 2)})
    b = dict(_base_kwargs(), config={"x": [1, 2]})
    assert compose_fingerprint(**a) == compose_fingerprint(**b)


# --------------------------------------------------------------------------- #
# SCHEMA_VERSION table completeness                                            #
# --------------------------------------------------------------------------- #


def test_schema_version_table_has_every_leaf_node_schema() -> None:
    """Every ``schema_kind`` produced by a leaf node has a ``SCHEMA_VERSION``
    entry.

    Pin: a missing entry means ``compose_fingerprint`` would silently fall
    back to the ``"0.0"`` default for that schema, so bumping the real schema
    could not invalidate caches. The expected set is the union of node output
    schema kinds (DESIGN §7.7.4).
    """
    expected = {
        "rows",
        "tokenized_rows",
        "packed_rows",
        "validate_report",
        "materialized",
        "mixed_rows",
        "chunked_rows",
    }
    for k in expected:
        assert k in SCHEMA_VERSION, f"missing schema_version for {k!r}"
