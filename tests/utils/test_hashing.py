"""Adversarial tests for ``lighttrain.utils.hashing``.

Coverage:

* **Deterministic** — same input → same output, across calls.
* **Distinct for distinct inputs** in a moderate (100-input) sweep.
* **Pin: 8-char default length**.
* **Pin: full sha256 hex when ``n=64``** — proves the underlying digest is sha256.
* **bytes input accepted directly** (no .encode())**.
* **Unicode input produces stable hash**.
* **Empty string is valid input**.
"""

from __future__ import annotations

import hashlib

import pytest

from lighttrain.utils.hashing import short_hash


def test_invariant_short_hash_deterministic_across_calls():
    """``short_hash("x")`` returns the same value every time."""
    a = short_hash("x")
    b = short_hash("x")
    c = short_hash("x")
    assert a == b == c


def test_invariant_short_hash_distinct_inputs_yield_distinct_outputs():
    """Sweep: 100 distinct inputs → 100 distinct outputs (no collisions
    in this range at the default 8-char length).

    8 hex chars = 32 bits → expected collision rate at 100 inputs is
    ~1 in 86 million pairs, so collisions are effectively impossible here.
    """
    hashes = {short_hash(f"input_{i}") for i in range(100)}
    assert len(hashes) == 100


def test_pin_short_hash_default_length_is_eight():
    """Pin: default ``n`` is 8 (line 8 of source)."""
    assert len(short_hash("anything")) == 8


@pytest.mark.parametrize("n", [1, 4, 8, 16, 32, 64])
def test_short_hash_length_matches_n_parameter(n):
    """``short_hash(text, n=N)`` returns exactly N characters."""
    out = short_hash("x", n=n)
    assert len(out) == n


def test_pin_short_hash_uses_sha256():
    """Pin: ``short_hash("hello", n=64)`` equals ``hashlib.sha256("hello"
    .encode()).hexdigest()``.

    This proves the underlying digest is sha256. If switched to e.g.
    blake3 or md5, this test fails and forces a documented change.
    """
    expected = hashlib.sha256(b"hello").hexdigest()
    assert short_hash("hello", n=64) == expected


def test_short_hash_accepts_bytes_input_directly():
    """Pin: bytes input bypasses the str→bytes encode step (line 10-11)."""
    # Both forms should produce the same digest.
    text_form = short_hash("hello")
    bytes_form = short_hash(b"hello")
    assert text_form == bytes_form


def test_short_hash_unicode_input_stable():
    """Unicode input produces a stable hash; the encoding is UTF-8."""
    a = short_hash("你好")
    b = short_hash("你好")
    assert a == b
    # Cross-check with hashlib direct call
    expected = hashlib.sha256("你好".encode()).hexdigest()[:8]
    assert a == expected


def test_short_hash_empty_string_returns_known_digest():
    """Empty string is valid input → first 8 chars of sha256("").

    Closed form: sha256("") starts with ``e3b0c442``.
    """
    assert short_hash("") == "e3b0c442"


def test_short_hash_consistency_across_str_and_bytes():
    """``short_hash(s)`` and ``short_hash(s.encode())`` are identical for
    ASCII; for unicode they're identical via UTF-8.
    """
    for s in ["a", "ab", "abc", "longer string with spaces"]:
        assert short_hash(s) == short_hash(s.encode("utf-8"))
