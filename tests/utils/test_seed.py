"""Adversarial tests for ``lighttrain.utils.seed``.

Coverage:

* **Reproducibility across Python, NumPy, torch RNGs**.
* **Seed clamped to 32 bits** (line 16 of seed.py).
* **Pin: 32-bit truncation causes collisions** when two seeds share their
  low 32 bits — documents the current sharp edge.
* **rng_state round-trip** — capture, advance, restore, re-advance →
  identical sequences.
* **restore_rng_state with a partial dict** (missing keys) does not raise.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lighttrain.utils.seed import restore_rng_state, rng_state, seed_everything


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_invariant_seed_everything_makes_torch_randn_reproducible():
    """Closed form: ``seed_everything(0)`` then ``torch.randn(3)`` twice
    yields identical tensors.
    """
    seed_everything(0)
    a = torch.randn(3)
    seed_everything(0)
    b = torch.randn(3)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_invariant_seed_everything_makes_numpy_random_reproducible():
    """``seed_everything`` also seeds NumPy."""
    seed_everything(0)
    a = np.random.randn(5)
    seed_everything(0)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


def test_invariant_seed_everything_makes_python_random_reproducible():
    """``seed_everything`` also seeds the Python random module."""
    seed_everything(0)
    a = [random.random() for _ in range(5)]
    seed_everything(0)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_distinct_seeds_produce_distinct_torch_sequences():
    """Different seeds (within the 32-bit range) → different sequences."""
    seed_everything(0)
    a = torch.randn(3)
    seed_everything(1)
    b = torch.randn(3)
    # Cosine similarity should be far from 1.0 between independent draws
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# 32-bit truncation pin
# ---------------------------------------------------------------------------

def test_invariant_seed_clamped_to_32_bits_returns_low_word():
    """``seed_everything(2**40 + 5)`` returns ``5`` (low 32 bits)
    (line 16 of source: ``seed & 0xFFFFFFFF``).
    """
    returned = seed_everything(2**40 + 5)
    assert returned == 5


def test_seed_does_not_raise_on_huge_int():
    """Pin: a huge seed like 2**60 does not raise (it's silently truncated)."""
    returned = seed_everything(2**60 + 12345)
    assert returned == 12345


def test_pin_seed_truncation_collides_at_low_32_bit_boundary():
    """Pin: seeds sharing their low 32 bits produce IDENTICAL RNG sequences.

    Setup: ``seed=2**40 + 5`` vs ``seed=5``.
    Expected: torch.randn outputs are exactly equal.

    This exposes a sharp edge: callers passing >32-bit seeds may unknowingly
    collide. If you fix this by switching to a wider hash, update this test
    AND callers that rely on the low-word semantics.
    """
    seed_everything(2**40 + 5)
    a = torch.randn(3)
    seed_everything(5)
    b = torch.randn(3)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# rng_state round-trip
# ---------------------------------------------------------------------------

def test_invariant_rng_state_round_trip_preserves_python_random():
    """Capture → advance → restore → re-advance yields identical sequences
    (Python random branch).
    """
    seed_everything(42)
    state = rng_state()
    a = [random.random() for _ in range(5)]
    restore_rng_state(state)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_invariant_rng_state_round_trip_preserves_torch():
    """Same for torch RNG."""
    seed_everything(42)
    state = rng_state()
    a = torch.randn(5)
    restore_rng_state(state)
    b = torch.randn(5)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_invariant_rng_state_round_trip_preserves_numpy():
    """NumPy RNG is also restored to the captured state."""
    seed_everything(42)
    state = rng_state()
    a = np.random.randn(5)
    restore_rng_state(state)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


def test_rng_state_keys_include_python_torch_numpy():
    """Pin: rng_state() returns at minimum {'python', 'torch', 'numpy'}
    on a system with numpy + torch installed.
    """
    state = rng_state()
    assert "python" in state
    assert "torch" in state
    assert "numpy" in state


def test_restore_rng_state_partial_dict_does_not_raise():
    """A partial state dict (only 'python' key) restores that source and
    silently skips the missing ones.

    Setup: seed; capture state; record the NEXT random.random() value as
    ``expected``; perturb RNGs; restore partial; verify next random.random()
    matches ``expected``.
    """
    seed_everything(0)
    state = rng_state()
    expected = random.random()  # the value the captured state will produce next
    # Perturb both python and torch RNG state.
    _ = random.random()
    torch.randn(3)
    partial = {"python": state["python"]}
    restore_rng_state(partial)
    # python RNG restored to the captured state → next draw equals `expected`
    assert random.random() == expected


def test_restore_rng_state_empty_dict_does_not_raise():
    """An empty state dict is a no-op (does not raise)."""
    restore_rng_state({})
