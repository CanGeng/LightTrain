"""Seed determinism + RNG snapshot/restore."""

from __future__ import annotations

import random

import pytest
import torch

from lighttrain.utils.seed import restore_rng_state, rng_state, seed_everything


def test_seed_python_random_deterministic():
    seed_everything(1234)
    a = [random.random() for _ in range(8)]
    seed_everything(1234)
    b = [random.random() for _ in range(8)]
    assert a == b


def test_seed_torch_deterministic():
    seed_everything(7)
    t1 = torch.randn(4, 4)
    seed_everything(7)
    t2 = torch.randn(4, 4)
    assert torch.equal(t1, t2)


def test_rng_state_round_trip():
    seed_everything(42)
    state = rng_state()
    a = (random.random(), torch.randn(2).tolist())
    restore_rng_state(state)
    b = (random.random(), torch.randn(2).tolist())
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])


def test_seed_returns_clamped_value():
    out = seed_everything(2**40 + 5)
    assert 0 <= out <= 2**32 - 1
