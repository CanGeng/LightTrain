"""Shared fixtures for adversarial losses/* tests.

These fixtures are scoped to ``tests/losses/`` so they do not leak into the
flat-layout legacy tests under ``tests/test_losses_*.py``.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.protocols import LossContext, ModelOutput


@pytest.fixture(autouse=True)
def seeded_rng():
    """Seed torch RNG per-test so adversarial tests stay deterministic.

    Autouse keeps every test in this directory reproducible. Failures will
    not jitter between runs.
    """
    torch.manual_seed(0)
    yield


@pytest.fixture
def dummy_ctx() -> LossContext:
    """Return a fresh, empty ``LossContext`` (no shared state across tests)."""
    return LossContext()


@pytest.fixture
def dummy_model_output() -> ModelOutput:
    """Return an empty ``ModelOutput`` — extend in-test as needed."""
    return ModelOutput(outputs={})
