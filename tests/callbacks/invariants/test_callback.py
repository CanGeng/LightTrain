"""InvariantsCallback abort/skip/warn behavior."""

from __future__ import annotations

import warnings

import pytest
import torch

from lighttrain.builtin_plugins.callbacks.invariants import InvariantsCallback
from lighttrain.callbacks.base import Signal
from lighttrain.callbacks.invariants import InvariantError


def test_invariant_abort_raises():
    cb = InvariantsCallback(
        specs=[{"check": "False", "action": "abort"}],
    )
    with pytest.raises(InvariantError):
        cb.on_loss_computed(
            step=1,
            loss=torch.tensor(1.0),
            outputs=None,
            batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)},
            model=None,
            metrics={},
        )


def test_invariant_skip_returns_signal():
    cb = InvariantsCallback(
        specs=[{"check": "False", "action": "skip"}],
    )
    sig = cb.on_loss_computed(
        step=2,
        loss=torch.tensor(1.0),
        outputs=None,
        batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)},
        model=None,
        metrics={},
    )
    assert sig == Signal.SKIP_STEP


def test_invariant_warn_emits_warning():
    cb = InvariantsCallback(
        specs=[{"check": "False", "action": "warn"}],
    )
    metrics: dict = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig = cb.on_loss_computed(
            step=3,
            loss=torch.tensor(1.0),
            outputs=None,
            batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)},
            model=None,
            metrics=metrics,
        )
    assert sig == Signal.CONTINUE
    assert any("violated" in str(x.message) for x in w)


def test_invariant_records_violation_into_metrics():
    cb = InvariantsCallback(
        specs=[{"check": "False", "action": "warn"}],
    )
    metrics: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb.on_loss_computed(
            step=4,
            loss=torch.tensor(1.0),
            outputs=None,
            batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)},
            model=None,
            metrics=metrics,
        )
    log = metrics.get("_invariant_violations")
    assert isinstance(log, list) and len(log) == 1
    assert log[0]["action"] == "warn"


def test_invariants_critical_flag():
    cb = InvariantsCallback()
    assert cb.critical is True
