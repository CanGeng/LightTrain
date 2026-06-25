"""Branch-exhaustive tests for the seven builtin invariant factories in
``lighttrain.builtin_plugins.callbacks.invariants.builtins``.

This file is the ``_cov`` companion to ``test_builtins.py`` (which pins only
the happy-path holds/violates contract). Here we drive every *reachable*
guard and exception branch of each invariant function toward full coverage:

* ``loss_finite``: None short-circuit; tensor isfinite; non-tensor float path
  for nan/inf/-inf; the ``except`` fallback for un-floatable losses.
* ``grad_norm_bounded``: falsy/empty metrics; ``grad_norm`` absent; numeric
  compare; ``except (TypeError, ValueError)`` fallback for un-floatable values.
* ``lr_nonneg``: None optimizer; ``.optimizer`` wrapper unwrap; missing/empty
  ``param_groups``; negative lr -> violation; ``continue`` on un-floatable lr.
* ``label_mask_nonzero``: non-dict batch; labels None; tensor path; non-tensor
  iterable path (holds / all-masked); the ``except`` fallback.
* ``param_count_stable``: None guards; first-call seeds prev; stable; trips
  when trainable-param count changes (freeze).
* ``dtype_stable``: None guards; empty-model ``StopIteration``; first-call
  seeds prev; stable; trips on dtype change.
* ``batch_nonempty``: non-dict batch; non-empty leading-dim tensor; the
  all-empty / no-tensor fall-through to ``False``.

The functions are imported directly from the module (importing the module also
executes the ``@register("invariant", ...)`` decorators as a side effect, so the
registration lines are covered too). No randomness; tensors are constructed with
fixed values, so nothing here is flaky.
"""

from __future__ import annotations

import logging

import pytest
import torch

import lighttrain.builtin_plugins.callbacks.invariants.builtins as bi
from lighttrain.builtin_plugins.callbacks.invariants.builtins import (
    batch_nonempty,
    dtype_stable,
    grad_norm_bounded,
    label_mask_nonzero,
    loss_finite,
    lr_nonneg,
    param_count_stable,
)


# --------------------------------------------------------------------------- #
# helpers / stubs
# --------------------------------------------------------------------------- #
class _Optim:
    """Bare optimizer exposing ``param_groups`` (no ``.optimizer`` wrapper)."""

    def __init__(self, groups):
        self.param_groups = groups


class _Wrapped:
    """Wrapper exposing an inner ``.optimizer`` (e.g. an accelerate-style shim)."""

    def __init__(self, groups):
        self.optimizer = _Optim(groups)


class _EmptyModule(torch.nn.Module):
    """A module with zero parameters (``next(model.parameters())`` raises)."""


# --------------------------------------------------------------------------- #
# loss_finite  (lines 33-41)
# --------------------------------------------------------------------------- #
def test_invariant_loss_finite_none_holds():
    """``loss_finite`` short-circuits to True when no loss is supplied (line 34)."""
    assert loss_finite() is True
    assert loss_finite(loss=None) is True


def test_invariant_loss_finite_tensor_paths():
    """Tensor branch: finite holds, any NaN/Inf element violates (line 36)."""
    assert loss_finite(loss=torch.tensor([1.0, 2.0, 3.0])) is True
    assert loss_finite(loss=torch.tensor([1.0, float("nan")])) is False
    assert loss_finite(loss=torch.tensor([float("inf")])) is False


def test_invariant_loss_finite_scalar_float_holds():
    """Non-tensor finite float takes the ``float(loss)`` path and holds (37-38)."""
    assert loss_finite(loss=1.5) is True
    assert loss_finite(loss=0) is True


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invariant_loss_finite_scalar_nonfinite_violates(bad):
    """Non-tensor nan/inf/-inf violate via the ``loss == loss``/abs check (38)."""
    assert loss_finite(loss=bad) is False


def test_pin_current_behavior_loss_finite_unfloatable_holds(caplog):
    """Pin (debatable): an un-floatable non-tensor loss (e.g. a string) hits the
    ``except`` fallback and is treated as *holding* (returns True), with a
    warning logged (lines 39-41). Arguably a bad loss object should surface as a
    violation, so this pins current lenient behavior."""
    with caplog.at_level(logging.WARNING, logger=bi.__name__):
        assert loss_finite(loss="not-a-number") is True
        assert loss_finite(loss=object()) is True
    assert any("loss_finite" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# grad_norm_bounded  (lines 47-55)
# --------------------------------------------------------------------------- #
def test_invariant_grad_norm_bounded_empty_metrics_holds():
    """Falsy/empty metrics short-circuit to True (line 48)."""
    assert grad_norm_bounded() is True
    assert grad_norm_bounded(metrics=None) is True
    assert grad_norm_bounded(metrics={}) is True


def test_invariant_grad_norm_bounded_missing_key_holds():
    """Metrics present but no ``grad_norm`` key -> holds (line 51)."""
    assert grad_norm_bounded(metrics={"loss": 0.3}) is True


def test_invariant_grad_norm_bounded_numeric_compare():
    """Numeric compare: below max holds, at/above max violates (line 53)."""
    assert grad_norm_bounded(metrics={"grad_norm": 5.0}, max=10) is True
    assert grad_norm_bounded(metrics={"grad_norm": 50.0}, max=10) is False
    # default max is 1e3
    assert grad_norm_bounded(metrics={"grad_norm": 999.0}) is True
    assert grad_norm_bounded(metrics={"grad_norm": 1e4}) is False


def test_pin_current_behavior_grad_norm_unfloatable_holds():
    """Pin (debatable): a non-numeric ``grad_norm`` raises in ``float`` and is
    swallowed as *holding* (lines 54-55), rather than flagged as a violation."""
    assert grad_norm_bounded(metrics={"grad_norm": "oops"}, max=10) is True
    assert grad_norm_bounded(metrics={"grad_norm": None}, max=10) is True


# --------------------------------------------------------------------------- #
# lr_nonneg  (lines 61-73)
# --------------------------------------------------------------------------- #
def test_invariant_lr_nonneg_none_optimizer_holds():
    """No optimizer -> holds (line 62)."""
    assert lr_nonneg() is True
    assert lr_nonneg(optimizer=None) is True


def test_invariant_lr_nonneg_missing_or_empty_groups_holds():
    """Optimizer with absent or empty ``param_groups`` -> holds (lines 64-66)."""
    assert lr_nonneg(optimizer=_Optim(None)) is True  # groups is None
    assert lr_nonneg(optimizer=_Optim([])) is True  # groups is empty


def test_invariant_lr_nonneg_wrapper_unwrapped_and_holds():
    """A ``.optimizer``-wrapped optimizer is unwrapped (line 63); all-nonneg
    LRs hold (lines 67-69, 73)."""
    assert lr_nonneg(optimizer=_Wrapped([{"lr": 0.1}, {"lr": 0.0}])) is True


def test_invariant_lr_nonneg_negative_violates():
    """A negative LR in any param group is a violation (lines 69-70)."""
    assert lr_nonneg(optimizer=_Optim([{"lr": 0.1}, {"lr": -1e-4}])) is False


def test_pin_current_behavior_lr_nonneg_unfloatable_continues():
    """Pin (debatable): an un-floatable lr raises in ``float`` and is skipped via
    ``continue`` (lines 71-72), so a group with a garbage lr does not trip the
    invariant. A group missing ``lr`` defaults to 0.0 and holds."""
    assert lr_nonneg(optimizer=_Optim([{"lr": object()}])) is True
    assert lr_nonneg(optimizer=_Optim([{}])) is True  # .get default 0.0


# --------------------------------------------------------------------------- #
# label_mask_nonzero  (lines 85-96)
# --------------------------------------------------------------------------- #
def test_invariant_label_mask_nonzero_non_dict_holds():
    """A non-dict batch short-circuits to True (line 86)."""
    assert label_mask_nonzero(batch=None) is True
    assert label_mask_nonzero(batch=[1, 2, 3]) is True


def test_invariant_label_mask_nonzero_missing_labels_holds():
    """Dict batch without a ``labels`` entry -> holds (line 89)."""
    assert label_mask_nonzero(batch={"input_ids": torch.zeros(2, 2)}) is True


def test_invariant_label_mask_nonzero_tensor_paths():
    """Tensor labels: at least one non-ignore position holds; all-masked
    violates; a custom ``ignore_index`` is honored (lines 90-91)."""
    good = {"labels": torch.tensor([[1, 2, -100], [-100, -100, 5]])}
    bad = {"labels": torch.tensor([[-100, -100], [-100, -100]])}
    assert label_mask_nonzero(batch=good) is True
    assert label_mask_nonzero(batch=bad) is False
    # custom ignore_index: every label equals 7 -> masked out -> violation
    assert label_mask_nonzero(batch={"labels": torch.tensor([7, 7])}, ignore_index=7) is False


def test_invariant_label_mask_nonzero_iterable_paths():
    """Non-tensor iterable labels take the ``any(int(x) ...)`` path (line 93):
    holds when an unmasked label exists, violates when all are ignore_index."""
    assert label_mask_nonzero(batch={"labels": [1, -100, -100]}) is True
    assert label_mask_nonzero(batch={"labels": [-100, -100]}) is False


def test_pin_current_behavior_label_mask_unintable_holds(caplog):
    """Pin (debatable): non-tensor labels that can't be ``int()``-ed raise and
    are swallowed as *holding* (lines 94-96), with a warning logged."""
    with caplog.at_level(logging.WARNING, logger=bi.__name__):
        assert label_mask_nonzero(batch={"labels": [object(), object()]}) is True
    assert any("label_mask_nonzero" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# param_count_stable  (lines 103-110)
# --------------------------------------------------------------------------- #
def test_invariant_param_count_stable_none_guards_hold():
    """Missing model or metrics -> holds (line 104)."""
    assert param_count_stable(model=None, metrics={}) is True
    assert param_count_stable(model=torch.nn.Linear(2, 2), metrics=None) is True


def test_invariant_param_count_stable_seeds_then_stable():
    """First call seeds ``_invariant_param_count`` and holds (108-109); a second
    call with an unchanged model still holds (line 110)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 4)  # 3*4 weights + 4 bias = 16 trainable params
    metrics: dict = {}
    assert param_count_stable(model=model, metrics=metrics) is True
    assert metrics["_invariant_param_count"] == pytest.approx(16.0)
    # second call: prev is set, count unchanged -> holds
    assert param_count_stable(model=model, metrics=metrics) is True


def test_invariant_param_count_stable_trips_on_freeze():
    """Freezing parameters changes the trainable count between calls and the
    invariant trips (line 110 -> False)."""
    model = torch.nn.Linear(3, 4)
    metrics: dict = {}
    assert param_count_stable(model=model, metrics=metrics) is True
    for p in model.parameters():
        p.requires_grad_(False)
    assert param_count_stable(model=model, metrics=metrics) is False


# --------------------------------------------------------------------------- #
# dtype_stable  (lines 116-127)
# --------------------------------------------------------------------------- #
def test_invariant_dtype_stable_none_guards_hold():
    """Missing model or metrics -> holds (line 117)."""
    assert dtype_stable(model=None, metrics={}) is True
    assert dtype_stable(model=torch.nn.Linear(2, 2), metrics=None) is True


def test_invariant_dtype_stable_empty_model_holds():
    """A parameterless model raises ``StopIteration`` and holds (lines 119-121)."""
    assert dtype_stable(model=_EmptyModule(), metrics={}) is True


def test_invariant_dtype_stable_seeds_then_stable():
    """First call records first-param dtype and holds (125-126); a second call
    with an unchanged dtype still holds (line 127)."""
    model = torch.nn.Linear(2, 2)
    metrics: dict = {}
    assert dtype_stable(model=model, metrics=metrics) is True
    assert metrics["_invariant_dtype"] == "torch.float32"
    assert dtype_stable(model=model, metrics=metrics) is True


def test_invariant_dtype_stable_trips_on_dtype_change():
    """When the recorded dtype differs from the current one, the invariant
    trips (line 127 -> False)."""
    model = torch.nn.Linear(2, 2)
    metrics = {"_invariant_dtype": "torch.float16"}
    assert dtype_stable(model=model, metrics=metrics) is False


# --------------------------------------------------------------------------- #
# batch_nonempty  (lines 133-138)
# --------------------------------------------------------------------------- #
def test_invariant_batch_nonempty_non_dict_holds():
    """A non-dict batch short-circuits to True (line 134)."""
    assert batch_nonempty(batch=None) is True
    assert batch_nonempty(batch=[torch.zeros(2, 2)]) is True


def test_invariant_batch_nonempty_tensor_with_leading_dim_holds():
    """A dict with a tensor having a non-zero leading dim holds (lines 136-137)."""
    assert batch_nonempty(batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)}) is True


def test_invariant_batch_nonempty_empty_or_no_tensor_violates():
    """All-empty tensors / a leading dim of zero / no tensors fall through to
    False (line 138)."""
    assert batch_nonempty(batch={"input_ids": torch.zeros(0, 3, dtype=torch.long)}) is False
    assert batch_nonempty(batch={"input_ids": torch.zeros(0)}) is False  # numel == 0
    assert batch_nonempty(batch={"meta": "no tensors here", "n": 3}) is False
    assert batch_nonempty(batch={}) is False
