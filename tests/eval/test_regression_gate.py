"""Adversarial tests for ``lighttrain.eval.suite.RegressionGate``.

Coverage:

* **Every operator** (<, <=, >, >=, ==, !=) parametrized with closed-form
  pass/fail values.
* **NaN propagation**: ``NaN <op> threshold`` always False; gate
  treats this as a violation across N consecutive checks.
* **Float equality pin**: ``op="=="`` uses exact compare; ``0.1+0.2 == 0.3``
  is False so the gate fires.
* **history_window** delays trigger until N+1 consecutive failures.
* **action="warn"** issues a warning, no raise.
* **action="skip"** is a true no-op.
* **action="abort"** raises InvariantError.
* **Metric absent** → silent skip (no raise, no warning).
* **Unknown op** raises ValueError.
* **Counter resets on pass**.
"""

from __future__ import annotations

import warnings

import pytest

from lighttrain.builtin_plugins.eval.regression_gate import RegressionGate
from lighttrain.callbacks.invariants import InvariantError
from lighttrain.eval.suite import EvalReport


def _gate(**kwargs) -> RegressionGate:
    defaults = dict(metric_name="loss", threshold=1.0, op="<", action="abort")
    defaults.update(kwargs)
    return RegressionGate(**defaults)


# ---------------------------------------------------------------------------
# Operator semantics (closed-form, hand-derived)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op,value,should_pass",
    [
        ("<",  0.5, True),
        ("<",  1.5, False),
        ("<=", 1.0, True),
        ("<=", 1.0001, False),
        (">",  1.5, True),
        (">",  0.5, False),
        (">=", 1.0, True),
        (">=", 0.99, False),
        ("==", 1.0, True),
        ("==", 1.0001, False),
        ("!=", 0.5, True),
        ("!=", 1.0, False),
    ],
)
def test_invariant_operator_semantics_pass_fail(op, value, should_pass):
    """Closed form: for ``threshold=1.0``, each operator/value pair pins
    pass/fail behavior precisely.
    """
    gate = _gate(op=op, action="skip")  # use skip to avoid raise
    gate.check({"loss": value})
    if should_pass:
        assert gate._fail_count == 0
    else:
        assert gate._fail_count == 1


# ---------------------------------------------------------------------------
# Float equality sharp edge
# ---------------------------------------------------------------------------

def test_pin_equality_operator_uses_exact_compare_not_isclose():
    """Pin: ``op="=="`` uses Python's ``==`` (line 247-248 of source).
    The classical float-rounding case ``0.1 + 0.2 == 0.3`` is FALSE, so
    the gate fires.

    If you intentionally switch to ``math.isclose`` for approximate
    equality, update this test AND callers that rely on bit-exact compare.
    """
    gate = _gate(threshold=0.3, op="==", action="skip")
    gate.check({"loss": 0.1 + 0.2})
    assert gate._fail_count == 1  # 0.30000000000000004 != 0.3 → fail


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_pin_nan_value_always_fails_every_op():
    """Pin: NaN compared with any number is always False, so a NaN value
    is a violation under any op (the gate increments _fail_count).

    Goal: document the current behavior — silent NaN passes would be a
    sharp edge.
    """
    gate = _gate(op="<", action="skip")
    gate.check({"loss": float("nan")})
    assert gate._fail_count == 1


# ---------------------------------------------------------------------------
# history_window
# ---------------------------------------------------------------------------

def test_invariant_history_window_delays_trigger_until_n_plus_1_failures():
    """``history_window=2`` means: first 2 failures don't trigger;
    failure #3 does (line 218-219: ``<=`` comparison).

    Closed form:
        check 1 (fail): fail_count=1, 1 <= 2, no trigger
        check 2 (fail): fail_count=2, 2 <= 2, no trigger
        check 3 (fail): fail_count=3, 3 > 2, trigger
    """
    gate = _gate(op="<", threshold=0.5, action="abort", history_window=2)
    gate.check({"loss": 1.0})  # fail 1 — no raise
    gate.check({"loss": 1.0})  # fail 2 — no raise
    with pytest.raises(InvariantError):
        gate.check({"loss": 1.0})  # fail 3 — raises


def test_invariant_history_window_counter_resets_on_pass():
    """A successful check resets ``_fail_count`` to 0 (line 213-214).

    Setup: with history_window=2, fail twice (no raise), pass once
    (counter reset), then fail twice more (still no raise — the counter
    is at 2, not 4).
    """
    gate = _gate(op="<", threshold=0.5, action="abort", history_window=2)
    gate.check({"loss": 1.0})  # fail 1
    gate.check({"loss": 1.0})  # fail 2
    gate.check({"loss": 0.1})  # pass — reset
    assert gate._fail_count == 0
    # Now fail twice more — still under the window
    gate.check({"loss": 1.0})
    gate.check({"loss": 1.0})
    # 2 consecutive failures, window=2 → still no trigger (3rd would).
    assert gate._fail_count == 2


# ---------------------------------------------------------------------------
# action variants
# ---------------------------------------------------------------------------

def test_action_abort_raises_invariant_error():
    """``action="abort"`` raises InvariantError on violation (line 226-231)."""
    gate = _gate(op="<", threshold=0.5, action="abort")
    with pytest.raises(InvariantError) as exc:
        gate.check({"loss": 1.0}, step=42)
    # Message contains metric name, op, threshold, and step
    msg = str(exc.value)
    assert "loss" in msg and "<" in msg and "step=42" in msg


def test_action_warn_emits_warning_no_raise():
    """``action="warn"`` issues a warning but does not raise."""
    gate = _gate(op="<", threshold=0.5, action="warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate.check({"loss": 1.0})
    assert len(caught) == 1
    assert "RegressionGate" in str(caught[0].message)


def test_action_skip_is_silent_no_raise_no_warning():
    """``action="skip"`` is a true no-op — no raise, no warning."""
    gate = _gate(op="<", threshold=0.5, action="skip")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate.check({"loss": 1.0})
    assert caught == []
    # Internal state still records the failure for counter purposes.
    assert gate._fail_count == 1


# ---------------------------------------------------------------------------
# Missing metric
# ---------------------------------------------------------------------------

def test_missing_metric_silently_skipped(capsys):
    """When the metric key is absent from the report, ``check`` returns
    silently (line 206-207).
    """
    gate = _gate(op="<", action="abort")
    gate.check({"other_metric": 0.0})  # 'loss' not present
    # No raise, no state mutation.
    assert gate._fail_count == 0


# ---------------------------------------------------------------------------
# Unknown op
# ---------------------------------------------------------------------------

def test_unknown_op_raises_value_error_on_check():
    """``op="nonsense"`` raises ValueError in ``_evaluate`` (line 251)."""
    gate = _gate(op="nonsense", action="skip")
    with pytest.raises(ValueError, match="unknown op"):
        gate.check({"loss": 1.0})


# ---------------------------------------------------------------------------
# EvalReport vs dict input
# ---------------------------------------------------------------------------

def test_accepts_eval_report_input():
    """The check accepts both an EvalReport instance and a plain dict
    (line 205 of source).
    """
    gate = _gate(op="<", threshold=0.5, action="skip")
    rep = EvalReport(task_name="t", metrics={"loss": 1.0}, step=5)
    gate.check(rep)
    assert gate._fail_count == 1


# ---------------------------------------------------------------------------
# last_value property
# ---------------------------------------------------------------------------

def test_last_value_reflects_most_recent_observation():
    """``last_value`` property returns the latest seen value."""
    gate = _gate(op="<", action="skip")
    assert gate.last_value is None
    gate.check({"loss": 0.5})
    assert gate.last_value == pytest.approx(0.5)
    gate.check({"loss": 0.7})
    assert gate.last_value == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_regression_gate_registered_under_invariant_namespace():
    """Pin: ``('invariant', 'regression_gate')``."""
    from lighttrain.registry import get
    assert get("invariant", "regression_gate") is RegressionGate
