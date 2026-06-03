"""EvalSuite / RegressionGate tests (M6)."""

from __future__ import annotations

import warnings

import pytest

from lighttrain.builtin_plugins.eval.regression_gate import RegressionGate
from lighttrain.eval.suite import EvalReport, Evaluator

# ---- Helpers ---------------------------------------------------------------

class _ConstTask:
    """Fake EvalTask returning a fixed score."""
    name = "const_task"

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def run(self, model, *, device=None, step=None):
        return {"task_name": self.name, "mean_score": self._score}


# ---- Evaluator ------------------------------------------------------------

def test_evaluator_should_eval_true_at_correct_step():
    ev = Evaluator([_ConstTask()], eval_every_n_steps=10)
    assert ev.should_eval(10)
    assert not ev.should_eval(9)


def test_evaluator_run_returns_report():
    ev = Evaluator([_ConstTask(score=0.42)], eval_every_n_steps=5)
    report = ev.run(model=None, step=5, force=True)
    assert report is not None
    assert isinstance(report, EvalReport)
    assert abs(report.metrics.get("mean_score", -1) - 0.42) < 1e-6


def test_evaluator_no_run_when_step_not_due():
    ev = Evaluator([_ConstTask()], eval_every_n_steps=100)
    assert ev.run(None, step=7) is None


def test_evaluator_on_report_callback_called():
    received = []
    ev = Evaluator([_ConstTask()], eval_every_n_steps=1, on_report=received.append)
    ev.run(None, step=1)
    assert len(received) == 1


# ---- RegressionGate -------------------------------------------------------

def test_regression_gate_passes_ok():
    gate = RegressionGate(metric_name="mean_score", threshold=0.5, op=">")
    report = EvalReport(task_name="t", metrics={"mean_score": 0.8})
    gate.check(report)   # should not raise


def test_regression_gate_abort_raises():
    from lighttrain.invariants import InvariantError
    gate = RegressionGate(metric_name="mean_score", threshold=0.5, op=">", action="abort")
    report = EvalReport(task_name="t", metrics={"mean_score": 0.3})
    with pytest.raises((InvariantError, RuntimeError)):
        gate.check(report)


def test_regression_gate_warn_no_raise():
    gate = RegressionGate(metric_name="val_loss", threshold=1.0, op="<", action="warn")
    report = EvalReport(task_name="t", metrics={"val_loss": 5.0})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gate.check(report)
    assert len(w) == 1


def test_regression_gate_missing_metric_ignored():
    gate = RegressionGate(metric_name="missing_metric", threshold=0.5, op=">")
    report = EvalReport(task_name="t", metrics={"other": 0.1})
    gate.check(report)   # silent — metric absent


def test_regression_gate_last_value_tracked():
    gate = RegressionGate(metric_name="x", threshold=0.5, op=">", action="warn")
    report = EvalReport(task_name="t", metrics={"x": 0.7})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        gate.check(report)
    assert gate.last_value is not None


def test_regression_gate_registers():
    from lighttrain.registry import get as resolve
    cls = resolve("invariant", "regression_gate")
    assert cls is RegressionGate
