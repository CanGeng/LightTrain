"""Adversarial tests for ``lighttrain.eval.suite.Evaluator``.

Coverage:

* **should_eval** at boundaries (step % N == 0, ``eval_every_n_steps=0``).
* **Force=True overrides should_eval**.
* **Run aggregates metrics**: single-task → bare keys; multi-task → prefixed.
* **Failing task warns + continues** to remaining tasks.
* **on_report callback called with the report**; its exception swallowed.
* **_last_eval_step prevents duplicate runs at the same step**.
* **EvalReport defaults**: metrics={} dict, timestamp populated.
"""

from __future__ import annotations

import warnings

import pytest

from lighttrain.eval.suite import EvalReport, Evaluator


class _Task:
    def __init__(self, name: str, result: dict) -> None:
        self.name = name
        self._result = dict(result)
        self.calls = 0

    def run(self, model, *, device=None, step=None):
        self.calls += 1
        return dict(self._result)


class _RaisingTask:
    name = "raiser"

    def run(self, model, *, device=None, step=None):
        raise RuntimeError("scheduled failure")


# ---------------------------------------------------------------------------
# should_eval
# ---------------------------------------------------------------------------

def test_invariant_should_eval_returns_true_on_step_multiple():
    """``should_eval(N)`` returns True when ``step % eval_every_n_steps == 0``."""
    e = Evaluator(tasks=[], eval_every_n_steps=10)
    assert e.should_eval(10) is True
    assert e.should_eval(20) is True


def test_invariant_should_eval_returns_false_on_non_multiple():
    """``should_eval(7)`` returns False when step is not a multiple of 10."""
    e = Evaluator(tasks=[], eval_every_n_steps=10)
    assert e.should_eval(7) is False


def test_pin_eval_every_n_steps_zero_disables_evaluation():
    """Pin: ``eval_every_n_steps=0`` returns False for every step
    (line 86-87).
    """
    e = Evaluator(tasks=[], eval_every_n_steps=0)
    for s in (0, 10, 100, 1000):
        assert e.should_eval(s) is False


def test_invariant_should_eval_false_on_repeated_step():
    """After ``run(model, step=10)``, calling ``should_eval(10)`` again
    returns False (the gate prevents repeated runs at the same step,
    line 88).
    """
    t = _Task("t", {"acc": 0.9})
    e = Evaluator(tasks=[t], eval_every_n_steps=10)
    e.run(model=None, step=10)
    # Second call at same step is gated
    assert e.should_eval(10) is False


# ---------------------------------------------------------------------------
# run() metric aggregation
# ---------------------------------------------------------------------------

def test_invariant_single_task_run_uses_bare_metric_keys():
    """When there is exactly 1 task, metrics use the bare keys
    (no task-name prefix — line 124-127 of source).
    """
    t = _Task("t1", {"acc": 0.9, "loss": 0.1})
    e = Evaluator(tasks=[t], eval_every_n_steps=10)
    rep = e.run(model=None, step=10, force=True)
    assert rep is not None
    assert set(rep.metrics) == {"acc", "loss"}
    assert rep.metrics["acc"] == pytest.approx(0.9)


def test_invariant_multi_task_run_uses_prefixed_metric_keys():
    """Multi-task → metrics are prefixed by task name (line 120-123)."""
    a = _Task("a", {"acc": 0.9})
    b = _Task("b", {"acc": 0.5})
    e = Evaluator(tasks=[a, b], eval_every_n_steps=10)
    rep = e.run(model=None, step=10, force=True)
    assert rep is not None
    assert set(rep.metrics) == {"a/acc", "b/acc"}
    assert rep.metrics["a/acc"] == pytest.approx(0.9)
    assert rep.metrics["b/acc"] == pytest.approx(0.5)


def test_force_true_bypasses_should_eval_gate():
    """``run(force=True)`` runs tasks even when ``should_eval`` would be False."""
    t = _Task("t", {"acc": 0.9})
    e = Evaluator(tasks=[t], eval_every_n_steps=10)
    rep = e.run(model=None, step=7, force=True)  # 7 not a multiple of 10
    assert rep is not None
    assert t.calls == 1


def test_run_returns_none_when_should_eval_false_and_not_forced():
    """``run`` returns None when ``should_eval(step)`` is False and force
    is False (line 104-105).
    """
    t = _Task("t", {"acc": 0.9})
    e = Evaluator(tasks=[t], eval_every_n_steps=10)
    out = e.run(model=None, step=7)
    assert out is None
    assert t.calls == 0


# ---------------------------------------------------------------------------
# Non-numeric metric value filtering
# ---------------------------------------------------------------------------

def test_invariant_non_numeric_metric_values_dropped():
    """Pin: only int/float metric values are included in the aggregated
    report (line 122/126 isinstance check).

    Setup: task returns {"acc": 0.9, "label": "good", "task_name": "t"}.
    Expected: aggregated metrics contain "acc" only; "label" and
    "task_name" are dropped.
    """
    t = _Task("t", {"acc": 0.9, "label": "good", "task_name": "t"})
    e = Evaluator(tasks=[t], eval_every_n_steps=10)
    rep = e.run(model=None, step=10, force=True)
    assert rep is not None
    assert "acc" in rep.metrics
    assert "label" not in rep.metrics
    assert "task_name" not in rep.metrics


# ---------------------------------------------------------------------------
# Failing task tolerance
# ---------------------------------------------------------------------------

def test_invariant_failing_task_warns_and_other_tasks_continue():
    """A task that raises emits a warning; the evaluator continues to the
    next task (line 113-116 of source).

    Setup: [raising_task, good_task].
    Expected: 1 warning issued; good task is called.
    """
    good = _Task("good", {"acc": 0.5})
    e = Evaluator(tasks=[_RaisingTask(), good], eval_every_n_steps=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = e.run(model=None, step=10, force=True)
    # At least one warning about the failing task
    assert any("raiser" in str(w.message) for w in caught)
    # Good task ran
    assert good.calls == 1
    # Report still produced
    assert rep is not None


# ---------------------------------------------------------------------------
# on_report callback
# ---------------------------------------------------------------------------

def test_invariant_on_report_called_with_the_report():
    """``on_report`` is invoked with the produced EvalReport (line 134-138)."""
    received: list[EvalReport] = []

    def hook(r: EvalReport) -> None:
        received.append(r)

    t = _Task("t", {"acc": 0.9})
    e = Evaluator(tasks=[t], eval_every_n_steps=10, on_report=hook)
    e.run(model=None, step=10, force=True)
    assert len(received) == 1
    assert received[0].metrics["acc"] == pytest.approx(0.9)


def test_on_report_exception_swallowed_does_not_break_run():
    """An on_report hook that raises does NOT prevent the run from
    returning the report (line 137-138).
    """
    def bad_hook(r: EvalReport) -> None:
        raise RuntimeError("hook boom")

    t = _Task("t", {"acc": 0.9})
    e = Evaluator(tasks=[t], eval_every_n_steps=10, on_report=bad_hook)
    rep = e.run(model=None, step=10, force=True)
    assert rep is not None
    assert rep.metrics["acc"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# EvalReport defaults
# ---------------------------------------------------------------------------

def test_eval_report_default_metrics_is_empty_dict():
    """``EvalReport(task_name="t")`` has metrics={}, step=None, timestamp set."""
    r = EvalReport(task_name="t")
    assert r.metrics == {}
    assert r.step is None
    assert isinstance(r.timestamp, float)


def test_eval_report_timestamp_is_unix_seconds():
    """``timestamp`` defaults to ``time.time()`` (Unix epoch seconds)."""
    import time
    before = time.time()
    r = EvalReport(task_name="t")
    after = time.time()
    assert before - 1.0 <= r.timestamp <= after + 1.0
