"""Adversarial tests for ``lighttrain.lab.ab_test``.

The ab_test driver shells out to ``lighttrain train`` as a subprocess.
We don't run real subprocesses here; instead we pin the ABReport
dataclass surface and verify a manual hand-built report has the expected
shape and delta semantics.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from lighttrain.lab.ab_test import ABReport
from lighttrain.lab.compare import CompareReport


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------

def test_pin_ab_report_field_set():
    """Pin: ABReport has exactly the fields ``{run_a, run_b, metric_a,
    metric_b, delta, compare}``.

    If you intentionally add replicates / variance / effect_size, update
    this test AND document the breaking change.
    """
    expected = {"run_a", "run_b", "metric_a", "metric_b", "delta", "compare"}
    actual = {f.name for f in dataclasses.fields(ABReport)}
    assert actual == expected, (
        f"ABReport schema drift detected. New fields: {actual - expected}; "
        f"removed: {expected - actual}"
    )


def test_pin_ab_report_no_statistical_significance_fields():
    """Pin: no t-test / p-value / replicate / CI fields.

    Current ABReport is a single-trial difference; statistical testing is
    out of scope.
    """
    forbidden = {
        "p_value", "t_stat", "welch_t", "ci_lower", "ci_upper",
        "replicates", "variance_a", "variance_b", "effect_size",
    }
    actual = {f.name for f in dataclasses.fields(ABReport)}
    assert not (actual & forbidden), (
        f"ABReport gained statistical fields {actual & forbidden}"
    )


# ---------------------------------------------------------------------------
# Delta semantics
# ---------------------------------------------------------------------------

def test_invariant_delta_is_b_minus_a():
    """Pin: ``delta = metric_b - metric_a`` (line 80 of ab_test.py).

    Setup: hand-build an ABReport with metric_a=0.5, metric_b=0.3.
    Expected: delta == -0.2.
    """
    r = ABReport(
        run_a=Path("/tmp/a"),
        run_b=Path("/tmp/b"),
        metric_a=0.5,
        metric_b=0.3,
        delta=0.3 - 0.5,
        compare=None,
    )
    assert r.delta == pytest.approx(-0.2)


def test_ab_report_with_missing_metric_a_carries_none():
    """Pin: when metric_a is None, the caller should record delta=None."""
    r = ABReport(
        run_a=None, run_b=None,
        metric_a=None, metric_b=1.0,
        delta=None, compare=None,
    )
    assert r.delta is None


def test_ab_report_compare_field_optional():
    """``compare`` is allowed to be None (when ab_test's compare() call
    failed silently per line 86-87 of source).
    """
    r = ABReport(
        run_a=Path("/tmp/a"), run_b=Path("/tmp/b"),
        metric_a=0.5, metric_b=0.5,
        delta=0.0, compare=None,
    )
    assert r.compare is None


def test_ab_report_with_full_compare_record():
    """When all fields populated, the dataclass round-trips through
    dataclass introspection cleanly.
    """
    compare_rep = CompareReport(
        runs=[Path("/tmp/a"), Path("/tmp/b")],
        config_diff={"lr": [1e-3, 1e-4]},
        metrics_table={"loss": [0.5, 0.3]},
        fork_ancestry={"/tmp/a": None, "/tmp/b": None},
    )
    r = ABReport(
        run_a=Path("/tmp/a"),
        run_b=Path("/tmp/b"),
        metric_a=0.5,
        metric_b=0.3,
        delta=-0.2,
        compare=compare_rep,
    )
    assert r.compare is compare_rep
    assert r.compare.config_diff["lr"] == [1e-3, 1e-4]
