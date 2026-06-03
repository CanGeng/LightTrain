"""Lab tooling.

* :mod:`~.estimate`         — pre-flight resource estimate
* :mod:`~.sweep`            — grid / random / median-stop hyperparameter search
* :mod:`~.compare`          — config diff + metric alignment across runs
* :mod:`~.fork`             — checkpoint duplication with lineage edge
* :mod:`~.ab_test`          — same-seed dual-stream A/B comparison
* :mod:`~.hypothesis`       — structured hypothesis logging
* :mod:`~.decision_record`  — opt-in ADR-style decision log
* :mod:`~.auto_report`      — Markdown report generation
"""

from __future__ import annotations

from .ab_test import ABReport, ab_test
from .auto_report import (
    render_compare_markdown,
    render_sweep_markdown,
    write_sweep_report,
)
from .compare import CompareReport, compare, render_ascii, render_png
from .decision_record import DecisionEntry, DecisionRecord
from .estimate import EstimateReport, OffloadEstimate, estimate
from .fork import ForkReport, fork
from .hypothesis import HypothesisEntry, HypothesisLog
from .sweep import SweepReport, SweepRunner, TrialResult

__all__ = [
    "estimate",
    "EstimateReport",
    "OffloadEstimate",
    "SweepRunner",
    "SweepReport",
    "TrialResult",
    "compare",
    "render_ascii",
    "render_png",
    "CompareReport",
    "fork",
    "ForkReport",
    "ab_test",
    "ABReport",
    "HypothesisLog",
    "HypothesisEntry",
    "DecisionRecord",
    "DecisionEntry",
    "render_sweep_markdown",
    "render_compare_markdown",
    "write_sweep_report",
]
