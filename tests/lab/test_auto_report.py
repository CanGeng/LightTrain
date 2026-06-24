"""Adversarial tests for ``lighttrain.lab.auto_report`` Markdown renderers.

What is pinned here:

* ``render_sweep_markdown`` — the "no successful trials" empty-table branch
  (line 59), the header counts, the best-metric line, the all-trials overflow
  table, the sensitivity table ordering, and the best-config YAML block.
* ``_guess_metric_key`` — both the empty-``ok`` and non-empty-``ok`` branches
  return the literal ``"metric"`` (the function is effectively a constant; this
  is flagged below as suspected dead logic).
* ``write_sweep_report`` — the ``out_path is None`` default-path branch
  (lines 115-117, written under ``runs/sweep_<name>/``) and the explicit
  ``out_path`` branch, including parent-dir creation.
* ``render_compare_markdown`` — run list with and without fork ancestry,
  the config-diff table vs. the "identical configs" fallback, the metrics
  table with ``None`` cells rendered as an em dash, and sorted key ordering.

All assertions are on pure string output or written files; nothing here spawns
a subprocess, touches a GPU, or needs an optional dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lighttrain.lab.auto_report import (
    _guess_metric_key,
    render_compare_markdown,
    render_sweep_markdown,
    write_sweep_report,
)
from lighttrain.lab.compare import CompareReport
from lighttrain.lab.sweep import SweepReport, TrialResult

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _trial(
    trial_id: int,
    *,
    metric: float | None,
    status: str = "ok",
    overrides: dict | None = None,
) -> TrialResult:
    """Construct a ``TrialResult`` with defaulted run_dir/overrides."""
    return TrialResult(
        trial_id=trial_id,
        config_overrides=dict(overrides or {}),
        metric=metric,
        status=status,
        run_dir=None,
    )


def _sweep_report(
    *,
    trials: list[TrialResult],
    best_config: dict | None = None,
    best_metric: float | None = None,
    direction: str = "minimize",
    sensitivity: dict | None = None,
    sweep_name: str = "lr_sweep",
    strategy: str = "grid",
) -> SweepReport:
    """Construct a ``SweepReport`` with sensible defaults for rendering."""
    return SweepReport(
        sweep_name=sweep_name,
        strategy=strategy,
        trials=trials,
        best_config=dict(best_config or {}),
        best_metric=best_metric,
        direction=direction,
        sensitivity=dict(sensitivity or {}),
    )


def _compare_report(
    *,
    runs: list[Path],
    config_diff: dict | None = None,
    metrics_table: dict | None = None,
    fork_ancestry: dict | None = None,
) -> CompareReport:
    """Construct a ``CompareReport`` with empty defaults for the maps."""
    return CompareReport(
        runs=runs,
        config_diff=dict(config_diff or {}),
        metrics_table=dict(metrics_table or {}),
        fork_ancestry=dict(fork_ancestry or {}),
    )


# ---------------------------------------------------------------------------
# render_sweep_markdown — header + counts
# ---------------------------------------------------------------------------


def test_invariant_sweep_header_counts_ok_pruned_failed():
    """Header line tallies ok / pruned / failed statuses independently."""
    trials = [
        _trial(0, metric=1.0, status="ok"),
        _trial(1, metric=2.0, status="ok"),
        _trial(2, metric=None, status="pruned"),
        _trial(3, metric=None, status="failed"),
    ]
    md = render_sweep_markdown(_sweep_report(trials=trials))
    assert "# Sweep report: lr_sweep" in md
    assert "**Strategy:** grid  " in md
    assert "**Metric:** minimize `metric`  " in md
    assert "**Trials:** 4 total (2 ok, 1 pruned, 1 failed)  " in md


def test_invariant_sweep_best_metric_line_uses_6g():
    """``best_metric`` is rendered with ``:.6g`` formatting when present."""
    md = render_sweep_markdown(
        _sweep_report(trials=[_trial(0, metric=0.123456789)], best_metric=0.123456789)
    )
    assert "**Best metric:** `0.123457`  " in md


def test_invariant_sweep_best_metric_line_absent_when_none():
    """No best-metric line is emitted when ``best_metric is None``."""
    md = render_sweep_markdown(_sweep_report(trials=[], best_metric=None))
    assert "**Best metric:**" not in md


# ---------------------------------------------------------------------------
# render_sweep_markdown — Top-K table & the empty branch (line 59)
# ---------------------------------------------------------------------------


def test_invariant_sweep_no_successful_trials_emits_placeholder():
    """Line 59: with no ``ok``+metric trials, the placeholder line is used and
    no table header is emitted.
    """
    trials = [
        _trial(0, metric=None, status="failed"),
        _trial(1, metric=5.0, status="pruned"),  # pruned excluded from ok_trials
    ]
    md = render_sweep_markdown(_sweep_report(trials=trials))
    assert "_No successful trials._" in md
    assert "| Rank | Trial | Metric |" not in md
    # Top-0 because min(top_k, len(ok_trials)) == 0
    assert "## Top-0 trials" in md


def test_invariant_sweep_ok_with_metric_none_excluded_from_table():
    """An ``ok`` trial whose metric is ``None`` is filtered out of the table,
    falling back to the placeholder when it is the only candidate.
    """
    md = render_sweep_markdown(
        _sweep_report(trials=[_trial(0, metric=None, status="ok")])
    )
    assert "_No successful trials._" in md


def test_invariant_sweep_topk_table_sorted_minimize_ascending():
    """For ``direction='minimize'`` the table is sorted ascending by metric and
    truncated to ``top_k`` rows.
    """
    trials = [
        _trial(0, metric=3.0, overrides={"lr": 0.3}),
        _trial(1, metric=1.0, overrides={"lr": 0.1}),
        _trial(2, metric=2.0, overrides={"lr": 0.2}),
    ]
    md = render_sweep_markdown(_sweep_report(trials=trials, direction="minimize"), top_k=2)
    lines = md.splitlines()
    rank_rows = [ln for ln in lines if ln.startswith("| 1 |") or ln.startswith("| 2 |")]
    assert rank_rows[0].startswith("| 1 | 1 |")  # trial_id 1, metric 1.0 is best
    assert rank_rows[1].startswith("| 2 | 2 |")  # trial_id 2, metric 2.0 second
    # top_k=2 truncates: trial 0 (metric 3.0) absent
    assert "## Top-2 trials" in md
    assert not any(ln.startswith("| 3 |") for ln in lines)


def test_invariant_sweep_topk_table_sorted_maximize_descending():
    """For ``direction='maximize'`` the table is sorted descending by metric."""
    trials = [
        _trial(0, metric=3.0, overrides={"lr": 0.3}),
        _trial(1, metric=1.0, overrides={"lr": 0.1}),
        _trial(2, metric=2.0, overrides={"lr": 0.2}),
    ]
    md = render_sweep_markdown(_sweep_report(trials=trials, direction="maximize"), top_k=5)
    lines = md.splitlines()
    rank_rows = [ln for ln in lines if ln[:5] in ("| 1 |", "| 2 |", "| 3 |")]
    assert rank_rows[0].startswith("| 1 | 0 |")  # metric 3.0 highest
    assert rank_rows[2].startswith("| 3 | 1 |")  # metric 1.0 lowest


def test_invariant_sweep_topk_param_columns_union_and_missing_dash():
    """Param columns are the sorted union of all override keys; trials lacking a
    key show an em dash in that cell.
    """
    trials = [
        _trial(0, metric=1.0, overrides={"lr": 0.1}),
        _trial(1, metric=2.0, overrides={"wd": 0.01}),
    ]
    md = render_sweep_markdown(_sweep_report(trials=trials), top_k=5)
    assert "| Rank | Trial | Metric | lr | wd |" in md
    # trial 0 (best, rank 1) has lr but no wd → "—" in wd column
    row0 = next(ln for ln in md.splitlines() if ln.startswith("| 1 |"))
    assert "0.1" in row0 and "—" in row0


# ---------------------------------------------------------------------------
# render_sweep_markdown — All-trials overflow table
# ---------------------------------------------------------------------------


def test_invariant_sweep_all_trials_table_appears_past_top_k():
    """When ``len(trials) > top_k`` an "All trials" table lists every trial with
    its status; ``None`` metrics render as an em dash.
    """
    trials = [_trial(i, metric=float(i)) for i in range(3)]
    trials.append(_trial(3, metric=None, status="failed"))
    md = render_sweep_markdown(_sweep_report(trials=trials), top_k=2)
    assert "## All trials" in md
    assert "| Trial | Status | Metric |" in md
    assert "| 3 | failed | `—` |" in md


def test_invariant_sweep_all_trials_table_absent_when_within_top_k():
    """No "All trials" section when ``len(trials) <= top_k``."""
    trials = [_trial(i, metric=float(i)) for i in range(2)]
    md = render_sweep_markdown(_sweep_report(trials=trials), top_k=5)
    assert "## All trials" not in md


# ---------------------------------------------------------------------------
# render_sweep_markdown — Sensitivity table
# ---------------------------------------------------------------------------


def test_invariant_sweep_sensitivity_sorted_descending():
    """Sensitivity rows are sorted by value descending and formatted ``:.4f``."""
    md = render_sweep_markdown(
        _sweep_report(
            trials=[_trial(0, metric=1.0)],
            sensitivity={"lr": 0.2, "wd": 0.9},
        )
    )
    assert "## Parameter sensitivity" in md
    lines = md.splitlines()
    wd_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `wd`"))
    lr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `lr`"))
    assert wd_idx < lr_idx  # 0.9 before 0.2
    assert "| `wd` | 0.9000 |" in md


def test_invariant_sweep_sensitivity_absent_when_empty():
    """No sensitivity section is emitted when the dict is empty."""
    md = render_sweep_markdown(_sweep_report(trials=[_trial(0, metric=1.0)]))
    assert "## Parameter sensitivity" not in md


# ---------------------------------------------------------------------------
# render_sweep_markdown — Best config YAML block
# ---------------------------------------------------------------------------


def test_invariant_sweep_best_config_yaml_block():
    """``best_config`` is rendered as a fenced ``yaml`` block of ``k: v`` pairs."""
    md = render_sweep_markdown(
        _sweep_report(
            trials=[_trial(0, metric=1.0)],
            best_config={"optim.lr": 0.001, "optim.weight_decay": 0.0},
        )
    )
    assert "## Best configuration overrides" in md
    assert "```yaml" in md
    assert "optim.lr: 0.001" in md
    assert "optim.weight_decay: 0.0" in md


def test_invariant_sweep_best_config_block_absent_when_empty():
    """No best-config section when ``best_config`` is falsy."""
    md = render_sweep_markdown(_sweep_report(trials=[_trial(0, metric=1.0)]))
    assert "## Best configuration overrides" not in md


# ---------------------------------------------------------------------------
# _guess_metric_key
# ---------------------------------------------------------------------------


def test_pin_current_behavior_guess_metric_key_always_returns_metric():
    """SUSPECTED DEAD LOGIC: ``_guess_metric_key`` returns the literal string
    ``"metric"`` for both the empty-``ok`` and non-empty-``ok`` branches, so the
    ``if not ok`` guard (line 102-103) has no observable effect. Pinned as
    current behavior; flagged in suspected_bugs.
    """
    empty = _sweep_report(trials=[_trial(0, metric=None, status="failed")])
    nonempty = _sweep_report(trials=[_trial(0, metric=1.0)])
    assert _guess_metric_key(empty) == "metric"
    assert _guess_metric_key(nonempty) == "metric"


# ---------------------------------------------------------------------------
# write_sweep_report
# ---------------------------------------------------------------------------


def test_invariant_write_sweep_report_explicit_path_creates_parents(tmp_path: Path):
    """An explicit ``out_path`` is written verbatim, creating missing parents."""
    out = tmp_path / "nested" / "deeper" / "report.md"
    written = write_sweep_report(
        _sweep_report(trials=[_trial(0, metric=1.0)]), out_path=out
    )
    assert written == out
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert body.startswith("# Sweep report: lr_sweep")
    # Round-trips exactly what render produced.
    assert body == render_sweep_markdown(
        _sweep_report(trials=[_trial(0, metric=1.0)])
    )


def test_invariant_write_sweep_report_default_path_under_runs(tmp_path: Path):
    """Lines 115-117: ``out_path=None`` writes to ``runs/sweep_<name>/sweep_report.md``
    relative to cwd; we chdir into a tmp dir to keep the repo clean.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        written = write_sweep_report(
            _sweep_report(trials=[_trial(0, metric=1.0)], sweep_name="demo")
        )
    finally:
        os.chdir(cwd)
    expected = tmp_path / "runs" / "sweep_demo" / "sweep_report.md"
    assert written == Path("runs") / "sweep_demo" / "sweep_report.md"
    assert expected.exists()
    assert expected.read_text(encoding="utf-8").startswith("# Sweep report: demo")


def test_invariant_write_sweep_report_respects_top_k(tmp_path: Path):
    """``top_k`` passed to ``write_sweep_report`` flows into the rendered file."""
    trials = [_trial(i, metric=float(i)) for i in range(4)]
    out = tmp_path / "r.md"
    write_sweep_report(_sweep_report(trials=trials), out_path=out, top_k=1)
    body = out.read_text(encoding="utf-8")
    assert "## Top-1 trials" in body
    assert "## All trials" in body  # 4 trials > top_k=1


# ---------------------------------------------------------------------------
# render_compare_markdown — run list + fork ancestry
# ---------------------------------------------------------------------------


def test_invariant_compare_header_and_run_count():
    """Header and run count reflect ``len(report.runs)``."""
    runs = [Path("/runs/a"), Path("/runs/b")]
    md = render_compare_markdown(_compare_report(runs=runs))
    assert md.startswith("# Compare report")
    assert "**Runs compared:** 2" in md


def test_invariant_compare_run_list_with_and_without_fork():
    """Each run is listed; a present ``fork_ancestry`` entry adds a fork suffix,
    while a ``None``/absent parent omits it (lines 136-139).
    """
    child = Path("/runs/child")
    plain = Path("/runs/plain")
    md = render_compare_markdown(
        _compare_report(
            runs=[child, plain],
            fork_ancestry={str(child): "/runs/parent", str(plain): None},
        )
    )
    assert f"- Run 0: `{child}` ← fork of `/runs/parent`" in md
    assert f"- Run 1: `{plain}`" in md
    # plain run has no fork suffix
    assert "Run 1: `/runs/plain` ← fork" not in md


# ---------------------------------------------------------------------------
# render_compare_markdown — config diff
# ---------------------------------------------------------------------------


def test_invariant_compare_config_diff_table_sorted(tmp_path: Path):
    """Config-diff table emits one column per run and sorts rows by key
    (lines 143-156).
    """
    runs = [Path("/runs/a"), Path("/runs/b")]
    md = render_compare_markdown(
        _compare_report(
            runs=runs,
            config_diff={"optim.lr": [0.1, 0.2], "data.seed": [1, 2]},
        )
    )
    assert "## Configuration differences" in md
    assert "_Only fields that differ across runs are shown._" in md
    assert "| Key | Run 0 | Run 1 |" in md
    lines = md.splitlines()
    seed_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `data.seed`"))
    lr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `optim.lr`"))
    assert seed_idx < lr_idx  # sorted: data.seed before optim.lr
    assert "| `optim.lr` | `0.1` | `0.2` |" in md


def test_invariant_compare_identical_configs_fallback():
    """Empty ``config_diff`` triggers the "identical configurations" fallback
    (lines 157-161), not a table.
    """
    md = render_compare_markdown(_compare_report(runs=[Path("a"), Path("b")]))
    assert "_All compared runs share identical configurations._" in md
    assert "| Key |" not in md


# ---------------------------------------------------------------------------
# render_compare_markdown — metrics table
# ---------------------------------------------------------------------------


def test_invariant_compare_metrics_table_none_renders_dash():
    """Metrics table renders numeric values with ``:.6g`` and ``None`` as an em
    dash; rows are sorted by metric name (lines 164-177).
    """
    runs = [Path("/runs/a"), Path("/runs/b")]
    md = render_compare_markdown(
        _compare_report(
            runs=runs,
            metrics_table={"loss": [0.123456789, None], "acc": [0.5, 0.9]},
        )
    )
    assert "## Final metrics" in md
    assert "| Metric | Run 0 | Run 1 |" in md
    lines = md.splitlines()
    acc_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `acc`"))
    loss_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| `loss`"))
    assert acc_idx < loss_idx  # sorted: acc before loss
    assert "| `loss` | `0.123457` | — |" in md
    assert "| `acc` | `0.5` | `0.9` |" in md


def test_invariant_compare_metrics_table_absent_when_empty():
    """No "Final metrics" section when ``metrics_table`` is empty."""
    md = render_compare_markdown(_compare_report(runs=[Path("a")]))
    assert "## Final metrics" not in md


def test_invariant_compare_zero_runs_renders_without_error():
    """Edge: an empty run list still produces a valid header with count 0."""
    md = render_compare_markdown(_compare_report(runs=[]))
    assert "**Runs compared:** 0" in md
    assert "_All compared runs share identical configurations._" in md


# ---------------------------------------------------------------------------
# Cross-cutting: rendered output is newline-joined plain text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sweep", "compare"])
def test_invariant_renderers_return_newline_joined_str(kind):
    """Both renderers return a ``str`` produced by ``"\\n".join``; each ends with
    a trailing newline because the body closes with an empty ``""`` line.
    """
    if kind == "sweep":
        out = render_sweep_markdown(_sweep_report(trials=[_trial(0, metric=1.0)]))
    else:
        out = render_compare_markdown(_compare_report(runs=[Path("a")]))
    assert isinstance(out, str)
    assert "\n" in out
    assert out.endswith("\n")
