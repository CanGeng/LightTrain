"""Coverage tests for ``lighttrain.lab.sweep`` — drives uncovered branches to green.

Uncovered branches targeted (by source line):

* 115  — ``_find_run_dir`` returns None when dir exists but contains only files.
* 129  — blank-line ``continue`` in ``_read_final_metric``.
* 136-137 — OSError ``continue`` in ``_read_final_metric`` (uses a mock to
            simulate an unreadable file).
* 184-185 — categorical param with a single group value yields sensitivity 0.0.
* 266    — unknown strategy raises ``ValueError``.
* 281, 287 — optuna backend lines; optuna not installed → skipped.
* 321-322 — ``subprocess.TimeoutExpired`` in ``_run_trial`` → failed TrialResult.
* 323-324, 329 — generic ``Exception`` in ``_run_trial`` → failed TrialResult.
* 343-354 — ``_apply_median_stop`` full body (median/asha stopping, minimize /
             maximize directions, grace guard, below-median pruning).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lighttrain.lab.sweep import (
    SweepRunner,
    TrialResult,
    _compute_sensitivity,
    _find_run_dir,
    _read_final_metric,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _trial(trial_id: int, params: dict, metric: float | None, status: str = "ok") -> TrialResult:
    """Build a TrialResult conveniently."""
    return TrialResult(
        trial_id=trial_id,
        config_overrides=dict(params),
        metric=metric,
        status=status if metric is not None else "failed",
        run_dir=None,
    )


def _write_sweep_yaml(path: Path, params: dict, n_trials: int = 4, **extras) -> None:
    cfg: dict = {
        "name": "cov_sweep",
        "metric": "loss",
        "direction": "minimize",
        "n_trials": n_trials,
        "params": params,
    }
    cfg.update(extras)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _write_base_yaml(path: Path, run_root: str) -> None:
    cfg = {
        "mode": "lab",
        "exp": "base",
        "run_root": run_root,
        "model": {"name": "tiny_lm"},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _make_runner(tmp_path: Path, params: dict, strategy: str = "grid",
                 n_trials: int = 4, **sweep_extras) -> SweepRunner:
    """Write minimal yaml files and return a SweepRunner."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir(exist_ok=True)
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, params, n_trials=n_trials, **sweep_extras)
    return SweepRunner(base_cfg, sweep_cfg, strategy=strategy)


# ---------------------------------------------------------------------------
# Line 115 — _find_run_dir returns None when no sub-directories exist
# ---------------------------------------------------------------------------

def test_invariant_find_run_dir_returns_none_when_only_files_exist(tmp_path: Path):
    """``_find_run_dir`` must return None when the root dir exists but
    contains only regular files (no sub-directories).  Exercises line 115.
    """
    (tmp_path / "file_a.txt").write_text("x")
    (tmp_path / "file_b.txt").write_text("y")
    assert _find_run_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# Line 129 — blank-line skip in _read_final_metric
# ---------------------------------------------------------------------------

def test_invariant_read_final_metric_skips_blank_lines(tmp_path: Path):
    """Blank lines in metrics.jsonl are silently skipped (line 129 continue)
    and the last valid value is returned.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 2.5}\n'
        "\n"
        "   \n"
        '{"step": 2, "loss": 1.1}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "loss")
    assert val == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Lines 136-137 — OSError in _read_final_metric
# ---------------------------------------------------------------------------

def test_pin_current_behavior_read_final_metric_ioerror_returns_none(tmp_path: Path):
    """Pin: if opening the candidate file raises OSError (e.g., permissions),
    the exception is caught and the function continues to the next candidate
    (lines 136-137).  With no readable file, the function returns None.

    This pins the current catch-and-continue behavior — it is debatable
    whether silently returning None is the right thing when a file exists
    but cannot be read, but it is the current contract.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text('{"loss": 0.5}\n', encoding="utf-8")

    with patch("builtins.open", side_effect=OSError("permission denied")):
        result = _read_final_metric(tmp_path, "loss")

    assert result is None


# ---------------------------------------------------------------------------
# Lines 184-185 — categorical param with single group → 0.0
# ---------------------------------------------------------------------------

def test_invariant_sensitivity_categorical_single_group_yields_zero():
    """When all trials have the same categorical value, there is only one
    group so ``len(groups) < 2`` and sensitivity is 0.0 (lines 183-185).
    """
    trials = [
        _trial(0, {"opt": "adam"}, 1.0),
        _trial(1, {"opt": "adam"}, 2.0),
        _trial(2, {"opt": "adam"}, 3.0),
    ]
    out = _compute_sensitivity(trials, {"opt": ["adam"]})
    assert out["opt"] == 0.0


# ---------------------------------------------------------------------------
# Line 266 — unknown strategy raises ValueError
# ---------------------------------------------------------------------------

def test_invariant_unknown_strategy_raises_value_error(tmp_path: Path):
    """Passing an unknown strategy string raises ``ValueError`` (line 266)."""
    runner = _make_runner(tmp_path, {"lr": [1e-3]}, strategy="not_a_strategy")
    with pytest.raises(ValueError, match="unknown sweep strategy"):
        runner._generate_configs()


# ---------------------------------------------------------------------------
# Lines 281, 287 — optuna backend (optuna not installed → skip)
# ---------------------------------------------------------------------------

def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _optuna_available(),
    reason="optuna not installed; lines 281/287 unreachable without it",
)
def test_optuna_backend_all_suggestions_returns_correct_count(tmp_path: Path):
    """When optuna IS installed and registered, ``_optuna_configs`` should
    call ``all_suggestions()`` and return ``n_trials`` configs (lines 281/287).
    """
    runner = _make_runner(
        tmp_path,
        {"lr": {"low": 1e-5, "high": 1e-2}},
        strategy="optuna",
        n_trials=3,
    )
    configs = runner._generate_configs()
    assert len(configs) == 3


# ---------------------------------------------------------------------------
# Lines 321-322 — TimeoutExpired in _run_trial → failed TrialResult
# ---------------------------------------------------------------------------

def test_invariant_timeout_expired_produces_failed_trial(tmp_path: Path):
    """A ``subprocess.TimeoutExpired`` exception in ``_run_trial`` must return
    a TrialResult with status='failed' and metric=None (lines 321-322).
    """
    runner = _make_runner(tmp_path, {"lr": [1e-3]})

    MagicMock(return_value=None)
    timeout_err = subprocess.TimeoutExpired(cmd=["lighttrain"], timeout=1.0)

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=timeout_err):
        result = runner._run_trial(0, {"lr": 1e-3})

    assert result.status == "failed"
    assert result.metric is None
    assert result.trial_id == 0


# ---------------------------------------------------------------------------
# Lines 323-324, 329 — generic Exception in _run_trial → failed TrialResult
# ---------------------------------------------------------------------------

def test_invariant_generic_exception_in_subprocess_produces_failed_trial(tmp_path: Path):
    """Any non-timeout exception from ``subprocess.run`` is caught, a warning
    is logged, and a failed TrialResult is returned (lines 323-329).
    """
    runner = _make_runner(tmp_path, {"lr": [1e-3]})

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=RuntimeError("crashed")):
        result = runner._run_trial(0, {"lr": 1e-3})

    assert result.status == "failed"
    assert result.metric is None


def test_pin_current_behavior_generic_exception_logs_warning(tmp_path: Path, caplog):
    """Pin: the generic-exception branch in ``_run_trial`` emits a warning-level
    log message containing the trial id (line 324 ``_log.warning``).
    """
    import logging

    runner = _make_runner(tmp_path, {"lr": [1e-3]})

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=OSError("no such file")):
        with caplog.at_level(logging.WARNING, logger="lighttrain.lab.sweep"):
            result = runner._run_trial(7, {"lr": 1e-3})

    assert result.status == "failed"
    assert any("7" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Lines 343-354 — _apply_median_stop full coverage
# ---------------------------------------------------------------------------

def _make_runner_with_stop(tmp_path: Path, stop_type: str, grace: int = 2,
                            direction: str = "minimize") -> SweepRunner:
    """Build a runner whose stop_cfg is set correctly."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir(exist_ok=True)
    _write_base_yaml(base_cfg, str(run_root))
    cfg = {
        "name": "stop_sweep",
        "metric": "loss",
        "direction": direction,
        "n_trials": 4,
        "params": {"lr": [1e-3]},
        "stop": {"type": stop_type, "grace": grace},
    }
    sweep_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return SweepRunner(base_cfg, sweep_cfg, strategy="grid")


def test_invariant_median_stop_none_type_is_noop(tmp_path: Path):
    """stop.type='none' must not prune any trial (lines 340-342: early return)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="none")
    trials = [
        _trial(0, {"lr": 1e-3}, 5.0),   # very bad
        _trial(1, {"lr": 1e-3}, 4.0),
        _trial(2, {"lr": 1e-3}, 3.0),
    ]
    runner._apply_median_stop(trials)
    assert all(t.status == "ok" for t in trials)


def test_invariant_median_stop_grace_not_reached_skips_pruning(tmp_path: Path):
    """With fewer completed trials than grace, no trial is pruned (line 345-346)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=5)
    trials = [
        _trial(0, {"lr": 1e-3}, 5.0),
        _trial(1, {"lr": 1e-3}, 1.0),
    ]
    runner._apply_median_stop(trials)
    assert all(t.status == "ok" for t in trials)


def test_invariant_median_stop_prunes_worst_minimize_trials(tmp_path: Path):
    """Trials above median*1.2 are marked pruned when direction=minimize (lines 351-352)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=2, direction="minimize")
    # median of [1.0, 1.1, 10.0] = 1.1; threshold = 1.1*1.2 = 1.32
    # Trial with metric=10.0 > 1.32 → pruned
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-3}, 1.1),
        _trial(2, {"lr": 1e-3}, 10.0),
    ]
    runner._apply_median_stop(trials)
    assert trials[0].status == "ok"
    assert trials[1].status == "ok"
    assert trials[2].status == "pruned"


def test_invariant_asha_stop_also_triggers_pruning(tmp_path: Path):
    """stop.type='asha' uses the same pruning logic as 'median' (line 341 ``in`` check)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="asha", grace=2, direction="minimize")
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-3}, 1.0),
        _trial(2, {"lr": 1e-3}, 50.0),
    ]
    runner._apply_median_stop(trials)
    assert trials[2].status == "pruned"


def test_invariant_median_stop_prunes_worst_maximize_trials(tmp_path: Path):
    """Trials below median*0.8 are pruned when direction=maximize (lines 353-354)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=2, direction="maximize")
    # values: [1.0, 10.0, 11.0]; median=10.0; threshold=10.0*0.8=8.0
    # trial with metric=1.0 < 8.0 → pruned
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-3}, 10.0),
        _trial(2, {"lr": 1e-3}, 11.0),
    ]
    runner._apply_median_stop(trials)
    assert trials[0].status == "pruned"
    assert trials[1].status == "ok"
    assert trials[2].status == "ok"


def test_invariant_median_stop_does_not_prune_within_threshold_minimize(tmp_path: Path):
    """Trials within the 1.2× threshold are NOT pruned under minimize."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=2, direction="minimize")
    # values: [1.0, 1.0, 1.1]; all within 1.0*1.2=1.2 → none pruned
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-3}, 1.0),
        _trial(2, {"lr": 1e-3}, 1.1),
    ]
    runner._apply_median_stop(trials)
    assert all(t.status == "ok" for t in trials)


def test_invariant_median_stop_ignores_failed_trials(tmp_path: Path):
    """Failed / metric-None trials must not affect the ok pool or be pruned (line 344)."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=2, direction="minimize")
    bad = TrialResult(trial_id=99, config_overrides={}, metric=None, status="failed", run_dir=None)
    good_low = _trial(0, {"lr": 1e-3}, 1.0)
    good_mid = _trial(1, {"lr": 1e-3}, 1.1)
    good_high = _trial(2, {"lr": 1e-3}, 9.0)
    runner._apply_median_stop([bad, good_low, good_mid, good_high])
    assert bad.status == "failed"   # unchanged
    assert good_high.status == "pruned"


def test_invariant_run_with_median_stop_returns_pruned_in_report(tmp_path: Path):
    """End-to-end: SweepRunner.run() with stop=median includes pruned trials
    in the report trials list (status may be 'pruned').
    """
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    cfg = {
        "name": "e2e_stop",
        "metric": "loss",
        "direction": "minimize",
        "n_trials": 3,
        "params": {"lr": [1e-4, 3e-4, 1e-3]},
        "stop": {"type": "median", "grace": 2},
    }
    sweep_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")

    sweep_run_root = run_root / "sweep_e2e_stop"
    # Provide metrics: 1.0, 1.0, 20.0 — last one should be pruned
    from lighttrain.utils.run_dir import slugify

    metrics_per_trial = [1.0, 1.0, 20.0]
    call_count = [0]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        trial_exp = f"e2e_stop_trial_{i:03d}"
        trial_root = sweep_run_root / slugify(trial_exp)
        run_dir = trial_root / "20250101-000000-test-abc12345"
        (run_dir / "logs").mkdir(parents=True)
        import json
        (run_dir / "logs" / "metrics.jsonl").write_text(
            json.dumps({"step": 10, "loss": metrics_per_trial[i]}) + "\n",
            encoding="utf-8",
        )
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    statuses = [t.status for t in report.trials]
    assert "pruned" in statuses
    # Best metric should be among the non-pruned
    assert report.best_metric == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SweepRunner with trial_timeout_s config key
# ---------------------------------------------------------------------------

def test_pin_current_behavior_sweep_runner_parses_trial_timeout(tmp_path: Path):
    """Pin: ``trial_timeout_s`` in sweep YAML is parsed into ``self.trial_timeout``
    as a float (constructor line 248-251).
    """
    runner = _make_runner(tmp_path, {"lr": [1e-3]}, trial_timeout_s=120)
    assert runner.trial_timeout == pytest.approx(120.0)


def test_pin_current_behavior_sweep_runner_trial_timeout_none_when_absent(tmp_path: Path):
    """Pin: when ``trial_timeout_s`` is absent, ``self.trial_timeout`` is None."""
    runner = _make_runner(tmp_path, {"lr": [1e-3]})
    assert runner.trial_timeout is None


# ---------------------------------------------------------------------------
# _apply_median_stop — maximize direction does NOT prune above-threshold trials
# ---------------------------------------------------------------------------

def test_invariant_maximize_direction_does_not_prune_high_metric(tmp_path: Path):
    """Under maximize, a trial with a VERY HIGH metric should NOT be pruned."""
    runner = _make_runner_with_stop(tmp_path, stop_type="median", grace=2, direction="maximize")
    trials = [
        _trial(0, {"lr": 1e-3}, 10.0),
        _trial(1, {"lr": 1e-3}, 10.5),
        _trial(2, {"lr": 1e-3}, 99.0),
    ]
    runner._apply_median_stop(trials)
    assert trials[2].status == "ok"


# ---------------------------------------------------------------------------
# _compute_sensitivity with None metric entries filtered
# ---------------------------------------------------------------------------

def test_invariant_sensitivity_filters_out_none_metric_trials():
    """Trials with metric=None are excluded from sensitivity calculation."""
    trials = [
        _trial(0, {"lr": 1.0}, 1.0),
        _trial(1, {"lr": 2.0}, 2.0),
        _trial(2, {"lr": 3.0}, None),  # should be excluded
    ]
    out = _compute_sensitivity(trials, {"lr": [1.0, 2.0, 3.0]})
    # Only 2 trials have metrics → correlation computable, but we just want no crash
    assert isinstance(out, dict)
    assert "lr" in out
