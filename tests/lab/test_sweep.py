"""Adversarial tests for ``lighttrain.lab.sweep``.

Focus on the pure / testable helpers that do not require subprocess spawn:

* ``_grid_configs`` — Cartesian product correctness.
* ``_random_configs`` — seed-determinism, distinct seeds → distinct samples,
  int / log-scale dispatch.
* ``_compute_sensitivity`` — numeric correlation formula, constant-param case,
  too-few-trials edge case.
* ``_read_final_metric`` — reads last occurrence; tolerates malformed lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lighttrain.lab.sweep import (
    SweepReport,
    SweepRunner,
    TrialResult,
    _compute_sensitivity,
    _find_run_dir,
    _grid_configs,
    _random_configs,
    _read_final_metric,
)

# ---------------------------------------------------------------------------
# _grid_configs
# ---------------------------------------------------------------------------

def test_invariant_grid_generates_cartesian_product():
    """Closed form: ``{a:[1,2], b:["x","y"]}`` → 4 configs covering every
    pair exactly once.
    """
    out = _grid_configs({"a": [1, 2], "b": ["x", "y"]})
    expected = [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert len(out) == 4
    assert sorted(out, key=str) == sorted(expected, key=str)


def test_grid_with_no_list_params_returns_singleton_empty_dict():
    """Pin: when no params are list-valued, returns ``[{}]`` (line 62-63
    of sweep.py).
    """
    assert _grid_configs({"a": 1, "b": "x"}) == [{}]


def test_grid_with_single_param_yields_len_equal_to_choices():
    """One list-param of length 3 → 3 configs."""
    out = _grid_configs({"lr": [1e-3, 1e-4, 1e-5]})
    assert len(out) == 3
    assert {c["lr"] for c in out} == {1e-3, 1e-4, 1e-5}


def test_grid_param_count_equals_product_of_lengths():
    """{a:[1,2,3], b:[x,y]} → 6 configs."""
    out = _grid_configs({"a": [1, 2, 3], "b": ["x", "y"]})
    assert len(out) == 6


# ---------------------------------------------------------------------------
# _random_configs determinism
# ---------------------------------------------------------------------------

def test_invariant_random_configs_deterministic_for_same_seed():
    """Same seed → identical trial list (line 73 of sweep.py).
    """
    params = {"lr": {"low": 1e-5, "high": 1e-2}, "depth": [4, 6, 8]}
    a = _random_configs(params, n_trials=5, seed=0)
    b = _random_configs(params, n_trials=5, seed=0)
    assert a == b


def test_invariant_random_configs_distinct_seeds_produce_distinct_samples():
    """Different seeds → distinct sample sequences (high probability)."""
    params = {"lr": {"low": 1e-5, "high": 1e-2}}
    a = _random_configs(params, n_trials=5, seed=0)
    b = _random_configs(params, n_trials=5, seed=42)
    assert a != b


def test_random_int_param_returns_int_values():
    """``type=int`` causes ``randint`` to be used (line 85-86 of sweep.py)."""
    params = {"depth": {"low": 1, "high": 100, "type": "int"}}
    out = _random_configs(params, n_trials=20, seed=0)
    for cfg in out:
        assert isinstance(cfg["depth"], int)
        assert 1 <= cfg["depth"] <= 100


def test_random_log_scale_param_stays_within_range():
    """``log=True`` log-uniform sampling stays in [low, high]."""
    params = {"lr": {"low": 1e-5, "high": 1e-1, "log": True}}
    out = _random_configs(params, n_trials=20, seed=0)
    for cfg in out:
        assert 1e-5 <= cfg["lr"] <= 1e-1


def test_random_list_param_chooses_from_provided_values():
    """List-valued params use ``random.choice`` (line 78-79)."""
    params = {"opt": ["adam", "lion", "sgd"]}
    out = _random_configs(params, n_trials=20, seed=0)
    for cfg in out:
        assert cfg["opt"] in {"adam", "lion", "sgd"}


def test_random_scalar_param_passed_through_unchanged():
    """Pin: a scalar (non-list, non-dict) param passes through as-is
    (line 93-94 of sweep.py).
    """
    params = {"warmup": 500}
    out = _random_configs(params, n_trials=3, seed=0)
    for cfg in out:
        assert cfg["warmup"] == 500


# ---------------------------------------------------------------------------
# _read_final_metric
# ---------------------------------------------------------------------------

def test_read_final_metric_returns_last_occurrence(tmp_path: Path):
    """When a metric appears in multiple lines, the LAST value is returned
    (line 130 of sweep.py: ``last_val = float(entry[metric_key])``).
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    log_file = logs / "metrics.jsonl"
    log_file.write_text(
        '{"step": 1, "loss": 1.5}\n'
        '{"step": 2, "loss": 1.2}\n'
        '{"step": 3, "loss": 0.9}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "loss")
    assert val == pytest.approx(0.9)


def test_read_final_metric_returns_none_when_jsonl_missing(tmp_path: Path):
    """No metrics.jsonl → None, not raise."""
    assert _read_final_metric(tmp_path, "loss") is None


def test_read_final_metric_tolerates_malformed_lines(tmp_path: Path):
    """Malformed JSON lines are silently skipped (line 131-132)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    log_file = logs / "metrics.jsonl"
    log_file.write_text(
        '{"step": 1, "loss": 1.5}\n'
        "this is not JSON\n"
        '{"step": 2, "loss": 0.7}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "loss")
    assert val == pytest.approx(0.7)


def test_read_final_metric_returns_none_when_key_absent(tmp_path: Path):
    """Metric file exists but the key is missing → None."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 1, "other": 1.5}\n', encoding="utf-8"
    )
    assert _read_final_metric(tmp_path, "loss") is None


def test_read_final_metric_falls_back_to_root_metrics_jsonl(tmp_path: Path):
    """When there is no ``logs/`` subdir, ``metrics.jsonl`` at the run-dir root
    is read (merged from test_lab_sweep.test_read_final_metric_fallback_root).
    """
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 5, "val_loss": 3.0}\n', encoding="utf-8"
    )
    assert _read_final_metric(tmp_path, "val_loss") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _compute_sensitivity
# ---------------------------------------------------------------------------

def _trial(trial_id: int, params: dict, metric: float | None) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config_overrides=dict(params),
        metric=metric,
        status="ok" if metric is not None else "failed",
        run_dir=None,
    )


def test_sensitivity_returns_empty_when_fewer_than_two_trials():
    """``_compute_sensitivity`` returns {} when fewer than 2 successful
    trials (line 151-152).
    """
    trials = [_trial(0, {"lr": 1e-3}, None)]
    out = _compute_sensitivity(trials, {"lr": [1e-3, 1e-4]})
    assert out == {}


def test_sensitivity_returns_zero_for_constant_metric():
    """When metric variance is below threshold (1e-14), every sensitivity
    value is 0 (line 156-157).
    """
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-4}, 1.0),
        _trial(2, {"lr": 1e-5}, 1.0),
    ]
    out = _compute_sensitivity(trials, {"lr": [1e-3, 1e-4, 1e-5]})
    assert out == {"lr": 0.0}


def test_sensitivity_constant_param_value_yields_zero():
    """A param that never varies has zero sensitivity (line 166-168)."""
    trials = [
        _trial(0, {"lr": 1e-3}, 1.0),
        _trial(1, {"lr": 1e-3}, 2.0),
        _trial(2, {"lr": 1e-3}, 3.0),
    ]
    out = _compute_sensitivity(trials, {"lr": [1e-3]})
    assert out["lr"] == 0.0


def test_sensitivity_perfect_correlation_yields_one():
    """Perfect linear correlation between param and metric yields ~1.0."""
    trials = [
        _trial(0, {"lr": 1.0}, 1.0),
        _trial(1, {"lr": 2.0}, 2.0),
        _trial(2, {"lr": 3.0}, 3.0),
        _trial(3, {"lr": 4.0}, 4.0),
    ]
    out = _compute_sensitivity(trials, {"lr": [1.0, 2.0, 3.0, 4.0]})
    assert out["lr"] == pytest.approx(1.0)


def test_sensitivity_categorical_param_uses_between_group_variance():
    """Categorical params (non-numeric) use between-group variance ratio
    (lines 175-185).

    Setup: opt='adam' → metric=1.0; opt='lion' → metric=10.0.
    Expected: high sensitivity (group means diverge a lot).
    """
    trials = [
        _trial(0, {"opt": "adam"}, 1.0),
        _trial(1, {"opt": "adam"}, 1.1),
        _trial(2, {"opt": "lion"}, 10.0),
        _trial(3, {"opt": "lion"}, 10.1),
    ]
    out = _compute_sensitivity(trials, {"opt": ["adam", "lion"]})
    assert out["opt"] > 0.5


# ---------------------------------------------------------------------------
# _find_run_dir
# ---------------------------------------------------------------------------

def test_find_run_dir_returns_none_when_root_missing(tmp_path: Path):
    """Non-existent trial_root → None (line 105-107)."""
    assert _find_run_dir(tmp_path / "nope") is None


def test_find_run_dir_returns_latest_subdir(tmp_path: Path):
    """Returns the last item via reversed-sorted iteration (line 108-111).

    With sorted subdirs ['a', 'b', 'c'], reversed picks 'c' first.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    out = _find_run_dir(tmp_path)
    assert out is not None
    assert out.name == "c"


def test_find_run_dir_ignores_files(tmp_path: Path):
    """Non-directory entries are skipped (line 110: ``if d.is_dir()``)."""
    (tmp_path / "a_file.txt").write_text("x")
    (tmp_path / "real_dir").mkdir()
    out = _find_run_dir(tmp_path)
    assert out is not None
    assert out.name == "real_dir"


# ---------------------------------------------------------------------------
# SweepRunner end-to-end (mocked subprocess)
# (merged from tests/test_lab_sweep.py — exercises the runner, not just helpers)
# ---------------------------------------------------------------------------

def _write_sweep_yaml(path: Path, params: dict, n_trials: int = 4) -> None:
    cfg = {
        "name": "test_sweep",
        "metric": "loss",
        "direction": "minimize",
        "n_trials": n_trials,
        "params": params,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _write_base_yaml(path: Path, run_root: str) -> None:
    cfg = {
        "mode": "lab",
        "exp": "base",
        "run_root": run_root,
        "model": {"name": "tiny_lm"},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _make_trial_run_dir(run_root: Path, trial_exp: str, metric_val: float) -> None:
    """Create a fake trial run directory with a metrics.jsonl."""
    from lighttrain.utils.run_dir import slugify

    trial_root = run_root / slugify(trial_exp)
    run_dir = trial_root / "20250101-000000-test-abc12345"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text(
        json.dumps({"step": 50, "loss": metric_val}) + "\n",
        encoding="utf-8",
    )


def test_sweep_runner_grid_generates_one_config_per_grid_point(tmp_path: Path):
    """Grid strategy: a 2-value param yields 2 trial configs."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    assert len(runner._generate_configs()) == 2


def test_sweep_runner_random_generates_n_trials_configs(tmp_path: Path):
    """Random strategy: number of generated configs equals n_trials."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": {"low": 1e-5, "high": 1e-2}}, n_trials=6)

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="random")
    assert len(runner._generate_configs()) == 6


def test_sweep_runner_run_collects_all_trial_results_and_picks_best(tmp_path: Path):
    """``run()`` returns a SweepReport with one ok trial per grid point and
    selects the minimizing config as best.
    """
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4, 1e-3]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    sweep_run_root = run_root / "sweep_test_sweep"
    metrics = [2.3, 1.9, 2.1]
    call_count = [0]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        _make_trial_run_dir(sweep_run_root, f"test_sweep_trial_{i:03d}", metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert isinstance(report, SweepReport)
    assert len(report.trials) == 3
    assert len([t for t in report.trials if t.status == "ok"]) == 3
    assert report.best_metric == pytest.approx(1.9)
    assert report.best_config == {"optim.lr": 3e-4}


def test_sweep_runner_marks_nonzero_exit_trials_failed(tmp_path: Path):
    """A subprocess returncode != 0 marks every trial failed and leaves
    best_metric None.
    """
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")

    def fake_run(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 1
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert all(t.status == "failed" for t in report.trials)
    assert report.best_metric is None


def test_sweep_report_includes_sensitivity_for_swept_param(tmp_path: Path):
    """A completed sweep reports a sensitivity entry for the swept parameter."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4, 1e-3]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    sweep_run_root = run_root / "sweep_test_sweep"
    metrics = [2.5, 1.5, 2.0]
    call_count = [0]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        _make_trial_run_dir(sweep_run_root, f"test_sweep_trial_{i:03d}", metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert "optim.lr" in report.sensitivity


# ---------------------------------------------------------------------------
# Optuna sweep backend wiring (merged from tests/test_lab_sweep.py)
# ---------------------------------------------------------------------------

def _optuna_installed() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def test_sweep_backend_is_a_known_registry_category():
    """The optuna plugin registers under ``sweep_backend``; that category must
    be declared in KNOWN_CATEGORIES so ``get(...)`` resolves it.
    """
    from lighttrain.registry._core import KNOWN_CATEGORIES

    assert "sweep_backend" in KNOWN_CATEGORIES


@pytest.mark.skipif(
    _optuna_installed(), reason="optuna installed; this exercises the missing-dep path"
)
def test_sweep_optuna_missing_dep_raises_friendly_error(tmp_path: Path):
    """With optuna absent, an optuna sweep raises a friendly RuntimeError
    pointing at ``pip install -e '.[sweep]'`` — not a bare registry exception.
    """
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="optuna")
    with pytest.raises(RuntimeError, match=r"\.\[sweep\]"):
        runner._generate_configs()
