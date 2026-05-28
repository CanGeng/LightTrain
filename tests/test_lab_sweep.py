"""Tests for lighttrain.lab.sweep — DESIGN §26.10 (M8)."""

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
    _grid_configs,
    _random_configs,
    _read_final_metric,
)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def test_grid_configs_cartesian_product():
    params = {"a": [1, 2], "b": [10, 20]}
    configs = _grid_configs(params)
    assert len(configs) == 4
    assert {"a": 1, "b": 10} in configs
    assert {"a": 2, "b": 20} in configs


def test_grid_configs_single_param():
    params = {"lr": [1e-4, 3e-4, 1e-3]}
    configs = _grid_configs(params)
    assert len(configs) == 3
    assert all(len(c) == 1 for c in configs)


def test_grid_configs_empty_params():
    configs = _grid_configs({})
    assert configs == [{}]


def test_random_configs_count():
    params = {"lr": {"low": 1e-5, "high": 1e-2}, "wd": {"low": 0.0, "high": 0.1}}
    configs = _random_configs(params, n_trials=8, seed=42)
    assert len(configs) == 8
    for c in configs:
        assert 1e-5 <= c["lr"] <= 1e-2
        assert 0.0 <= c["wd"] <= 0.1


def test_random_configs_reproducible():
    params = {"lr": [1e-4, 3e-4, 1e-3]}
    c1 = _random_configs(params, 5, seed=99)
    c2 = _random_configs(params, 5, seed=99)
    assert c1 == c2


def test_random_configs_int_type():
    params = {"layers": {"low": 2, "high": 8, "type": "int"}}
    configs = _random_configs(params, 20, seed=7)
    for c in configs:
        assert isinstance(c["layers"], int)
        assert 2 <= c["layers"] <= 8


# ---------------------------------------------------------------------------
# Metric reading
# ---------------------------------------------------------------------------


def test_read_final_metric_from_logs(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 10, "loss": 2.5}\n{"step": 20, "loss": 2.1}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "loss")
    assert val == pytest.approx(2.1)


def test_read_final_metric_fallback_root(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 5, "val_loss": 3.0}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "val_loss")
    assert val == pytest.approx(3.0)


def test_read_final_metric_missing_key(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "other": 0.5}\n',
        encoding="utf-8",
    )
    val = _read_final_metric(tmp_path, "loss")
    assert val is None


def test_read_final_metric_no_file(tmp_path: Path):
    val = _read_final_metric(tmp_path, "loss")
    assert val is None


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def test_compute_sensitivity_perfect_correlation():
    trials = [
        TrialResult(0, {"lr": 0.01}, 2.0, "ok", None),
        TrialResult(1, {"lr": 0.1}, 1.0, "ok", None),
        TrialResult(2, {"lr": 1.0}, 0.5, "ok", None),
    ]
    sens = _compute_sensitivity(trials, {"lr": [0.01, 0.1, 1.0]})
    assert "lr" in sens
    assert sens["lr"] > 0.5


def test_compute_sensitivity_no_variance():
    trials = [
        TrialResult(0, {"lr": 1e-4}, 2.0, "ok", None),
        TrialResult(1, {"lr": 1e-4}, 2.0, "ok", None),
    ]
    sens = _compute_sensitivity(trials, {"lr": [1e-4]})
    assert sens.get("lr", 0.0) == 0.0


def test_compute_sensitivity_too_few_trials():
    trials = [TrialResult(0, {"lr": 1e-4}, 2.0, "ok", None)]
    sens = _compute_sensitivity(trials, {"lr": [1e-4]})
    assert sens == {}


# ---------------------------------------------------------------------------
# SweepRunner (mocked subprocess)
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
    """Create a fake trial run directory with metrics.jsonl."""
    from lighttrain.utils.run_dir import slugify

    trial_root = run_root / slugify(trial_exp)
    run_dir = trial_root / "20250101-000000-test-abc12345"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text(
        json.dumps({"step": 50, "loss": metric_val}) + "\n",
        encoding="utf-8",
    )


def test_sweep_runner_grid_structure(tmp_path: Path):
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    assert len(runner._generate_configs()) == 2


def test_sweep_runner_random_structure(tmp_path: Path):
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": {"low": 1e-5, "high": 1e-2}}, n_trials=6)

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="random")
    assert len(runner._generate_configs()) == 6


def test_sweep_runner_run_collects_results(tmp_path: Path):
    """SweepRunner.run() should return a SweepReport with all trial results."""
    base_cfg = tmp_path / "base.yaml"
    sweep_cfg = tmp_path / "sweep.yaml"
    run_root = tmp_path / "runs"
    run_root.mkdir()
    _write_base_yaml(base_cfg, str(run_root))
    _write_sweep_yaml(sweep_cfg, {"optim.lr": [1e-4, 3e-4, 1e-3]})

    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    sweep_run_root = run_root / "sweep_test_sweep"

    # Stub subprocess.run to return success and create fake run dirs
    metrics = [2.3, 1.9, 2.1]

    call_count = [0]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        trial_exp = f"test_sweep_trial_{i:03d}"
        _make_trial_run_dir(sweep_run_root, trial_exp, metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert isinstance(report, SweepReport)
    assert len(report.trials) == 3
    ok = [t for t in report.trials if t.status == "ok"]
    assert len(ok) == 3
    assert report.best_metric == pytest.approx(1.9)
    assert report.best_config == {"optim.lr": 3e-4}


def test_sweep_runner_handles_failed_trial(tmp_path: Path):
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


def test_sweep_report_has_sensitivity(tmp_path: Path):
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
        trial_exp = f"test_sweep_trial_{i:03d}"
        _make_trial_run_dir(sweep_run_root, trial_exp, metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert "optim.lr" in report.sensitivity
