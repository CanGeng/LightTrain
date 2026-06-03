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

from pathlib import Path

import pytest

from lighttrain.lab.sweep import (
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
