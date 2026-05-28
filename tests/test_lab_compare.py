"""Tests for lighttrain.lab.compare — DESIGN §26.10 (M8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lighttrain.lab.compare import (
    CompareReport,
    _diff_configs,
    _flatten,
    _read_last_metrics,
    compare,
    render_ascii,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    base: Path,
    name: str,
    config: dict,
    metrics: list[dict],
    fork_of: str | None = None,
) -> Path:
    run_dir = base / name
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    with open(run_dir / "logs" / "metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    if fork_of:
        (run_dir / "fork_meta.json").write_text(
            json.dumps({"fork_of_run_dir": fork_of}), encoding="utf-8"
        )
    return run_dir


# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------


def test_flatten_simple():
    d = {"a": 1, "b": {"c": 2, "d": 3}}
    flat = _flatten(d)
    assert flat == {"a": 1, "b.c": 2, "b.d": 3}


def test_flatten_empty():
    assert _flatten({}) == {}


# ---------------------------------------------------------------------------
# _diff_configs
# ---------------------------------------------------------------------------


def test_diff_shows_only_changed_fields():
    cfg_a = {"lr": 1e-4, "model": {"d": 128}}
    cfg_b = {"lr": 3e-4, "model": {"d": 128}}
    diff = _diff_configs([cfg_a, cfg_b])
    assert "lr" in diff
    assert "model.d" not in diff


def test_diff_identical_configs():
    cfg = {"lr": 1e-4, "wd": 0.1}
    diff = _diff_configs([cfg, cfg])
    assert diff == {}


def test_diff_three_runs():
    a = {"lr": 1e-4}
    b = {"lr": 3e-4}
    c = {"lr": 3e-4}
    diff = _diff_configs([a, b, c])
    assert "lr" in diff
    assert diff["lr"] == [1e-4, 3e-4, 3e-4]


# ---------------------------------------------------------------------------
# _read_last_metrics
# ---------------------------------------------------------------------------


def test_read_metrics_from_logs_subdir(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "metrics.jsonl").write_text(
        '{"step": 10, "loss": 2.5}\n{"step": 20, "loss": 1.8}\n',
        encoding="utf-8",
    )
    m = _read_last_metrics(tmp_path)
    assert m["loss"] == pytest.approx(1.8)
    assert "step" not in m


def test_read_metrics_empty(tmp_path: Path):
    assert _read_last_metrics(tmp_path) == {}


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_basic(tmp_path: Path):
    r1 = _make_run(tmp_path, "run1", {"lr": 1e-4}, [{"step": 10, "loss": 2.0}])
    r2 = _make_run(tmp_path, "run2", {"lr": 3e-4}, [{"step": 10, "loss": 1.5}])

    report = compare([r1, r2])

    assert isinstance(report, CompareReport)
    assert len(report.runs) == 2
    assert "lr" in report.config_diff
    assert report.metrics_table["loss"] == [pytest.approx(2.0), pytest.approx(1.5)]


def test_compare_shows_only_diff_fields(tmp_path: Path):
    r1 = _make_run(tmp_path, "run1", {"lr": 1e-4, "wd": 0.1}, [{"step": 1}])
    r2 = _make_run(tmp_path, "run2", {"lr": 3e-4, "wd": 0.1}, [{"step": 1}])

    report = compare([r1, r2])

    assert "lr" in report.config_diff
    assert "wd" not in report.config_diff


def test_compare_detects_fork_ancestry(tmp_path: Path):
    r1 = _make_run(tmp_path, "run1", {}, [{"step": 1}])
    r2 = _make_run(tmp_path, "run2", {}, [{"step": 1}], fork_of=str(r1))

    report = compare([r1, r2])

    assert report.fork_ancestry[str(r2)] == str(r1)
    assert report.fork_ancestry[str(r1)] is None


def test_compare_three_runs(tmp_path: Path):
    r1 = _make_run(tmp_path, "r1", {"lr": 1e-4}, [{"step": 1, "loss": 3.0}])
    r2 = _make_run(tmp_path, "r2", {"lr": 3e-4}, [{"step": 1, "loss": 2.0}])
    r3 = _make_run(tmp_path, "r3", {"lr": 1e-3}, [{"step": 1, "loss": 2.5}])

    report = compare([r1, r2, r3])

    assert len(report.metrics_table["loss"]) == 3
    assert len(report.config_diff["lr"]) == 3


# ---------------------------------------------------------------------------
# render_ascii
# ---------------------------------------------------------------------------


def test_render_ascii_shows_diff_header(tmp_path: Path):
    r1 = _make_run(tmp_path, "run1", {"lr": 1e-4}, [{"step": 1, "loss": 2.0}])
    r2 = _make_run(tmp_path, "run2", {"lr": 3e-4}, [{"step": 1, "loss": 1.5}])

    report = compare([r1, r2])
    out = render_ascii(report)

    assert "Config diff" in out
    assert "lr" in out
    assert "loss" in out


def test_render_ascii_no_diff(tmp_path: Path):
    cfg = {"lr": 1e-4, "wd": 0.1}
    r1 = _make_run(tmp_path, "run1", cfg, [{"step": 1, "loss": 2.0}])
    r2 = _make_run(tmp_path, "run2", cfg, [{"step": 1, "loss": 1.5}])

    report = compare([r1, r2])
    out = render_ascii(report)

    assert "no differences" in out


def test_render_ascii_shows_fork_ancestry(tmp_path: Path):
    r1 = _make_run(tmp_path, "run1", {}, [{"step": 1}])
    r2 = _make_run(tmp_path, "run2", {}, [{"step": 1}], fork_of=str(r1))

    report = compare([r1, r2])
    out = render_ascii(report)

    assert "fork of" in out.lower()
