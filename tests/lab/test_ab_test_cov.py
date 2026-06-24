"""Coverage-supplement tests for ``lighttrain.lab.ab_test``.

Pins the behaviour of ``ab_test()`` itself (lines 53-102), which the
existing test_ab_test.py does not reach because it only exercises the
ABReport dataclass.

What we cover / pin:

* Line 53        – run_root_path is resolved via Path(...).resolve()
* Lines 55-57    – ``_launch`` closure: exp / cmd construction from cfg stem + variant
* Lines 64-70    – subprocess.run called with correct args (mocked)
* Lines 71-72    – TimeoutExpired → warning logged, execution continues
* Lines 78-79    – _find_run_dir / _read_final_metric called; rd=None → metric=None
* Line 80        – (rd, metric) tuple returned from _launch
* Lines 82-83    – both variants launched in sequence
* Lines 85-87    – delta = metric_b - metric_a; None when either metric missing
* Lines 89-93    – compare() called when both run dirs are not None
* Lines 93-100   – compare() exception swallowed → compare_report stays None
* Line 102       – ABReport constructed and returned
"""

from __future__ import annotations

import importlib
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lighttrain.lab.ab_test import ABReport, ab_test
from lighttrain.lab.compare import CompareReport

# Use importlib to get the actual module object (not the function that the
# package __init__.py re-exports under the same name "lighttrain.lab.ab_test").
_module = importlib.import_module("lighttrain.lab.ab_test")

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeCompareReport:
    """Minimal stand-in so compare() returns something truthy."""
    runs: list = []
    config_diff: dict = {}
    metrics_table: dict = {}
    fork_ancestry: dict = {}


def _make_compare_report(run_a: Path, run_b: Path) -> CompareReport:
    return CompareReport(
        runs=[run_a, run_b],
        config_diff={},
        metrics_table={"loss": [0.5, 0.3]},
        fork_ancestry={str(run_a): None, str(run_b): None},
    )


# ---------------------------------------------------------------------------
# Happy path: both variants succeed, metrics found, compare succeeds
# ---------------------------------------------------------------------------

def test_invariant_ab_test_happy_path_returns_correct_delta(tmp_path, monkeypatch):
    """When both subprocesses succeed and metrics are found, delta = B - A.

    All external calls are mocked so no real training occurs.
    """
    cfg_a = tmp_path / "recipe_a.yaml"
    cfg_b = tmp_path / "recipe_b.yaml"
    cfg_a.write_text("seed: 42\n", encoding="utf-8")
    cfg_b.write_text("seed: 42\n", encoding="utf-8")

    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    # Patch subprocess.run to be a no-op
    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))

    # Patch _find_run_dir to return the fake dirs
    dirs = iter([run_a_dir, run_b_dir])
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: next(dirs))

    # Patch _read_final_metric to return deterministic values
    metrics = iter([0.5, 0.3])
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: next(metrics))

    # Patch compare to return a canned report
    fake_compare = _make_compare_report(run_a_dir, run_b_dir)
    monkeypatch.setattr(_module, "compare", lambda _runs: fake_compare)

    report = ab_test(cfg_a, cfg_b, seed=7, metric_key="loss", run_root=str(tmp_path / "runs"))

    assert isinstance(report, ABReport)
    assert report.metric_a == pytest.approx(0.5)
    assert report.metric_b == pytest.approx(0.3)
    assert report.delta == pytest.approx(0.3 - 0.5)
    assert report.compare is fake_compare


def test_invariant_subprocess_cmd_contains_seed_and_exp(tmp_path, monkeypatch):
    """The subprocess cmd must include ``++seed=<seed>`` and ``++exp=ab_a_<stem>``.

    Pins lines 56-63 of ab_test.py.
    """
    cfg_a = tmp_path / "my_config.yaml"
    cfg_b = tmp_path / "my_config.yaml"  # same file is fine for this test
    cfg_a.write_text("seed: 0\n", encoding="utf-8")

    run_dir = tmp_path / "r"
    run_dir.mkdir()

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr(_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: run_dir)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: 0.1)
    monkeypatch.setattr(_module, "compare", lambda _: _make_compare_report(run_dir, run_dir))

    ab_test(cfg_a, cfg_b, seed=99, metric_key="loss", run_root=str(tmp_path / "runs"))

    assert len(captured_cmds) == 2, "Expected exactly 2 subprocess calls (one per variant)"

    # Variant A command
    cmd_a = captured_cmds[0]
    assert "++seed=99" in cmd_a
    assert any("ab_a_my_config" in arg for arg in cmd_a), (
        f"Expected ab_a_my_config in cmd_a, got {cmd_a}"
    )

    # Variant B command
    cmd_b = captured_cmds[1]
    assert "++seed=99" in cmd_b
    assert any("ab_b_my_config" in arg for arg in cmd_b), (
        f"Expected ab_b_my_config in cmd_b, got {cmd_b}"
    )


def test_invariant_run_root_injected_into_subprocess_cmd(tmp_path, monkeypatch):
    """The resolved ``run_root`` is passed as ``++run_root=...`` (line 62)."""
    cfg_a = tmp_path / "cfg.yaml"
    cfg_a.write_text("", encoding="utf-8")

    run_dir = tmp_path / "r"
    run_dir.mkdir()

    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr(_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: run_dir)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: 0.0)
    monkeypatch.setattr(_module, "compare", lambda _: _make_compare_report(run_dir, run_dir))

    run_root_arg = str(tmp_path / "myroot")
    ab_test(cfg_a, cfg_a, run_root=run_root_arg)

    resolved = str(Path(run_root_arg).resolve())
    for cmd in captured:
        assert any(arg.startswith(f"++run_root={resolved}") for arg in cmd), (
            f"++run_root not found in {cmd}"
        )


# ---------------------------------------------------------------------------
# Timeout branch (lines 71-77)
# ---------------------------------------------------------------------------

def test_invariant_timeout_logs_warning_and_continues(tmp_path, monkeypatch, caplog):
    """A TimeoutExpired from subprocess.run is caught; a warning is emitted;
    execution continues and returns an ABReport (lines 71-77).
    """
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    def _timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=0.01)

    monkeypatch.setattr(_module.subprocess, "run", _timeout_run)
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: None)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: None)

    with caplog.at_level(logging.WARNING, logger="lighttrain.lab.ab_test"):
        report = ab_test(cfg_a, cfg_b, trial_timeout_s=0.01)

    assert isinstance(report, ABReport)
    # Warning must mention the variant
    assert any(
        "crashed or timed out" in r.message or "timed out" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ), f"Expected timeout warning; got records: {[r.message for r in caplog.records]}"


def test_invariant_generic_exception_in_subprocess_logs_warning(tmp_path, monkeypatch, caplog):
    """A generic exception (not TimeoutExpired) from subprocess.run is also caught
    by the BLE001 broad except (line 71) and logs a warning.
    """
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    def _crash_run(cmd, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_module.subprocess, "run", _crash_run)
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: None)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: None)

    with caplog.at_level(logging.WARNING, logger="lighttrain.lab.ab_test"):
        report = ab_test(cfg_a, cfg_b)

    assert isinstance(report, ABReport)
    assert any(
        "crashed or timed out" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


# ---------------------------------------------------------------------------
# _find_run_dir returns None (lines 78-80)
# ---------------------------------------------------------------------------

def test_invariant_no_run_dir_means_no_metric(tmp_path, monkeypatch):
    """When _find_run_dir returns None for a variant, its metric is None
    and _read_final_metric is NOT called for that variant (line 79).
    """
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: None)

    metric_calls: list = []

    def _track_metric(*a, **kw):
        metric_calls.append(a)
        return 0.5

    monkeypatch.setattr(_module, "_read_final_metric", _track_metric)

    report = ab_test(cfg_a, cfg_b)

    assert report.metric_a is None
    assert report.metric_b is None
    assert metric_calls == [], "Expected _read_final_metric not called when rd is None"


# ---------------------------------------------------------------------------
# Delta calculation branches (lines 85-87)
# ---------------------------------------------------------------------------

def test_invariant_delta_is_b_minus_a_numerically(tmp_path, monkeypatch):
    """delta = metric_b - metric_a when both metrics are not None (line 87)."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_dir = tmp_path / "r"
    run_dir.mkdir()

    dirs = iter([run_dir, run_dir])
    metrics = iter([1.0, 0.6])

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: next(dirs))
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: next(metrics))
    monkeypatch.setattr(_module, "compare", lambda _: _make_compare_report(run_dir, run_dir))

    report = ab_test(cfg_a, cfg_b)

    assert report.delta == pytest.approx(0.6 - 1.0)


def test_invariant_delta_is_none_when_metric_a_is_none(tmp_path, monkeypatch):
    """delta stays None when metric_a is None (line 86 condition false)."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_b_dir = tmp_path / "run_b"
    run_b_dir.mkdir()

    call_count = 0

    def _find(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return None if call_count == 1 else run_b_dir

    metric_call_count = 0

    def _metric(*_a, **_kw):
        nonlocal metric_call_count
        metric_call_count += 1
        return 0.4

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", _find)
    monkeypatch.setattr(_module, "_read_final_metric", _metric)
    monkeypatch.setattr(_module, "compare", lambda _: None)

    report = ab_test(cfg_a, cfg_b)

    assert report.metric_a is None
    assert report.metric_b == pytest.approx(0.4)
    assert report.delta is None


def test_invariant_delta_is_none_when_metric_b_is_none(tmp_path, monkeypatch):
    """delta stays None when metric_b is None."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_a_dir = tmp_path / "run_a"
    run_a_dir.mkdir()

    call_count = 0

    def _find(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return run_a_dir if call_count == 1 else None

    metric_call_count = 0

    def _metric(*_a, **_kw):
        nonlocal metric_call_count
        metric_call_count += 1
        return 0.7

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", _find)
    monkeypatch.setattr(_module, "_read_final_metric", _metric)

    report = ab_test(cfg_a, cfg_b)

    assert report.metric_a == pytest.approx(0.7)
    assert report.metric_b is None
    assert report.delta is None


# ---------------------------------------------------------------------------
# compare() path (lines 89-100)
# ---------------------------------------------------------------------------

def test_invariant_compare_not_called_when_run_a_is_none(tmp_path, monkeypatch):
    """compare() is not called when run_a_dir is None (line 90 condition)."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_b_dir = tmp_path / "run_b"
    run_b_dir.mkdir()

    call_count = 0

    def _find(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return None if call_count == 1 else run_b_dir

    compare_called = []
    def _compare(runs):
        compare_called.append(runs)
        return _make_compare_report(*runs)

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", _find)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: 0.1)
    monkeypatch.setattr(_module, "compare", _compare)

    report = ab_test(cfg_a, cfg_b)

    assert compare_called == [], "compare() must NOT be called when run_a is None"
    assert report.compare is None


def test_invariant_compare_not_called_when_run_b_is_none(tmp_path, monkeypatch):
    """compare() is not called when run_b_dir is None (line 90 condition)."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_a_dir = tmp_path / "run_a"
    run_a_dir.mkdir()

    call_count = 0

    def _find(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return run_a_dir if call_count == 1 else None

    compare_called = []

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", _find)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: None)
    monkeypatch.setattr(_module, "compare", lambda runs: compare_called.append(runs))

    report = ab_test(cfg_a, cfg_b)

    assert compare_called == []
    assert report.compare is None


def test_invariant_compare_exception_swallowed_returns_none(tmp_path, monkeypatch, caplog):
    """When compare() raises, the exception is swallowed, a warning logged,
    and compare_report stays None (lines 93-100).
    """
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    dirs = iter([run_a_dir, run_b_dir])
    metrics = iter([0.2, 0.4])

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: next(dirs))
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: next(metrics))

    def _bad_compare(_runs):
        raise RuntimeError("compare boom")

    monkeypatch.setattr(_module, "compare", _bad_compare)

    with caplog.at_level(logging.WARNING, logger="lighttrain.lab.ab_test"):
        report = ab_test(cfg_a, cfg_b)

    assert isinstance(report, ABReport)
    assert report.compare is None
    assert any(
        "compare" in r.message.lower() and "failed" in r.message.lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ), f"Expected compare-failed warning; got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# ABReport fields from ab_test() (line 102)
# ---------------------------------------------------------------------------

def test_invariant_returned_abreport_has_correct_run_dirs(tmp_path, monkeypatch):
    """ab_test() returns an ABReport where run_a / run_b are the dirs from
    _find_run_dir (line 102).
    """
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    dirs = iter([run_a_dir, run_b_dir])
    metrics = iter([0.5, 0.3])

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: next(dirs))
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: next(metrics))
    monkeypatch.setattr(_module, "compare", lambda _: _make_compare_report(run_a_dir, run_b_dir))

    report = ab_test(cfg_a, cfg_b)

    assert report.run_a == run_a_dir
    assert report.run_b == run_b_dir


def test_invariant_abreport_none_compare_when_both_dirs_none(tmp_path, monkeypatch):
    """When both run dirs are None, compare is never called and report.compare is None."""
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    cfg_a.write_text("", encoding="utf-8")
    cfg_b.write_text("", encoding="utf-8")

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: None)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: None)

    compare_called = []
    monkeypatch.setattr(_module, "compare", lambda _: compare_called.append(True))

    report = ab_test(cfg_a, cfg_b)

    assert report.run_a is None
    assert report.run_b is None
    assert report.compare is None
    assert compare_called == []


# ---------------------------------------------------------------------------
# Default arguments (seed, metric_key, run_root)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_pin_seed_injected_as_override(tmp_path, monkeypatch, seed):
    """The seed argument is injected as ``++seed=<seed>`` in the command."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")

    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr(_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: None)
    monkeypatch.setattr(_module, "_read_final_metric", lambda *_a, **_kw: None)

    ab_test(cfg, cfg, seed=seed)

    for cmd in captured:
        assert f"++seed={seed}" in cmd


def test_pin_metric_key_forwarded_to_read_final_metric(tmp_path, monkeypatch):
    """The metric_key argument is forwarded to _read_final_metric (line 79)."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")

    run_dir = tmp_path / "r"
    run_dir.mkdir()

    received_keys: list[str] = []

    def _metric(rd, key):
        received_keys.append(key)
        return 0.1

    monkeypatch.setattr(_module.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr(_module, "_find_run_dir", lambda *_a, **_kw: run_dir)
    monkeypatch.setattr(_module, "_read_final_metric", _metric)
    monkeypatch.setattr(_module, "compare", lambda _: _make_compare_report(run_dir, run_dir))

    ab_test(cfg, cfg, metric_key="my_metric")

    assert all(k == "my_metric" for k in received_keys), (
        f"Expected all metric keys to be 'my_metric', got {received_keys}"
    )
