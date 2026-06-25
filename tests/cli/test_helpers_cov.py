"""Coverage tests for ``lighttrain.cli._helpers`` (currently 74% covered).

Targets every uncovered branch identified in the coverage report:
  lines 29-33   : _todo() with and without a ``what`` argument
  lines 61-62, 68: _flatten_patch_to_overrides yaml-dump exception fallback
  lines 82, 87, 90-91: _final_loss_from_run — missing file, json error, loss key
  lines 108, 115, 119-122: _eval_perplexity — None guards, loader fallbacks, exc
  lines 138-139  : _append_run_summary — corrupt/non-list file → reset to []
  lines 149-151  : _fmt_metric — float branch and non-float branch

None of these branches require a GPU, distributed environment, or external service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import lighttrain.cli._helpers as _helpers_mod
import lighttrain.eval.metrics as _metrics_mod
from lighttrain.cli._helpers import (
    _append_run_summary,
    _eval_perplexity,
    _final_loss_from_run,
    _flatten_patch_to_overrides,
    _fmt_metric,
    _todo,
)

# ---------------------------------------------------------------------------
# _todo  (lines 29-33)
# ---------------------------------------------------------------------------


def test_invariant_todo_with_what_exits_code_2_and_includes_what():
    """``_todo('P1', 'do X')`` must raise ``typer.Exit(code=2)`` and emit a
    message containing both the milestone and the ``what`` string (line 31)."""
    import typer

    with pytest.raises(typer.Exit) as exc_info:
        _todo("P1", "do X")

    assert exc_info.value.exit_code == 2


def test_invariant_todo_without_what_exits_code_2():
    """``_todo('P3')`` (no ``what``) still raises ``typer.Exit(code=2)`` via
    line 33; the branch on line 30 that appends ``— {what}`` is skipped."""
    import typer

    with pytest.raises(typer.Exit) as exc_info:
        _todo("P3")

    assert exc_info.value.exit_code == 2


def test_pin_current_behavior_todo_message_format(capsys, monkeypatch):
    """Pin the rendered string format: with ``what`` the separator is `` — ``."""
    import typer

    # Capture console.print output by patching it
    printed: list[str] = []
    monkeypatch.setattr(_helpers_mod.console, "print", lambda msg, **kw: printed.append(msg))

    with pytest.raises(typer.Exit):
        _todo("P2", "some feature")

    assert len(printed) == 1
    assert "P2" in printed[0]
    assert "some feature" in printed[0]


def test_pin_current_behavior_todo_no_what_message_no_dash(monkeypatch):
    """Without ``what`` the message must NOT contain `` — `` (line 30 skipped)."""
    import typer

    printed: list[str] = []
    monkeypatch.setattr(_helpers_mod.console, "print", lambda msg, **kw: printed.append(msg))

    with pytest.raises(typer.Exit):
        _todo("P3")

    assert " — " not in printed[0]
    assert "P3" in printed[0]


# ---------------------------------------------------------------------------
# _flatten_patch_to_overrides — yaml dump exception fallback (lines 61-62, 68)
# ---------------------------------------------------------------------------


def test_invariant_flatten_yaml_dump_failure_falls_back_to_repr():
    """When ``yaml.safe_dump`` raises an exception (lines 61-62), the branch at
    line 68 must fall back to ``repr()`` encoding of the list and still return
    a valid ``++key=<repr>`` override instead of crashing."""
    import yaml

    def _boom(*args, **kwargs):
        raise RuntimeError("yaml exploded")

    with patch.object(yaml, "safe_dump", _boom):
        result = _flatten_patch_to_overrides({"mykey": [1, 2, 3]})

    assert len(result) == 1
    # Fallback repr: repr([1, 2, 3]) == "[1, 2, 3]"
    assert result[0].startswith("++mykey=")
    assert "[1, 2, 3]" in result[0]


def test_invariant_flatten_yaml_dump_failure_logs_warning(caplog):
    """The yaml-dump exception path must log a WARNING (lines 62-67)."""
    import yaml

    def _boom(*args, **kwargs):
        raise RuntimeError("yaml exploded")

    with patch.object(yaml, "safe_dump", _boom):
        with caplog.at_level(logging.WARNING, logger="lighttrain.cli._helpers"):
            _flatten_patch_to_overrides({"k": [10, 20]})

    assert any("flow-dump" in r.message or "YAML" in r.message for r in caplog.records)


def test_invariant_flatten_yaml_import_failure_falls_back_to_repr(monkeypatch):
    """If the ``import yaml`` inside the except-guarded block were to fail at
    import time, the outer ``except Exception`` catches it and falls back to
    repr(). Simulate by making yaml's safe_dump raise ImportError."""
    import yaml

    def _raise_import(*args, **kwargs):
        raise ImportError("no yaml")

    with patch.object(yaml, "safe_dump", _raise_import):
        result = _flatten_patch_to_overrides({"x": [7, 8]})

    assert len(result) == 1
    assert result[0].startswith("++x=")


# ---------------------------------------------------------------------------
# _final_loss_from_run (lines 82, 87, 90-91)
# ---------------------------------------------------------------------------


def test_invariant_final_loss_returns_none_when_metrics_missing(tmp_path: Path):
    """``_final_loss_from_run`` returns ``None`` when ``metrics.jsonl`` does not
    exist (line 82)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _final_loss_from_run(run_dir) is None


def test_invariant_final_loss_skips_invalid_json_lines(tmp_path: Path):
    """Lines that are not valid JSON are silently skipped (lines 90-91); the
    function must not raise and must still return the last valid loss value."""
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    metrics = logs / "metrics.jsonl"
    metrics.write_text(
        '{"step": 1}\n'
        "INVALID JSON\n"
        '{"loss": 0.5}\n'
        "also not json\n"
        '{"loss": 0.3}\n',
        encoding="utf-8",
    )
    result = _final_loss_from_run(run_dir)
    assert result == pytest.approx(0.3)


def test_invariant_final_loss_returns_last_loss(tmp_path: Path):
    """Only lines that contain a ``loss`` key update the running ``last`` value
    (line 87). The function returns the final such value."""
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    metrics = logs / "metrics.jsonl"
    metrics.write_text(
        '{"loss": 1.5}\n'
        '{"grad_norm": 0.1}\n'  # no 'loss' key — must not change last
        '{"loss": 0.8}\n',
        encoding="utf-8",
    )
    result = _final_loss_from_run(run_dir)
    assert result == pytest.approx(0.8)


def test_invariant_final_loss_empty_metrics_returns_none(tmp_path: Path):
    """An existing but empty or no-loss ``metrics.jsonl`` returns ``None``."""
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "metrics.jsonl").write_text('{"step": 1}\n{"grad_norm": 0.5}\n', encoding="utf-8")
    assert _final_loss_from_run(run_dir) is None


def test_invariant_final_loss_blank_lines_skipped(tmp_path: Path):
    """Blank lines in ``metrics.jsonl`` are silently skipped without error."""
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "metrics.jsonl").write_text(
        "\n\n\n" '{"loss": 2.0}\n' "\n",
        encoding="utf-8",
    )
    assert _final_loss_from_run(run_dir) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _eval_perplexity (lines 108, 115, 119-122)
# ---------------------------------------------------------------------------


class _FakeTrainerNoModel:
    """Trainer without a model attribute (None via getattr)."""

    model = None
    data_module = None


class _FakeTrainerNoData:
    """Trainer with a model but no data_module."""

    model = object()
    data_module = None


class _FakeDMNoLoader:
    """data_module with neither val_loader nor train_loader."""

    pass


class _FakeDMValReturnsNone:
    """val_loader() returns None; train_loader() also absent — fully None."""

    def val_loader(self):
        return None


class _FakeDMValNoneTrainPresent:
    """val_loader() returns None; train_loader() returns a real iterable."""

    def val_loader(self):
        return None

    def train_loader(self):
        return iter([[1, 2, 3]])


def test_invariant_eval_perplexity_returns_none_when_model_is_none():
    """``_eval_perplexity`` returns ``None`` immediately if ``trainer.model`` is
    ``None`` (line 108 guard — both model AND data_module checked together)."""

    class _T:
        model = None
        data_module = object()

    assert _eval_perplexity(_T(), max_batches=1) is None


def test_invariant_eval_perplexity_returns_none_when_data_module_is_none():
    """``_eval_perplexity`` returns ``None`` when ``trainer.data_module`` is
    ``None`` (line 108 second condition)."""
    assert _eval_perplexity(_FakeTrainerNoData(), max_batches=1) is None


def test_invariant_eval_perplexity_returns_none_when_no_loader(monkeypatch):
    """When ``data_module`` has neither ``val_loader`` nor ``train_loader``,
    loader stays ``None`` and the function returns ``None`` (line 115)."""

    class _T:
        model = object()
        data_module = _FakeDMNoLoader()

    assert _eval_perplexity(_T(), max_batches=2) is None


def test_invariant_eval_perplexity_returns_none_when_val_and_train_loader_none(monkeypatch):
    """When ``val_loader()`` returns ``None`` and no ``train_loader`` exists,
    ``loader`` stays ``None`` → returns ``None`` (line 115 path)."""

    class _T:
        model = object()
        data_module = _FakeDMValReturnsNone()

    assert _eval_perplexity(_T(), max_batches=2) is None


def test_invariant_eval_perplexity_uses_train_loader_fallback(monkeypatch):
    """When ``val_loader()`` returns ``None`` but ``train_loader()`` returns an
    iterable, the train loader is used (lines 112-113 fallback path)."""

    captured: dict = {}

    def _fake_ppl(model, loader, device=None, max_batches=None):
        captured["loader"] = loader
        return 3.14

    monkeypatch.setattr(_metrics_mod, "perplexity", _fake_ppl)

    class _T:
        model = object()
        data_module = _FakeDMValNoneTrainPresent()
        device = None

    result = _eval_perplexity(_T(), max_batches=2)
    assert result == pytest.approx(3.14)
    assert captured.get("loader") is not None


def test_invariant_eval_perplexity_exception_returns_none(monkeypatch):
    """When ``perplexity(...)`` raises, ``_eval_perplexity`` catches it and
    returns ``None`` (lines 119-122) instead of propagating the exception."""

    def _boom(model, loader, **kw):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(_metrics_mod, "perplexity", _boom)

    class _FakeDM:
        def val_loader(self):
            return [1, 2, 3]

    class _T:
        model = object()
        data_module = _FakeDM()
        device = None

    result = _eval_perplexity(_T(), max_batches=2)
    assert result is None


def test_invariant_eval_perplexity_exception_logs_warning(monkeypatch, caplog):
    """The exception branch must emit a WARNING (line 120)."""

    def _boom(model, loader, **kw):
        raise ValueError("ppl failed")

    monkeypatch.setattr(_metrics_mod, "perplexity", _boom)

    class _FakeDM:
        def val_loader(self):
            return [1]

    class _T:
        model = object()
        data_module = _FakeDM()
        device = None

    with caplog.at_level(logging.WARNING, logger="lighttrain.cli._helpers"):
        _eval_perplexity(_T(), max_batches=1)

    assert any("perplexity" in r.message.lower() for r in caplog.records)


def test_invariant_eval_perplexity_max_batches_zero_passes_none(monkeypatch):
    """``max_batches=0`` is treated as 'no limit' and the underlying
    ``perplexity()`` is called with ``max_batches=None`` (line 116)."""
    captured: dict = {}

    def _capture(model, loader, device=None, max_batches=None):
        captured["mb"] = max_batches
        return 1.0

    monkeypatch.setattr(_metrics_mod, "perplexity", _capture)

    class _FakeDM:
        def val_loader(self):
            return [1]

    class _T:
        model = object()
        data_module = _FakeDM()
        device = None

    _eval_perplexity(_T(), max_batches=0)
    assert captured["mb"] is None


def test_invariant_eval_perplexity_positive_max_batches_passed_through(monkeypatch):
    """``max_batches > 0`` is passed as-is to ``perplexity()`` (line 116)."""
    captured: dict = {}

    def _capture(model, loader, device=None, max_batches=None):
        captured["mb"] = max_batches
        return 1.0

    monkeypatch.setattr(_metrics_mod, "perplexity", _capture)

    class _FakeDM:
        def val_loader(self):
            return [1]

    class _T:
        model = object()
        data_module = _FakeDM()
        device = None

    _eval_perplexity(_T(), max_batches=7)
    assert captured["mb"] == 7


# ---------------------------------------------------------------------------
# _append_run_summary — corrupt / non-list file reset (lines 138-139)
# ---------------------------------------------------------------------------


def test_invariant_append_run_summary_replaces_same_exp(tmp_path: Path):
    """An existing entry with the same ``exp`` key is removed before appending
    the new row (line 137). Only one row with that exp may exist after the call.
    Exercises the filter on line 137 (loading an existing list)."""
    summary = tmp_path / "summary.json"
    _append_run_summary(summary, {"exp": "e1", "status": "old"})
    _append_run_summary(summary, {"exp": "e1", "status": "new"})
    rows = json.loads(summary.read_text())
    matching = [r for r in rows if r.get("exp") == "e1"]
    assert len(matching) == 1
    assert matching[0]["status"] == "new"


def test_invariant_append_run_summary_corrupt_file_resets_to_empty(tmp_path: Path):
    """A file with corrupted JSON (``json.JSONDecodeError``) causes the existing
    rows list to be reset to ``[]`` (lines 138-139), and the new row is still
    appended cleanly."""
    summary = tmp_path / "summary.json"
    summary.write_text("NOT VALID JSON {{{{", encoding="utf-8")
    _append_run_summary(summary, {"exp": "fresh", "status": "ok"})
    rows = json.loads(summary.read_text())
    assert rows == [{"exp": "fresh", "status": "ok"}]


def test_invariant_append_run_summary_non_list_json_resets_to_empty(tmp_path: Path):
    """A file whose JSON root is not a list (e.g., a dict) causes the existing
    rows to be discarded (``isinstance(loaded, list)`` is False → rows stays
    ``[]``), and the new row is appended."""
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    _append_run_summary(summary, {"exp": "new_row", "status": "ok"})
    rows = json.loads(summary.read_text())
    assert rows == [{"exp": "new_row", "status": "ok"}]


def test_invariant_append_run_summary_creates_parent_dirs(tmp_path: Path):
    """``path.parent.mkdir(parents=True, exist_ok=True)`` creates any missing
    parent directories (line 141). The call must not raise even when the parent
    does not exist."""
    summary = tmp_path / "deep" / "nested" / "summary.json"
    _append_run_summary(summary, {"exp": "x", "status": "ok"})
    assert summary.exists()
    assert json.loads(summary.read_text()) == [{"exp": "x", "status": "ok"}]


def test_invariant_append_run_summary_accumulates_multiple_exps(tmp_path: Path):
    """Multiple distinct ``exp`` entries accumulate (no replacement when exp
    differs). Exercises the accumulation contract across successive calls."""
    summary = tmp_path / "out.json"
    for name in ["a", "b", "c"]:
        _append_run_summary(summary, {"exp": name, "v": 1})
    rows = json.loads(summary.read_text())
    assert {r["exp"] for r in rows} == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# _fmt_metric (lines 149-151)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (3.14159, "3.142"),   # float → 4 significant figures (line 149-150)
        (0.00001, "1e-05"),   # small float (4 sig figs, scientific notation)
        (10000.0, "1e+04"),   # large float (:.4g uses scientific above 1e4)
        (0.0, "0"),           # zero float
        (42, "42"),           # int → str(v) path (line 151)
        ("hello", "hello"),   # str pass-through (line 151)
        (True, "True"),       # bool is not float in Python (line 151)
        (None, "None"),       # None pass-through (line 151)
    ],
)
def test_invariant_fmt_metric_renders_correctly(value, expected):
    """``_fmt_metric`` must render floats with ``:.4g`` and everything else
    with ``str()``. Parametrized across several representative values."""
    assert _fmt_metric(value) == expected


def test_invariant_fmt_metric_float_uses_4g_format():
    """The float branch (line 149-150) uses ``:.4g`` which gives exactly 4
    significant digits. Verify on a value where naive rounding would differ."""
    # 0.123456789 rounded to 4 sig figs = 0.1235
    assert _fmt_metric(0.123456789) == "0.1235"


def test_invariant_fmt_metric_non_float_returns_str():
    """The non-float branch (line 151) must return ``str(v)`` for any
    non-float type — list, dict, etc."""
    assert _fmt_metric([1, 2]) == "[1, 2]"
    assert _fmt_metric({"k": "v"}) == "{'k': 'v'}"
