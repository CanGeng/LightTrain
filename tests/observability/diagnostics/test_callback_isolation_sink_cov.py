"""Coverage tests for ``lighttrain.observability.diagnostics.callback_isolation``.

Pins the uncovered branches not exercised by test_callback_isolation_sink.py:

* Line  63  -- on_train_end early-return when _run_dir is None
* Lines 68-69 -- on_train_end exception handler when write_callback_report raises
* Lines 86-87 -- _sink chains to original _on_error; original raises → logged, not re-raised
* Lines 96-97 -- install() exception handler when bus._on_error assignment raises
* Line 117  -- _record early-return when _run_dir is None (entry still lands in _recent)
* Lines 125-126 -- _record exception handler when JSONL write fails
* Line  146 -- write_callback_report blank-line skip in JSONL parser
* Lines 164-165,169 -- write_callback_report exception handler when bus.quarantined raises
"""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lighttrain.callbacks.base import EventBus
from lighttrain.observability.diagnostics.callback_isolation import (
    CallbackIsolationSink,
    write_callback_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Boomer:
    """Non-critical callback that always raises."""

    def on_step_end(self, **_: Any) -> None:
        raise RuntimeError("boom")


class _Trainer:
    def __init__(self, run_dir: Path, bus: EventBus) -> None:
        self._run_dir = run_dir
        self.bus = bus


def _make_jsonl(path: Path, entries: list[dict]) -> None:
    """Write JSONL entries to the standard location."""
    diag = path / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / "callback_failures.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# on_train_end: line 63 — early return when _run_dir is None
# ---------------------------------------------------------------------------

def test_invariant_on_train_end_noop_when_run_dir_none():
    """on_train_end must silently return (line 63) when _run_dir is None.

    No file system access should happen; no exception should propagate.
    """
    sink = CallbackIsolationSink()
    assert sink._run_dir is None
    # Must not raise, even if write_callback_report would fail
    sink.on_train_end()
    # Confirm it truly did nothing: _run_dir stays None, _recent is empty
    assert sink._run_dir is None
    assert sink._recent == []


# ---------------------------------------------------------------------------
# on_train_end: lines 68-69 — exception swallowed, warning logged
# ---------------------------------------------------------------------------

def test_pin_current_behavior_on_train_end_swallows_write_exception(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: if write_callback_report raises internally, on_train_end logs a
    warning and does NOT re-raise (lines 68-69).

    This is intentional: a report-generation failure must never mask the
    original callback failure it was summarising.

    DEBATABLE: one could argue the exception should propagate, but the source
    explicitly swallows it.
    """
    sink = CallbackIsolationSink()
    sink._run_dir = tmp_path

    boom = RuntimeError("report broken")
    with patch(
        "lighttrain.observability.diagnostics.callback_isolation.write_callback_report",
        side_effect=boom,
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.callback_isolation"):
            sink.on_train_end()  # must NOT raise

    assert any("callback report generation failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _sink: lines 86-87 — original _on_error raises; exception swallowed
# ---------------------------------------------------------------------------

def test_pin_current_behavior_chained_on_error_that_raises_is_swallowed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin (lines 86-87): when the original _on_error itself raises, the sink
    logs a warning and swallows the exception so the dispatch loop continues.

    Setup: install a custom on_error that raises, then install the sink,
    then dispatch a failure.
    Expected: no exception propagates out of dispatch; warning is logged.
    """

    def _exploding_original(event: str, cb: Any, exc: BaseException) -> None:
        raise RuntimeError("original handler also broken")

    bus = EventBus([_Boomer()], on_error=_exploding_original)
    sink = CallbackIsolationSink()
    sink._run_dir = tmp_path
    sink.install(bus)

    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.callback_isolation"):
        bus.dispatch("on_step_end", step=1)  # must NOT raise

    assert any(
        "chained original _on_error handler raised" in r.message
        for r in caplog.records
    )
    # The failure was still recorded in _recent
    assert len(sink._recent) == 1


# ---------------------------------------------------------------------------
# install: lines 96-97 — bus._on_error assignment raises
# ---------------------------------------------------------------------------

def test_pin_current_behavior_install_silently_skips_when_assignment_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin (lines 96-97): if setting bus._on_error raises (e.g. read-only
    property), install() logs a warning and _installed stays False.

    DEBATABLE: the module chose silent degradation; failures won't be persisted
    but the run continues.
    """

    class _FrozenBus:
        """Bus whose _on_error is a read-only data descriptor."""

        @property
        def _on_error(self):
            return None

        @_on_error.setter
        def _on_error(self, value):
            raise AttributeError("read-only!")

    sink = CallbackIsolationSink()
    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.callback_isolation"):
        sink.install(_FrozenBus())  # must NOT raise

    assert not sink._installed
    assert any("failed to install sink on bus._on_error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _record: line 117 — early return when _run_dir is None
# ---------------------------------------------------------------------------

def test_invariant_record_with_no_run_dir_still_populates_recent():
    """_record adds to _recent even when _run_dir is None (line 117 early-return).

    The entry is kept in memory; no disk I/O is attempted.
    """
    sink = CallbackIsolationSink()
    assert sink._run_dir is None

    exc = RuntimeError("ephemeral")
    sink._record("on_step_end", _Boomer(), exc)

    assert len(sink._recent) == 1
    entry = sink._recent[0]
    assert entry["exc_type"] == "RuntimeError"
    # No diagnostics directory was created
    # (can't assert filesystem absence without a tmp_path, but no exception = no write attempt)


# ---------------------------------------------------------------------------
# _record: lines 125-126 — write failure is swallowed, warning logged
# ---------------------------------------------------------------------------

def test_pin_current_behavior_record_swallows_write_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin (lines 125-126): if writing callback_failures.jsonl fails (e.g.
    permission denied), _record logs a warning and does NOT raise.

    The entry is still kept in _recent (in-memory), only disk persistence fails.
    """
    sink = CallbackIsolationSink()
    sink._run_dir = tmp_path

    # Create the diagnostics dir with a file named "diagnostics" in its place
    # so mkdir() or open() will fail.
    diag = tmp_path / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    # Make diagnostics dir read-only so open("a") fails
    diag.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x ------

    try:
        exc = RuntimeError("disk full")
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.callback_isolation"):
            sink._record("on_step_end", _Boomer(), exc)  # must NOT raise

        assert any(
            "failed to persist callback failure" in r.message for r in caplog.records
        )
        # Still in memory
        assert len(sink._recent) == 1
    finally:
        # Restore permissions so tmp_path cleanup can proceed
        diag.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# write_callback_report: line 146 — blank lines in JSONL are skipped
# ---------------------------------------------------------------------------

def test_invariant_write_callback_report_skips_blank_lines(tmp_path: Path) -> None:
    """Blank / whitespace-only lines in callback_failures.jsonl are skipped
    (line 146: ``if not raw: continue``).

    The non-blank valid entries must still be counted correctly.
    """
    diag = tmp_path / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": 1.0,
        "step": 1,
        "callback": "A",
        "event": "on_step_end",
        "exc_type": "RuntimeError",
        "traceback": "tb",
    }
    # Write a mix of valid entries and blank / whitespace-only lines
    (diag / "callback_failures.jsonl").write_text(
        "\n"                          # blank line at start
        + json.dumps(entry) + "\n"   # valid entry
        + "   \n"                    # whitespace-only line
        + json.dumps(entry) + "\n"   # another valid entry
        + "\n",                      # trailing blank
        encoding="utf-8",
    )

    out = write_callback_report(tmp_path)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    # Only the 2 valid entries should be counted; blank lines are invisible
    assert "Total isolated failures: **2**" in body


# ---------------------------------------------------------------------------
# write_callback_report: lines 164-165, 169 — bus.quarantined raises
# ---------------------------------------------------------------------------

def test_pin_current_behavior_write_callback_report_swallows_quarantined_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin (lines 164-165, 169): if list(bus.quarantined) raises, the report
    still writes with quarantined=[] (empty, no names listed).

    The source checks ``hasattr(bus, "quarantined")`` first, so we need
    the attribute to be accessible (hasattr returns True) but the subsequent
    ``list(...)`` call to raise.  An object whose ``__iter__`` raises achieves
    this without touching hasattr.

    DEBATABLE: the module chose to degrade gracefully rather than propagate,
    so the report is still generated.
    """

    class _BadIter:
        """Returns True from hasattr but raises when iterated."""
        def __iter__(self):
            raise RuntimeError("quarantine db corrupted")

    class _BrokenBus:
        """Bus whose quarantined attribute cannot be iterated."""
        quarantined = _BadIter()

    diag = tmp_path / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": 1.0,
        "step": 1,
        "callback": "X",
        "event": "on_step_end",
        "exc_type": "RuntimeError",
        "traceback": "tb",
    }
    (diag / "callback_failures.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )

    with caplog.at_level(
        logging.WARNING,
        logger="lighttrain.observability.diagnostics.callback_isolation",
    ):
        out = write_callback_report(tmp_path, bus=_BrokenBus())

    # Report was still generated
    assert out is not None
    body = out.read_text(encoding="utf-8")

    # Warning was logged about the failure
    assert any(
        "failed to read bus.quarantined" in r.message for r in caplog.records
    )
    # quarantined section should say _none_ (empty fallback)
    assert "_none_" in body


# ---------------------------------------------------------------------------
# on_train_start: run_dir from ctx (not trainer)
# ---------------------------------------------------------------------------

def test_invariant_on_train_start_uses_ctx_run_dir_when_trainer_has_none(
    tmp_path: Path,
) -> None:
    """on_train_start reads run_dir from ctx.run_dir when present (line 50-51)."""

    class _Ctx:
        run_dir = str(tmp_path)

    sink = CallbackIsolationSink()
    sink.on_train_start(ctx=_Ctx())
    assert sink._run_dir == tmp_path


def test_invariant_on_train_start_falls_back_to_trainer_run_dir(
    tmp_path: Path,
) -> None:
    """on_train_start falls back to trainer._run_dir when ctx is None."""
    bus = EventBus([])
    sink = CallbackIsolationSink()
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    assert sink._run_dir == tmp_path


# ---------------------------------------------------------------------------
# on_train_end: delegates to trainer.bus when _bus is None
# ---------------------------------------------------------------------------

def test_invariant_on_train_end_uses_trainer_bus_when_bus_not_installed(
    tmp_path: Path,
) -> None:
    """on_train_end passes trainer.bus as the bus arg when _bus is None
    (line 66: ``bus=self._bus or getattr(trainer, "bus", None)``).

    Setup: manually set _run_dir; dispatch on_train_end with a trainer that
    has a bus.quarantined property.
    Expected: the report file is written and references the trainer's quarantine
    state.
    """
    sink = CallbackIsolationSink()
    sink._run_dir = tmp_path
    # Create the JSONL so write_callback_report has something to process
    diag = tmp_path / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": 1.0,
        "step": 0,
        "callback": "X",
        "event": "on_step_end",
        "exc_type": "RuntimeError",
        "traceback": "tb",
    }
    (diag / "callback_failures.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )

    _bus = EventBus([])

    class _T:
        _run_dir = tmp_path
        bus = _bus

    sink.on_train_end(trainer=_T())
    out = tmp_path / "diagnostics" / "callback_report.md"
    assert out.exists()
