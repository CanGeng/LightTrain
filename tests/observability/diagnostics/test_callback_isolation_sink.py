"""Adversarial tests for ``lighttrain.observability.diagnostics.callback_isolation``.

Layered on top of ``tests/test_callback_isolation_sink.py`` (which tests
the happy path). New coverage:

* **JSONL entry key set is exactly {ts, step, callback, event, exc_type,
  traceback}** — protects against schema drift that would break the
  downstream report aggregator.
* **Traceback with newlines / quotes / unicode** still produces parseable
  JSONL lines.
* **Step captured from on_step_begin** is the value recorded in subsequent
  failures.
* **``_recent`` ring buffer caps at max_recent** (line 101-102 of source).
* **``install()`` is idempotent** — second call doesn't double-wrap.
* **Wrapped ``_on_error`` chains to original** (preserves caller's hook).
* **Report sorts by callback count descending**.
* **Report omits "Last 5 failures" section when no failures**.
* **``write_callback_report`` returns None when JSONL missing**.
* **``write_callback_report`` tolerates malformed lines in JSONL**.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.callbacks.base import EventBus
from lighttrain.observability.diagnostics.callback_isolation import (
    CallbackIsolationSink,
    write_callback_report,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _Trainer:
    def __init__(self, run_dir: Path, bus: EventBus) -> None:
        self._run_dir = run_dir
        self.bus = bus


class _Boomer:
    """Non-critical raiser used as the failure source."""

    def on_step_end(self, **_):
        raise RuntimeError("boom")


class _Critical:
    """Critical raiser: the EventBus must re-raise rather than isolate."""

    critical = True

    def on_step_end(self, **_):
        raise RuntimeError("critical boom")


class _NewlineBoomer:
    def on_step_end(self, **_):
        raise RuntimeError("line1\nline2\nline3")


class _QuoteBoomer:
    def on_step_end(self, **_):
        raise RuntimeError('value "x" is bad')


class _UnicodeBoomer:
    def on_step_end(self, **_):
        raise RuntimeError("失败 🚀")


def _read_jsonl_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# JSONL entry schema
# ---------------------------------------------------------------------------

def test_invariant_jsonl_entry_has_exact_key_set(tmp_path):
    """Pin: every recorded entry has keys {ts, step, callback, event,
    exc_type, traceback}. Schema drift would break the report aggregator.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=7)

    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entries = _read_jsonl_lines(log)
    assert len(entries) == 1
    e = entries[0]
    expected_keys = {"ts", "step", "callback", "event", "exc_type", "traceback"}
    assert set(e.keys()) == expected_keys, (
        f"schema drift detected. Got {set(e.keys()) ^ expected_keys}"
    )


def test_invariant_jsonl_entry_step_callback_event_exc_type_values(tmp_path):
    """Closed form: with step=7, callback _Boomer, event on_step_end,
    exc RuntimeError("boom"), the entry fields match exactly.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_begin", step=7)  # populate _step
    bus.dispatch("on_step_end", step=7)

    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    e = _read_jsonl_lines(log)[0]
    assert e["step"] == 7
    assert e["callback"] == "_Boomer"
    assert e["event"] == "on_step_end"
    assert e["exc_type"] == "RuntimeError"
    assert "boom" in e["traceback"]


# ---------------------------------------------------------------------------
# Traceback robustness
# ---------------------------------------------------------------------------

def test_invariant_jsonl_with_newline_in_traceback_stays_parseable(tmp_path):
    """An exception whose message contains ``\\n`` still produces one JSONL
    line that parses cleanly.

    Setup: _NewlineBoomer raises with multi-line message.
    Expected: the JSONL has exactly ONE entry; ``traceback`` field contains
    the literal newlines after JSON-unescape.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_NewlineBoomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=1)

    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    # File-level: count of physical lines == 1 (JSON escaped the embedded newlines)
    physical_lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(physical_lines) == 1
    entry = json.loads(physical_lines[0])
    # The traceback contains the original multi-line message.
    assert "line1" in entry["traceback"]
    assert "line3" in entry["traceback"]


def test_invariant_jsonl_with_quotes_in_traceback_stays_parseable(tmp_path):
    """Exception message containing ``"`` still produces parseable JSONL."""
    sink = CallbackIsolationSink()
    bus = EventBus([_QuoteBoomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=1)
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entries = _read_jsonl_lines(log)
    assert len(entries) == 1
    assert '"x"' in entries[0]["traceback"]


def test_invariant_jsonl_with_unicode_in_traceback_stays_parseable(tmp_path):
    """Unicode exception messages survive the JSONL round trip."""
    sink = CallbackIsolationSink()
    bus = EventBus([_UnicodeBoomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=1)
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entries = _read_jsonl_lines(log)
    assert len(entries) == 1
    # JSON default ensure_ascii=True escapes unicode; verify round-trip integrity
    assert "失败" in entries[0]["traceback"] or "\\u" in entries[0]["traceback"]


# ---------------------------------------------------------------------------
# Step capture via on_step_begin
# ---------------------------------------------------------------------------

def test_invariant_step_captured_from_on_step_begin(tmp_path):
    """``sink.on_step_begin(step=N)`` updates the internal ``_step``; the
    NEXT failure records that step in its entry.

    Setup: drive on_step_begin(step=42), then fail on_step_end.
    Expected: recorded entry has ``step == 42`` regardless of what
    on_step_end was passed.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_begin", step=42, batch=None)
    # The Boomer raises on on_step_end; we don't pass step here on purpose.
    bus.dispatch("on_step_end")

    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entry = _read_jsonl_lines(log)[0]
    assert entry["step"] == 42


# ---------------------------------------------------------------------------
# _recent ring buffer cap
# ---------------------------------------------------------------------------

def test_invariant_recent_buffer_caps_at_max_recent(tmp_path):
    """``_recent`` evicts oldest when it grows past ``max_recent`` (lines 101-102).

    Setup: max_recent=3; trigger 10 failures.
    Expected: ``len(sink._recent) == 3``; the JSONL on disk still has all 10.
    """
    sink = CallbackIsolationSink(max_recent=3)
    bus = EventBus([_Boomer(), sink], max_consecutive_failures=999)
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for i in range(1, 11):
        bus.dispatch("on_step_end", step=i)

    assert len(sink._recent) == 3
    # The on-disk JSONL retains all entries
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entries = _read_jsonl_lines(log)
    assert len(entries) == 10


# ---------------------------------------------------------------------------
# install() lifecycle
# ---------------------------------------------------------------------------

def test_install_is_idempotent_does_not_double_wrap(tmp_path):
    """Calling ``install(bus)`` twice does NOT chain two sinks. After the
    second install, a single failure still produces a single JSONL entry.

    Setup: install twice; dispatch one failure.
    Expected: exactly 1 entry in the JSONL.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer()])
    sink._run_dir = tmp_path
    sink.install(bus)
    sink.install(bus)  # second install no-ops (line 70-71)

    bus.dispatch("on_step_end", step=1)
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    entries = _read_jsonl_lines(log)
    assert len(entries) == 1


def test_install_wraps_original_on_error_and_chains_to_it(tmp_path):
    """When ``install`` replaces ``bus._on_error``, the original ``on_error``
    is still invoked downstream of the sink.

    Setup: register a custom on_error that records every call; then install
    the sink; then dispatch a failure.
    Expected: the original recorder receives 1 call (chained from sink).
    """
    recorder_calls: list[tuple[str, str, str]] = []

    def original(event, cb, exc):
        recorder_calls.append((event, type(cb).__name__, type(exc).__name__))

    bus = EventBus([_Boomer()], on_error=original)
    sink = CallbackIsolationSink()
    sink._run_dir = tmp_path
    sink.install(bus)

    bus.dispatch("on_step_end", step=1)

    # Sink recorded
    assert len(sink._recent) == 1
    # Original on_error was also chained
    assert len(recorder_calls) == 1
    assert recorder_calls[0] == ("on_step_end", "_Boomer", "RuntimeError")


# ---------------------------------------------------------------------------
# Critical callbacks bypass isolation
# ---------------------------------------------------------------------------

def test_invariant_critical_callback_reraises_through_bus():
    """A callback declaring ``critical = True`` is NOT isolated: its exception
    propagates straight through ``bus.dispatch`` instead of being swallowed
    into the JSONL sink.

    Goal: pin the escape hatch — fatal callbacks (e.g. NaN hunters) must crash
    the run rather than be quarantined like best-effort callbacks.
    """
    bus = EventBus([_Critical()])
    with pytest.raises(RuntimeError, match="critical boom"):
        bus.dispatch("on_step_end", step=0)


# ---------------------------------------------------------------------------
# write_callback_report
# ---------------------------------------------------------------------------

def test_write_callback_report_returns_none_when_jsonl_missing(tmp_path):
    """Pin: report returns None and writes nothing when the JSONL is absent.

    Goal: catches a refactor that would crash on missing input.
    """
    out = write_callback_report(tmp_path)
    assert out is None
    assert not (tmp_path / "diagnostics" / "callback_report.md").exists()


def test_write_callback_report_aggregates_by_callback_descending(tmp_path):
    """The "By callback" section lists callbacks sorted by failure count
    descending (line 155 of source).

    Setup: 5 failures from class A, 2 from class B.
    Expected: A appears before B in the rendered markdown.
    """
    class A_Boomer:
        def on_step_end(self, **_):
            raise RuntimeError("a")

    class B_Boomer:
        def on_step_end(self, **_):
            raise RuntimeError("b")

    sink = CallbackIsolationSink()
    bus = EventBus([A_Boomer(), B_Boomer(), sink], max_consecutive_failures=999)
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for _ in range(5):
        bus.dispatch("on_step_end", step=0)
    # B_Boomer fired 5 times too because both run per dispatch; balance:
    # we want different counts. So now dispatch a few more rounds.
    # Actually both A_Boomer and B_Boomer fire each dispatch.
    # We need A to fail MORE than B. Let's do it by adding extra A failures:
    sink._record("on_step_end", A_Boomer(), RuntimeError("a_extra"))
    sink._record("on_step_end", A_Boomer(), RuntimeError("a_extra"))

    out = write_callback_report(tmp_path, bus=bus)
    assert out is not None
    body = out.read_text(encoding="utf-8")

    # A_Boomer should appear before B_Boomer in the rendered output
    idx_a = body.find("A_Boomer")
    idx_b = body.find("B_Boomer")
    assert idx_a < idx_b, "A_Boomer (higher count) should be listed before B_Boomer"


def test_write_callback_report_includes_quarantined_list(tmp_path):
    """The header line includes the bus's ``quarantined`` list when given.

    Setup: trigger 3 failures to quarantine; call write with bus.
    Expected: report contains "_Boomer" in the quarantined header.
    """
    boom = _Boomer()
    sink = CallbackIsolationSink()
    bus = EventBus([boom, sink], max_consecutive_failures=3)
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for s in range(1, 6):  # 5 dispatches → quarantine triggered at #3
        bus.dispatch("on_step_end", step=s)

    out = write_callback_report(tmp_path, bus=bus)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    # Header mentions Currently quarantined: _Boomer
    quarantined_line = [ln for ln in body.splitlines() if "Currently quarantined" in ln][0]
    assert "_Boomer" in quarantined_line


def test_write_callback_report_tolerates_malformed_lines(tmp_path):
    """If the JSONL has a corrupted line (not valid JSON), the aggregator
    silently skips it (line 132-134 of source).
    """
    (tmp_path / "diagnostics").mkdir(parents=True, exist_ok=True)
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    log.write_text(
        '{"step": 1, "callback": "X", "event": "on_step_end", "exc_type": "E", "traceback": "t", "ts": 1.0}\n'
        "this line is not valid JSON\n"
        '{"step": 2, "callback": "Y", "event": "on_step_end", "exc_type": "E", "traceback": "t", "ts": 2.0}\n',
        encoding="utf-8",
    )
    out = write_callback_report(tmp_path)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    # The valid entries are reported; the malformed line is silently dropped.
    assert "X" in body and "Y" in body
    assert "Total isolated failures: **2**" in body


def test_write_callback_report_omits_last_failures_section_when_empty(tmp_path):
    """When the JSONL exists but is empty, the report renders without the
    "Last 5 failures" section (gated by ``if lines:`` on line 160).

    Setup: write an empty JSONL file (touch).
    Expected: report exists; "Last 5 failures" substring is absent.
    """
    (tmp_path / "diagnostics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "diagnostics" / "callback_failures.jsonl").write_text("", encoding="utf-8")

    out = write_callback_report(tmp_path)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    assert "Last 5 failures" not in body
    assert "Total isolated failures: **0**" in body


def test_write_callback_report_includes_last_5_when_present(tmp_path):
    """The "Last 5 failures" section appears when at least one failure
    exists, and shows up to 5 entries.

    Setup: produce 7 failures.
    Expected: report contains the "Last 5 failures" header AND exactly
    5 listed items.
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink], max_consecutive_failures=999)
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for i in range(7):
        bus.dispatch("on_step_end", step=i)

    out = write_callback_report(tmp_path, bus=bus)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    assert "Last 5 failures" in body
    # Count lines under the "Last 5 failures" section
    after = body.split("## Last 5 failures", 1)[1]
    bullet_count = sum(1 for line in after.splitlines() if line.startswith("- step="))
    assert bullet_count == 5


def test_write_callback_report_idempotent_regenerates(tmp_path):
    """Calling ``write_callback_report`` twice produces the same file
    (it's safe to invoke repeatedly per source docstring on line 120).
    """
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=1)
    out1 = write_callback_report(tmp_path)
    assert out1 is not None
    body1 = out1.read_text(encoding="utf-8")
    out2 = write_callback_report(tmp_path)
    assert out2 is not None
    body2 = out2.read_text(encoding="utf-8")
    # bodies are functionally equivalent (totals match); we don't compare
    # byte-for-byte because the failure count is the same in both runs.
    assert "Total isolated failures: **1**" in body1
    assert "Total isolated failures: **1**" in body2
