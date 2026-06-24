"""CallbackIsolationSink writes failures + report (F3 — DESIGN §18.5)."""

from __future__ import annotations

import pytest

from lighttrain.callbacks.base import EventBus
from lighttrain.observability.diagnostics.callback_isolation import (
    CallbackIsolationSink,
    write_callback_report,
)


class _Trainer:
    def __init__(self, run_dir, bus):
        self._run_dir = run_dir
        self.bus = bus


class _Boomer:
    """Non-critical callback that always raises."""

    def on_step_end(self, **_):
        raise RuntimeError("boom")


class _Critical:
    critical = True

    def on_step_end(self, **_):
        raise RuntimeError("critical boom")


def test_sink_writes_failures_jsonl(tmp_path):
    sink = CallbackIsolationSink()
    bus = EventBus([_Boomer(), sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    bus.dispatch("on_step_end", step=1)
    bus.dispatch("on_step_end", step=2)
    log = tmp_path / "diagnostics" / "callback_failures.jsonl"
    assert log.exists()
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2


def test_critical_callback_reraises_through_bus():
    bus = EventBus([_Critical()])
    with pytest.raises(RuntimeError, match="critical boom"):
        bus.dispatch("on_step_end", step=0)


def test_three_failures_then_quarantine(tmp_path):
    boom = _Boomer()
    sink = CallbackIsolationSink()
    bus = EventBus([boom, sink], max_consecutive_failures=3)
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for s in range(1, 6):
        bus.dispatch("on_step_end", step=s)
    assert "_Boomer" in bus.quarantined


def test_callback_report_aggregates(tmp_path):
    boom = _Boomer()
    sink = CallbackIsolationSink()
    bus = EventBus([boom, sink])
    sink.on_train_start(trainer=_Trainer(tmp_path, bus))
    for s in range(1, 4):
        bus.dispatch("on_step_end", step=s)
    out = write_callback_report(tmp_path, bus=bus)
    assert out is not None and out.exists()
    body = out.read_text(encoding="utf-8")
    assert "_Boomer" in body
