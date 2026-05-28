"""LoggerBus fan-out + per-backend exception isolation."""

from __future__ import annotations

import json
from pathlib import Path

from lighttrain.logging._bus import LoggerBus
from lighttrain.logging.backends.console import ConsoleLogger
from lighttrain.logging.backends.jsonl import JSONLLogger


class _Recorder:
    def __init__(self) -> None:
        self.scalars: list[tuple[dict, int]] = []
        self.text: list[tuple[str, int]] = []

    def log_scalars(self, scalars, step):
        self.scalars.append((dict(scalars), int(step)))

    def log_text(self, text, step):
        self.text.append((str(text), int(step)))


class _Broken:
    def log_scalars(self, scalars, step):
        raise RuntimeError("boom")


def test_bus_fans_out_to_all_backends():
    a, b = _Recorder(), _Recorder()
    bus = LoggerBus([a, b])
    bus.log_scalars({"loss": 1.0}, step=1)
    assert a.scalars == [({"loss": 1.0}, 1)]
    assert b.scalars == [({"loss": 1.0}, 1)]


def test_bus_isolates_exceptions(capsys):
    good = _Recorder()
    bus = LoggerBus([_Broken(), good])
    bus.log_scalars({"loss": 1.0}, step=1)
    # The good backend still got the record.
    assert good.scalars == [({"loss": 1.0}, 1)]
    err = capsys.readouterr().err
    assert "boom" in err or "raised" in err


def test_log_dict_prefix_namespaces_keys():
    rec = _Recorder()
    LoggerBus([rec]).log_dict({"loss": 0.5}, step=3, prefix="train")
    assert rec.scalars == [({"train/loss": 0.5}, 3)]


def test_jsonl_backend_writes_one_line_per_record(tmp_path: Path):
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=1)
    j.log_scalars({"loss": 0.4}, step=2)
    j.close()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["step"] == 1 and parsed[0]["loss"] == 0.5
    assert parsed[1]["step"] == 2 and parsed[1]["loss"] == 0.4


def test_console_backend_throttles(capsys):
    cl = ConsoleLogger(log_every=10)
    cl.log_scalars({"loss": 1.0}, step=5)  # not a multiple → silent
    cl.log_scalars({"loss": 1.0}, step=10)  # multiple of 10 → printed
    out = capsys.readouterr().out
    # Rich uses ANSI; check core substrings only.
    assert out.count("step=") == 1
