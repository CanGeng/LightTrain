"""FileSignalsCallback runtime knobs (DESIGN §20.7)."""

from __future__ import annotations

import json

import torch

from lighttrain.builtin_plugins.realtime_control.file_signals import FileSignalsCallback
from lighttrain.callbacks.base import Signal
from lighttrain.engine._context import StepContext


class _Trainer:
    def __init__(self, model, optimizer, run_dir, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._run_dir = run_dir


def _setup(tmp_path):
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    ctx = StepContext(run_dir=tmp_path, optimizer=optimizer)
    cb = FileSignalsCallback(poll_every=1)
    trainer = _Trainer(model=None, optimizer=optimizer, run_dir=tmp_path)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    return cb, ctx, optimizer


def test_lr_scale(tmp_path):
    cb, ctx, optimizer = _setup(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 0.5}), encoding="utf-8"
    )
    cb.on_step_end(step=1)
    assert optimizer.param_groups[0]["lr"] == 0.5 * 1e-3
    # File is removed after read.
    assert not (tmp_path / "control" / "lr.json").exists()


def test_stop_signal(tmp_path):
    cb, ctx, _ = _setup(tmp_path)
    (tmp_path / "control" / "stop").write_text("", encoding="utf-8")
    sig = cb.on_step_end(step=1)
    assert sig == Signal.STOP_TRAINING


def test_eval_now_sets_flag(tmp_path):
    cb, ctx, _ = _setup(tmp_path)
    (tmp_path / "control" / "eval_now").write_text("", encoding="utf-8")
    cb.on_step_end(step=1)
    assert ctx.extras.get("force_eval") is True


def test_inject_exec(tmp_path):
    cb, ctx, _ = _setup(tmp_path)
    (tmp_path / "control" / "inject.py").write_text(
        "ctx.diagnostics['injected'] = 42", encoding="utf-8"
    )
    cb.on_step_end(step=1)
    assert ctx.diagnostics.get("injected") == 42
    events = ctx.diagnostics.get("realtime_events", [])
    assert any(e["event"] == "inject" for e in events)
