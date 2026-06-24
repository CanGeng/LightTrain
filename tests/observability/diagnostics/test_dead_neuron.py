"""DeadNeuronCallback rolling stats."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.observability.diagnostics.dead_neuron import (
    DeadNeuronCallback,
)
from lighttrain.engine._context import StepContext
from tests._diagnostics import expect_nonempty


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        return self.fc(x)


class _Trainer:
    def __init__(self, model, run_dir):
        self.model = model
        self._run_dir = run_dir


def test_dead_neuron_writes_snapshot(tmp_path):
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(trainer=_Trainer(model, tmp_path), ctx=ctx)
    for s in range(1, 5):
        _ = model(torch.randn(2, 4))
        cb.on_step_end(step=s)
    cb.on_train_end()
    files = sorted((tmp_path / "diagnostics").glob("dead_neurons_*.json"))
    expect_nonempty(files, tmp_path, what="a dead_neurons_<step>.json snapshot")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    assert "fc" in payload
    assert "zero_ratio_mean" in payload["fc"]
