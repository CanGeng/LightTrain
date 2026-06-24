"""GradFlowCallback per-layer grad norm snapshot."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.diagnostics.grad_flow import GradFlowCallback
from lighttrain.engine._context import StepContext
from tests._diagnostics import expect_nonempty


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _Trainer:
    def __init__(self, model, run_dir):
        self.model = model
        self._run_dir = run_dir


def test_grad_flow_writes_per_layer_norms(tmp_path):
    model = _Tiny()
    cb = GradFlowCallback(every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(trainer=_Trainer(model, tmp_path), ctx=ctx)
    x = torch.randn(2, 4)
    y = model(x).sum()
    y.backward()
    cb.on_backward_post(step=1, loss=y)
    cb.on_step_end(step=1)
    snaps = sorted((tmp_path / "diagnostics").glob("grad_flow_*.json"))
    expect_nonempty(snaps, tmp_path, what="a grad_flow_<step>.json snapshot")
    data = json.loads(snaps[-1].read_text(encoding="utf-8"))
    assert any(k.startswith("fc1.") for k in data)
    assert any(k.startswith("fc2.") for k in data)
