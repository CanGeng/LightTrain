"""Functional replay of a frozen step bundle (DESIGN §18.1)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.engine._context import StepContext
from lighttrain.observability.diagnostics.frozen_step import (
    FrozenStepWriter,
    replay_step_bundle,
)


def _make_bundle(tmp_path):
    torch.manual_seed(42)
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = StepContext(step=5, epoch=0)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 4)),
    }
    writer = FrozenStepWriter(tmp_path, mode="lab", run_id="r")
    writer.snapshot(
        step=5, ctx=ctx, batch=batch, model=model, optimizer=optimizer
    )
    return writer.commit(reason="scheduled"), model, batch


def test_replay_runs_forward_and_backward(tmp_path):
    path, _, _ = _make_bundle(tmp_path)
    result = replay_step_bundle(path, loss_fn=CrossEntropyLoss())
    assert result["step"] == 5
    assert result["reason"] == "scheduled"
    assert result["loss"] is not None
    assert result["grad_norm"] is not None
    assert result["grad_norm"] >= 0.0


def test_replay_without_loss_only_forward(tmp_path):
    path, _, _ = _make_bundle(tmp_path)
    result = replay_step_bundle(path, loss_fn=None, do_backward=False)
    assert result["loss"] is None
    assert result["logits_shape"] is not None


def test_replay_inject_script_runs(tmp_path):
    path, _, _ = _make_bundle(tmp_path)
    inject = tmp_path / "inject.py"
    inject.write_text("setattr(model, '_lt_inject_marker', True)", encoding="utf-8")
    result = replay_step_bundle(path, loss_fn=None, inject=inject, do_backward=False)
    assert result["logits_shape"] is not None
