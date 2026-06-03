"""StandardUpdateRule RETRY_STEP true replay (M4 — Phase A2)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.builtin_plugins.update_rules.standard import StandardUpdateRule
from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext


def _make_step_ctx(model, optimizer, bus, loss_fn):
    ctx = StepContext()
    ctx.model = model
    ctx.optimizer = optimizer
    ctx.loss_fn = loss_fn
    ctx.bus = bus
    ctx.metrics = {}
    return ctx


class _RetryThenOK:
    """Returns RETRY_STEP the first N times on_loss_computed fires."""

    def __init__(self, fire_n_retries: int):
        self.fire_n_retries = fire_n_retries
        self.called = 0

    def on_loss_computed(self, **_):
        self.called += 1
        if self.called <= self.fire_n_retries:
            return Signal.RETRY_STEP
        return Signal.CONTINUE


def test_retry_then_succeed_replays_forward():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cb = _RetryThenOK(fire_n_retries=2)
    bus = EventBus([cb])
    ctx = _make_step_ctx(model, optimizer, bus, CrossEntropyLoss())
    rule = StandardUpdateRule(max_retries=5)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 4)),
    }
    metrics = rule.step(model, batch, ctx)
    assert cb.called == 3  # initial + 2 retries → 3 dispatches
    assert metrics["retries"] == 2.0
    assert "retry_exhausted" not in metrics
    # final signal was CONTINUE so normal step path ran (skipped=0).
    assert metrics["skipped"] == 0.0


def test_retry_exhausted_falls_back_to_skip():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cb = _RetryThenOK(fire_n_retries=999)  # never satisfied
    bus = EventBus([cb])
    ctx = _make_step_ctx(model, optimizer, bus, CrossEntropyLoss())
    rule = StandardUpdateRule(max_retries=2)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 4)),
    }
    metrics = rule.step(model, batch, ctx)
    assert metrics["retries"] == 2.0
    assert metrics["retry_exhausted"] == 1.0
    assert metrics["skipped"] == 1.0
    # Engine writes ctx.extras["loss_signal"] = SKIP_STEP after exhaustion.
    assert int(ctx.extras["loss_signal"]) == int(Signal.SKIP_STEP)
