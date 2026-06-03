"""StandardEngine + StandardUpdateRule single-step round-trip."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.engine.standard import StandardEngine
from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.update_rules.standard import StandardUpdateRule
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        from lighttrain.protocols import ModelOutput

        h = self.emb(input_ids)
        logits = self.head(h)
        return ModelOutput(outputs={"logits": logits})


def _toy_batch(B=2, T=4, V=8):
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
    }


def test_engine_single_step_updates_params_and_metrics():
    torch.manual_seed(0)
    model = _ToyLM()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    rule = StandardUpdateRule(grad_clip=1.0, accumulate_grad_batches=1)
    engine = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())

    ctx = StepContext(model=model, optimizer=optim, bus=EventBus([]))
    ctx.loss_fn = CrossEntropyLoss()

    before = model.head.weight.detach().clone()
    metrics = engine.step(_toy_batch(), ctx)
    after = model.head.weight.detach().clone()

    assert "loss" in metrics
    assert metrics["loss"] > 0
    assert "grad_norm" in metrics
    assert metrics["skipped"] == 0.0
    assert not torch.equal(before, after)


def test_skip_step_signal_aborts_backward():
    from lighttrain.callbacks.base import Signal

    torch.manual_seed(0)
    model = _ToyLM()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    bus = EventBus([_Skipper()])
    rule = StandardUpdateRule()
    engine = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())

    ctx = StepContext(model=model, optimizer=optim, bus=bus)
    ctx.loss_fn = CrossEntropyLoss()

    before = model.head.weight.detach().clone()
    metrics = engine.step(_toy_batch(), ctx)
    after = model.head.weight.detach().clone()

    assert metrics["skipped"] == 1.0
    # Backward + optimizer.step were skipped: weights unchanged.
    assert torch.equal(before, after)


def test_grad_accumulation_holds_off_optimizer():
    torch.manual_seed(0)
    model = _ToyLM()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    rule = StandardUpdateRule(accumulate_grad_batches=2)
    engine = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())

    ctx = StepContext(model=model, optimizer=optim, bus=EventBus([]))
    ctx.loss_fn = CrossEntropyLoss()

    before = model.head.weight.detach().clone()
    engine.step(_toy_batch(), ctx)
    mid = model.head.weight.detach().clone()
    # First micro-step: still accumulating, no update yet.
    assert torch.equal(before, mid)
    engine.step(_toy_batch(), ctx)
    after = model.head.weight.detach().clone()
    # Second micro-step: optimizer fires.
    assert not torch.equal(before, after)
