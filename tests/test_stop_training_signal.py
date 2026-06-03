"""STOP_TRAINING from ``on_loss_computed`` must actually stop the fit loop.

Pre-fix, the StandardUpdateRule collapsed SKIP_STEP / STOP_TRAINING / RETRY_STEP
into the same ``skipped=1`` path and the trainer kept looping.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.builtin_plugins.engine.standard import StandardEngine
from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.protocols import ModelOutput
from lighttrain.builtin_plugins.update_rules.standard import StandardUpdateRule


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return ModelOutput(outputs={"logits": self.head(self.emb(input_ids))})


def _batch():
    return {
        "input_ids": torch.randint(0, 8, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.randint(0, 8, (2, 4)),
    }


class _Stopper:
    def on_loss_computed(self, **_):
        return Signal.STOP_TRAINING


def test_stop_training_from_on_loss_computed_propagates_to_ctx_extras():
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    bus = EventBus([_Stopper()])
    rule = StandardUpdateRule()
    eng = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())
    ctx = StepContext(model=model, optimizer=opt, bus=bus)
    ctx.loss_fn = CrossEntropyLoss()

    metrics = eng.step(_batch(), ctx)
    # Backwards-compat: the step is still recorded as skipped (no backward).
    assert metrics["skipped"] == 1.0
    # New: the strongest signal is surfaced via ctx.extras so trainers can stop.
    assert ctx.extras.get("loss_signal") == int(Signal.STOP_TRAINING)


def test_skip_step_does_not_set_stop_signal():
    """SKIP_STEP is the milder of the three — must NOT set STOP_TRAINING."""
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    bus = EventBus([_Skipper()])
    rule = StandardUpdateRule()
    eng = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())
    ctx = StepContext(model=model, optimizer=opt, bus=bus)
    ctx.loss_fn = CrossEntropyLoss()

    metrics = eng.step(_batch(), ctx)
    assert metrics["skipped"] == 1.0
    assert ctx.extras.get("loss_signal") == int(Signal.SKIP_STEP)
    assert ctx.extras["loss_signal"] != int(Signal.STOP_TRAINING)


def test_pretrain_trainer_stops_on_loss_computed_stop_signal():
    """End-to-end: a 3-step fit() with a STOP_TRAINING-from-on_loss_computed
    callback must stop after 1 step, not run to completion."""
    from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer

    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    class _DM:
        def train_loader(self):
            return iter([_batch() for _ in range(5)])

        def val_loader(self):
            return None

        def state_dict(self):
            return {}

    rule = StandardUpdateRule()
    eng = StandardEngine(update_rule=rule, loss_fn=CrossEntropyLoss())
    trainer = PretrainTrainer(
        engine=eng,
        data_module=_DM(),
        optimizer=opt,
        model=model,
        callbacks=[_Stopper()],
        max_steps=5,
        log_every=1,
        ckpt_every=0,
    )
    trainer.ctx.loss_fn = CrossEntropyLoss()
    trainer.fit()
    # Stop should fire on the very first step (step counter advanced once).
    assert trainer.ctx.step == 1
