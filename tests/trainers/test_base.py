"""Adversarial tests for ``Trainer`` ABC at lighttrain/trainers/base.py.

Pins:
  - Constructor wires EventBus into ctx.bus, optimizer/scheduler into ctx.*
  - state_dict / load_state_dict roundtrip
  - train_step normalization (dict → StepOutput, StepOutput passthrough,
    invalid type raises clearly)
  - is_main returns True with no parallel_ctx (single-GPU default)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.protocols import StepOutput
from lighttrain.trainers.base import Trainer


class _FakeEngine:
    pass


class _FakeDM:
    def train_loader(self):
        while True:
            yield {}


def _base_kwargs(**over):
    base = dict(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=MagicMock(),
        max_steps=3,
    )
    base.update(over)
    return base


class _DictStepTrainer(Trainer):
    """Minimal Trainer subclass that returns a dict from _step."""

    def fit(self, *, steps=None):  # pragma: no cover — not exercised here
        ...

    def _step(self, batch):
        return {"loss": 0.25, "acc": 0.9}


class _StepOutputTrainer(Trainer):
    def fit(self, *, steps=None):  # pragma: no cover
        ...

    def _step(self, batch):
        return StepOutput(loss=0.42, metrics={"loss": 0.42, "ppl": 2.0})


class _BadTrainer(Trainer):
    def fit(self, *, steps=None):  # pragma: no cover
        ...

    def _step(self, batch):
        return 42  # not a dict, not StepOutput


# ===========================================================================
# Constructor wiring
# ===========================================================================


def test_trainer_constructor_wires_ctx_bus_to_event_bus():
    """Goal: ``trainer.ctx.bus`` is the SAME object as ``trainer.bus``.

    Catches a refactor that creates two distinct EventBus instances or
    forgets to set ``ctx.bus`` at all — callbacks would silently miss
    update_rule's lifecycle events.
    """
    trainer = _DictStepTrainer(**_base_kwargs())
    assert trainer.ctx.bus is trainer.bus
    assert isinstance(trainer.bus, EventBus)


def test_trainer_constructor_wires_optimizer_and_scheduler_into_ctx():
    """Goal: optimizer and scheduler are exposed on ctx for the update_rule
    to use.

    Catches a refactor that drops lines 48-49 in base.py — update_rule would
    see ``ctx.optimizer is None`` and raise.
    """
    opt = MagicMock(name="optimizer")
    sched = MagicMock(name="scheduler")
    trainer = _DictStepTrainer(**_base_kwargs(optimizer=opt, scheduler=sched))

    assert trainer.ctx.optimizer is opt
    assert trainer.ctx.scheduler is sched


def test_trainer_constructor_wires_logger_into_ctx():
    """Goal: logger flows into ctx so update_rule / loss_fn can log on it."""
    logger = MagicMock(name="logger")
    trainer = _DictStepTrainer(**_base_kwargs(logger=logger))
    assert trainer.ctx.logger is logger


def test_trainer_callbacks_list_is_distinct_from_constructor_arg():
    """Goal: passing ``callbacks=[cb_a]`` makes self.callbacks contain it
    but stores a fresh list (not the caller's reference).

    Catches a refactor that uses the constructor list directly — caller
    mutations to the list would leak into the trainer.
    """
    cb_a = object()
    caller_list = [cb_a]
    trainer = _DictStepTrainer(**_base_kwargs(callbacks=caller_list))

    # cb_a is registered
    assert cb_a in trainer.callbacks
    # but mutations to caller_list don't affect trainer
    caller_list.append(object())
    assert len(trainer.callbacks) == 1


# ===========================================================================
# Step normalization
# ===========================================================================


def test_train_step_normalizes_dict_to_stepoutput():
    """Goal: dict result is wrapped into StepOutput; loss is extracted from
    the 'loss' key; the full dict survives as ``metrics``.
    """
    trainer = _DictStepTrainer(**_base_kwargs())
    out = trainer.train_step({})

    assert isinstance(out, StepOutput)
    assert out.loss == 0.25
    assert out.metrics == {"loss": 0.25, "acc": 0.9}


def test_train_step_passes_through_stepoutput_unchanged():
    """Goal: when _step returns a StepOutput, train_step returns it as-is."""
    trainer = _StepOutputTrainer(**_base_kwargs())
    out = trainer.train_step({})

    assert isinstance(out, StepOutput)
    assert out.loss == 0.42
    assert out.metrics == {"loss": 0.42, "ppl": 2.0}


def test_train_step_raises_on_invalid_return_type():
    """Goal: returning anything other than dict/StepOutput raises a clear
    TypeError with the actual type name.
    """
    trainer = _BadTrainer(**_base_kwargs())
    with pytest.raises(TypeError, match="must return StepOutput or dict"):
        trainer.train_step({})


# ===========================================================================
# state_dict / load_state_dict
# ===========================================================================


def test_state_dict_load_state_dict_roundtrip():
    """Goal: step/epoch/global_step/max_steps roundtrip exactly through
    state_dict → load_state_dict.
    """
    trainer = _DictStepTrainer(**_base_kwargs(max_steps=10))
    trainer.ctx.step = 5
    trainer.ctx.epoch = 2
    trainer.ctx.global_step = 7

    sd = trainer.state_dict()
    assert sd == {"step": 5, "epoch": 2, "global_step": 7, "max_steps": 10}

    trainer2 = _DictStepTrainer(**_base_kwargs(max_steps=1))
    trainer2.load_state_dict(sd)
    assert trainer2.ctx.step == 5
    assert trainer2.ctx.epoch == 2
    assert trainer2.ctx.global_step == 7
    assert trainer2.max_steps == 10


# ===========================================================================
# Distributed defaults
# ===========================================================================


def test_is_main_returns_true_when_no_parallel_ctx():
    """Goal: with no parallel_ctx (single-GPU), ``_is_main`` returns True
    so logging and checkpoints always fire.

    Catches a refactor that defaults to False or raises when parallel_ctx
    is missing.
    """
    trainer = _DictStepTrainer(**_base_kwargs())
    assert trainer.ctx.parallel_ctx is None
    assert trainer._is_main() is True


def test_pctx_property_falls_back_to_single_gpu():
    """Goal: ``_pctx`` returns a ParallelContext.single_gpu() proxy when
    none is set.
    """
    from lighttrain.distributed._context import ParallelContext

    trainer = _DictStepTrainer(**_base_kwargs())
    pctx = trainer._pctx
    assert isinstance(pctx, ParallelContext)
    assert pctx.is_main_process is True
