"""Adversarial tests for GRPOTrainer — fit lifecycle, engine bypass,
loss_signal clearing, callback chain order.

GRPO is similar to PPO in structure (same RL flow), but has no value head
and no target_kl early stop. Group-relative advantage normalization happens
inside GRPOLoss, not the trainer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.protocols import ModelOutput
from lighttrain.trainers.grpo import GRPOTrainer


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _FakeEngine:
    pass


class _FakeDM:
    def train_loader(self):
        V, T, B = 16, 4, 2
        while True:
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }


def _reward(ids, batch):
    n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
    return [1.0] * n


def _make_grpo(*, model=None, callbacks=None, engine=None, **over) -> GRPOTrainer:
    if model is None:
        model = _TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return GRPOTrainer(
        engine=engine if engine is not None else _FakeEngine(),
        data_module=_FakeDM(),
        optimizer=opt,
        model=model,
        callbacks=callbacks,
        max_steps=over.pop("max_steps", 1),
        group_size=over.pop("group_size", 2),
        ppo_epochs=over.pop("ppo_epochs", 1),
        mini_batch_size=over.pop("mini_batch_size", 2),
        max_new_tokens=over.pop("max_new_tokens", 4),
        reward_fn=over.pop("reward_fn", _reward),
        **over,
    )


def _stub_rollout_phase(trainer: GRPOTrainer, rewards: torch.Tensor) -> None:
    trainer._rollout_engine.rollout = MagicMock(return_value=[])
    trainer._buffer.clear = MagicMock()
    trainer._buffer.add = MagicMock()
    trainer._buffer.all_rewards = MagicMock(return_value=rewards)
    trainer._buffer.batches = MagicMock(return_value=iter([]))


def _grpo_batch(V: int = 16, T: int = 4, B: int = 4) -> dict:
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.tensor([1.0, 0.5, 0.5, -0.5]),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }


# ===========================================================================
# fit lifecycle
# ===========================================================================


def test_grpo_fit_lifecycle_strict_order():
    """Goal: pin exact temporal order of fit-level events.

    Expected:
      [on_train_start, on_epoch_begin, on_rollout_begin, on_rollout_end,
       on_reward_computed, on_train_end]
    """
    events: list[str] = []

    class _Rec:
        def on_train_start(self, **_): events.append("on_train_start")
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")
        def on_rollout_begin(self, **_): events.append("on_rollout_begin")
        def on_rollout_end(self, **_): events.append("on_rollout_end")
        def on_reward_computed(self, **_): events.append("on_reward_computed")
        def on_train_end(self, **_): events.append("on_train_end")

    trainer = _make_grpo(callbacks=[_Rec()], max_steps=1)
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    trainer.fit()

    assert events == [
        "on_train_start",
        "on_epoch_begin",
        "on_rollout_begin",
        "on_rollout_end",
        "on_reward_computed",
        "on_train_end",
    ]


def test_grpo_fit_emits_epoch_end_then_epoch_begin_on_iterator_exhaustion():
    """Pin lines 139-143 in grpo.py: on iterator exhaustion,
    on_epoch_end → epoch++ → on_epoch_begin in that order.
    """
    events: list[str] = []

    class _Rec:
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")

    class _OneBatchIter:
        def __iter__(self_inner):
            V, T, B = 16, 4, 2
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }

    class _OneShotDM:
        def train_loader(self):
            return _OneBatchIter()

    model = _TinyLM()
    trainer = GRPOTrainer(
        engine=_FakeEngine(),
        data_module=_OneShotDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Rec()],
        max_steps=2,
        group_size=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_reward,
    )
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    trainer.fit()

    assert events[:3] == ["on_epoch_begin", "on_epoch_end", "on_epoch_begin"]


def test_grpo_fit_raises_when_model_is_none():
    trainer = _make_grpo()
    trainer.model = None
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.fit()


def test_grpo_fit_raises_when_optimizer_is_none():
    trainer = _make_grpo()
    trainer.optimizer = None
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        trainer.fit()


def test_grpo_fit_raises_when_reward_fn_is_none():
    trainer = _make_grpo()
    trainer.reward_fn = None
    with pytest.raises(RuntimeError, match="reward_fn is not set"):
        trainer.fit()


# ===========================================================================
# Engine bypass
# ===========================================================================


def test_grpo_step_bypasses_standard_engine_step():
    """Goal: GRPOTrainer must NOT call engine.step — same contract as PPO.

    Catches a refactor that routes GRPO through the engine, which would
    silently apply StandardUpdateRule and re-forward the model.
    """
    engine_mock = MagicMock()
    trainer = _make_grpo(engine=engine_mock)

    trainer._grpo_step(_grpo_batch())

    engine_mock.step.assert_not_called()


# ===========================================================================
# loss_signal clearing
# ===========================================================================


def test_grpo_step_clears_loss_signal_extras_per_call():
    """Goal: line 283 in grpo.py — ``_step`` pops ``loss_signal`` from
    ctx.extras before delegating to ``_grpo_step``.
    """
    trainer = _make_grpo()
    trainer.ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)

    trainer._step(_grpo_batch())

    assert "loss_signal" not in trainer.ctx.extras


# ===========================================================================
# Per-step callback chain (strict order)
# ===========================================================================


def test_grpo_step_fires_full_callback_chain_in_strict_order():
    """Goal: ordered list of per-step events fired by RLUpdateRule under GRPO.
    Same expected sequence as PPO (both use RLUpdateRule).
    """
    events: list[str] = []

    class _Rec:
        def on_step_begin(self, **_): events.append("on_step_begin")
        def on_loss_computed(self, **_): events.append("on_loss_computed")
        def on_backward_pre(self, **_): events.append("on_backward_pre")
        def on_backward_post(self, **_): events.append("on_backward_post")
        def on_clip_grad(self, **_): events.append("on_clip_grad")
        def on_optimizer_step_pre(self, **_): events.append("on_optimizer_step_pre")
        def on_optimizer_step_post(self, **_): events.append("on_optimizer_step_post")
        def on_zero_grad(self, **_): events.append("on_zero_grad")
        def on_step_end(self, **_): events.append("on_step_end")

    trainer = _make_grpo(callbacks=[_Rec()])
    trainer._grpo_step(_grpo_batch())

    expected = [
        "on_step_begin",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
        "on_step_end",
    ]
    assert events == expected


# ===========================================================================
# GRPO-specific contract
# ===========================================================================


def test_grpo_has_no_value_head_attribute():
    """Goal: pin contract — GRPO does NOT have a value head. Introducing
    one would change advantage computation semantics (GRPO uses
    group-relative normalization, not GAE).
    """
    trainer = _make_grpo()
    assert not hasattr(trainer, "_value_head")
    assert not hasattr(trainer, "_use_value_head")


def test_grpo_stop_requested_breaks_outer_loop():
    """Goal: ``trainer._stop_requested = True`` from any callback breaks the
    outer loop early.
    """

    class _Stopper:
        def __init__(self):
            self.trainer = None
        def on_reward_computed(self, **_):
            self.trainer._stop_requested = True

    stopper = _Stopper()
    trainer = _make_grpo(callbacks=[stopper], max_steps=10)
    stopper.trainer = trainer
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0]))

    trainer.fit()

    assert trainer.ctx.step == 1
