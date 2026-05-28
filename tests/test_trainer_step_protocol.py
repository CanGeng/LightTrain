"""Trainer Step Protocol tests (Task 1).

Three layers:
  1. Protocol normalization — DummyTrainer stubs, no real model.
  2. Interface coverage — hasattr checks over all 9 concrete trainer classes.
  3. Smoke tests — one minimal end-to-end call per trainer family.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput, StepOutput
from lighttrain.trainers.base import Trainer
from lighttrain.trainers.dpo import DPOTrainer
from lighttrain.trainers.grpo import GRPOTrainer
from lighttrain.trainers.ipo import IPOTrainer
from lighttrain.trainers.kto import KTOTrainer
from lighttrain.trainers.orpo import ORPOTrainer
from lighttrain.trainers.ppo import PPOTrainer
from lighttrain.trainers.pretrain import PretrainTrainer
from lighttrain.trainers.rm import RewardModelTrainer
from lighttrain.trainers.simpo import SimPOTrainer


# ---------------------------------------------------------------------------
# Layer 1: Protocol normalization tests (DummyTrainers, no real model)
# ---------------------------------------------------------------------------

class _FakeEngine:
    mixed_precision = "no"


class _FakeDM:
    def train_loader(self):
        while True:
            yield {}


def _base_kwargs():
    return dict(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=MagicMock(),
    )


class _DummyDictTrainer(Trainer):
    """_step returns a plain dict."""
    def fit(self, *, steps=None): ...
    def _step(self, batch):
        return {"loss": 1.0, "acc": 0.5}


class _DummyStepOutputTrainer(Trainer):
    """_step returns a StepOutput directly."""
    def fit(self, *, steps=None): ...
    def _step(self, batch):
        return StepOutput(loss=0.5, metrics={"loss": 0.5, "acc": 0.9})


class _DummyNoLossTrainer(Trainer):
    """_step returns dict without 'loss' key."""
    def fit(self, *, steps=None): ...
    def _step(self, batch):
        return {"acc": 0.5}


class _DummyBadReturnTrainer(Trainer):
    """_step returns an invalid type (int)."""
    def fit(self, *, steps=None): ...
    def _step(self, batch):
        return 42


def test_dict_result_normalized_to_stepoutput():
    trainer = _DummyDictTrainer(**_base_kwargs())
    out = trainer.train_step({})
    assert isinstance(out, StepOutput)


def test_stepoutput_passthrough():
    trainer = _DummyStepOutputTrainer(**_base_kwargs())
    out = trainer.train_step({})
    assert isinstance(out, StepOutput)
    assert out.loss == 0.5


def test_metrics_retains_loss_key():
    trainer = _DummyDictTrainer(**_base_kwargs())
    out = trainer.train_step({})
    assert "loss" in out.metrics
    assert out.metrics["loss"] == 1.0


def test_loss_is_extracted_from_metrics():
    trainer = _DummyDictTrainer(**_base_kwargs())
    out = trainer.train_step({})
    assert out.loss == 1.0


def test_loss_none_when_missing_from_dict():
    trainer = _DummyNoLossTrainer(**_base_kwargs())
    out = trainer.train_step({})
    assert out.loss is None
    assert "acc" in out.metrics


def test_invalid_return_type_raises_typeerror():
    trainer = _DummyBadReturnTrainer(**_base_kwargs())
    with pytest.raises(TypeError, match="_step\\(\\) must return StepOutput or dict"):
        trainer.train_step({})


def test_trainer_base_is_abstract():
    """Trainer cannot be instantiated directly — both fit() and _step() are abstract."""
    with pytest.raises(TypeError, match="abstract"):
        Trainer(**_base_kwargs())  # type: ignore[abstract]


def test_subclass_without_step_cannot_instantiate():
    """Forgetting to implement _step() raises TypeError at instantiation time."""
    class _MissingStep(Trainer):
        def fit(self, *, steps=None): ...
    with pytest.raises(TypeError, match="abstract"):
        _MissingStep(**_base_kwargs())


def test_subclass_without_fit_cannot_instantiate():
    """Forgetting to implement fit() raises TypeError at instantiation time."""
    class _MissingFit(Trainer):
        def _step(self, batch): return {"loss": 0.0}
    with pytest.raises(TypeError, match="abstract"):
        _MissingFit(**_base_kwargs())


# ---------------------------------------------------------------------------
# Layer 2: Interface coverage — parametrized, no step execution needed
# ---------------------------------------------------------------------------

CONCRETE_TRAINERS = [
    PretrainTrainer,
    DPOTrainer,
    IPOTrainer,
    KTOTrainer,
    ORPOTrainer,
    SimPOTrainer,
    RewardModelTrainer,
    PPOTrainer,
    GRPOTrainer,
]


@pytest.mark.parametrize("trainer_cls", CONCRETE_TRAINERS, ids=lambda c: c.__name__)
def test_has_step_method(trainer_cls):
    assert hasattr(trainer_cls, "_step"), f"{trainer_cls.__name__} missing _step"


@pytest.mark.parametrize("trainer_cls", CONCRETE_TRAINERS, ids=lambda c: c.__name__)
def test_has_train_step_method(trainer_cls):
    assert hasattr(trainer_cls, "train_step"), f"{trainer_cls.__name__} missing train_step"


@pytest.mark.parametrize("trainer_cls", CONCRETE_TRAINERS, ids=lambda c: c.__name__)
def test_step_is_not_base_stub(trainer_cls):
    assert trainer_cls._step is not Trainer._step, (
        f"{trainer_cls.__name__}._step is still the base stub — add a concrete override"
    )


def test_all_concrete_trainers_covered():
    assert len(CONCRETE_TRAINERS) == 9, (
        "CONCRETE_TRAINERS list is out of date — update it when adding a new trainer"
    )


@pytest.mark.parametrize("trainer_cls", CONCRETE_TRAINERS, ids=lambda c: c.__name__)
def test_concrete_trainers_instantiable(trainer_cls):
    """Regression guard: all concrete trainers can be instantiated after ABC change."""
    # Build minimal kwargs; RL trainers need extra fields
    kw: dict = dict(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=MagicMock(),
        max_steps=1,
    )
    model = _TinyLM()
    if trainer_cls in (DPOTrainer, IPOTrainer, KTOTrainer, ORPOTrainer, SimPOTrainer):
        kw["model"] = model
    elif trainer_cls is RewardModelTrainer:
        kw["model"] = _TinyBackbone()
    elif trainer_cls is PretrainTrainer:
        kw["model"] = model
    elif trainer_cls is PPOTrainer:
        kw["model"] = model
        kw["reward_fn"] = lambda ids, b: [0.5] * (ids.shape[0] if hasattr(ids, "shape") else len(ids))
    elif trainer_cls is GRPOTrainer:
        kw["model"] = model
        kw["reward_fn"] = lambda ids, b: [0.5] * (ids.shape[0] if hasattr(ids, "shape") else len(ids))
    instance = trainer_cls(**kw)
    assert instance is not None


# ---------------------------------------------------------------------------
# Layer 3: Smoke tests — one per trainer family
# ---------------------------------------------------------------------------

# ---- shared tiny model ----

class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _TinyBackbone(nn.Module):
    """Backbone that returns hidden_states (needed by RewardModelTrainer)."""
    class _Cfg:
        hidden_size = 8

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.config = self._Cfg()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(
            outputs={"logits": self.proj(h)},
            hidden_states=[h],
        )


# ---- PretrainTrainer: delegates to mocked engine ----

def test_pretrain_train_step_delegates_to_engine():
    mock_engine = MagicMock()
    mock_engine.mixed_precision = "no"
    mock_engine.step.return_value = {"loss": 1.23, "ppl": 3.4}

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
    out = trainer.train_step(batch)

    mock_engine.step.assert_called_once()
    assert isinstance(out, StepOutput)
    assert out.loss == 1.23
    assert "ppl" in out.metrics


# ---- PreferenceTrainer family (DPOTrainer representative) ----

class _PrefDM:
    def train_loader(self):
        V, T, B = 16, 5, 2
        while True:
            yield {
                "chosen_input_ids": torch.randint(0, V, (B, T)),
                "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
                "chosen_labels": torch.randint(0, V, (B, T)),
                "rejected_input_ids": torch.randint(0, V, (B, T)),
                "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
                "rejected_labels": torch.randint(0, V, (B, T)),
            }


def _pref_batch(V: int = 16, T: int = 5, B: int = 2) -> dict:
    return {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "chosen_labels": torch.randint(0, V, (B, T)),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_labels": torch.randint(0, V, (B, T)),
        # DPO / IPO / KTO need reference log-probs from artifact store
        "aux.ref.chosen_logprobs": torch.randn(B) - 1.0,
        "aux.ref.rejected_logprobs": torch.randn(B) - 2.0,
    }


def test_preference_family_train_step_returns_stepoutput():
    V = 16
    model = _TinyLM(V=V)
    trainer = DPOTrainer(
        engine=_FakeEngine(),
        data_module=_PrefDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        beta=0.1,
    )
    out = trainer.train_step(_pref_batch(V=V))
    assert isinstance(out, StepOutput)
    assert out.loss is not None
    assert "loss" in out.metrics
    assert math.isfinite(float(out.loss))


# ---- RewardModelTrainer smoke test ----

class _RmDM:
    def train_loader(self):
        B, T, V = 2, 5, 16
        while True:
            yield {
                "chosen_input_ids": torch.randint(0, V, (B, T)),
                "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
                "rejected_input_ids": torch.randint(0, V, (B, T)),
                "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
            }


def _rm_batch(V: int = 16, T: int = 5, B: int = 2) -> dict:
    return {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }


def _make_rm_trainer():
    model = _TinyBackbone()
    return RewardModelTrainer(
        engine=_FakeEngine(),
        data_module=_RmDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )


def test_rm_train_step_returns_reward_metrics():
    trainer = _make_rm_trainer()
    out = trainer.train_step(_rm_batch())
    assert isinstance(out, StepOutput)
    assert "reward_chosen" in out.metrics
    assert "reward_rejected" in out.metrics
    assert "reward_margin" in out.metrics


def test_rm_has_reward_step_method():
    assert hasattr(RewardModelTrainer, "_reward_step")


def test_rm_reward_step_returns_reward_keys():
    trainer = _make_rm_trainer()
    result = trainer._reward_step(_rm_batch())
    assert {"reward_chosen", "reward_rejected", "reward_margin"} <= result.keys()


def test_rm_preference_step_backward_compat():
    """_preference_step alias routes to _reward_step (backward compat)."""
    trainer = _make_rm_trainer()
    result = trainer._preference_step(_rm_batch())
    assert "reward_chosen" in result  # proves it reached _reward_step, not base


# ---- Backward compat: old method names still callable ----

def test_preference_step_still_callable():
    V = 16
    model = _TinyLM(V=V)
    trainer = DPOTrainer(
        engine=_FakeEngine(),
        data_module=_PrefDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        beta=0.1,
    )
    result = trainer._preference_step(_pref_batch(V=V))
    assert isinstance(result, dict)
    assert "loss" in result


def test_ppo_step_still_callable():
    V, T, B = 16, 4, 2
    model = _TinyLM(V=V)

    def _reward_fn(ids, batch):
        n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
        return [0.5] * n

    trainer = PPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        rollout_steps=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_reward_fn,
    )
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }
    result = trainer._ppo_step(batch)
    assert isinstance(result, dict)
    assert "loss" in result


def test_grpo_step_still_callable():
    V, T, B = 16, 4, 2
    model = _TinyLM(V=V)

    def _reward_fn(ids, batch):
        n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
        return [0.5] * n

    trainer = GRPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        max_new_tokens=4,
        reward_fn=_reward_fn,
    )
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "group_ids": torch.zeros(B, dtype=torch.long),
        "rewards": torch.ones(B),
    }
    result = trainer._grpo_step(batch)
    assert isinstance(result, dict)
    assert "loss" in result
