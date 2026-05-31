"""BatchValidationError tests (Task 5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from lighttrain.exceptions import BatchValidationError, LightTrainError
from lighttrain.losses.preference import DPOLoss
from lighttrain.protocols import ModelOutput
from lighttrain.trainers._preference_base import PreferenceTrainer
from lighttrain.trainers.grpo import GRPOTrainer
from lighttrain.trainers.ppo import PPOTrainer
from lighttrain.trainers.rm import RewardModelTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _FakeEngine:
    mixed_precision = "no"


class _FakeDM:
    def train_loader(self):
        while True:
            yield {}


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _TinyBackbone(nn.Module):
    class _Cfg:
        hidden_size = 8

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.config = self._Cfg()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)}, hidden_states=[h])


def _make_dpo_trainer():
    model = _TinyLM()
    trainer = PreferenceTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )
    trainer.ctx.loss_fn = DPOLoss(beta=0.1)
    return trainer


def _make_rm_trainer():
    model = _TinyBackbone()
    return RewardModelTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )


def _make_ppo_trainer():
    model = _TinyLM()
    return PPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        rollout_steps=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=lambda ids, b: [0.5] * (ids.shape[0] if hasattr(ids, "shape") else len(ids)),
    )


def _make_grpo_trainer():
    model = _TinyLM()
    return GRPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        max_new_tokens=4,
        reward_fn=lambda ids, b: [0.5] * (ids.shape[0] if hasattr(ids, "shape") else len(ids)),
    )


# ---------------------------------------------------------------------------
# BatchValidationError class tests
# ---------------------------------------------------------------------------

def test_batch_validation_error_is_lighttrain_error():
    err = BatchValidationError("TestTrainer", ["key_a"], ["key_b", "key_c"])
    assert isinstance(err, LightTrainError)


def test_error_message_includes_trainer_name():
    err = BatchValidationError("PPOTrainer", ["advantages_buf"], ["input_ids"])
    assert "PPOTrainer" in str(err)


def test_error_message_includes_missing_key():
    err = BatchValidationError("PPOTrainer", ["advantages_buf"], ["input_ids"])
    assert "advantages_buf" in str(err)


def test_error_message_caps_present_keys():
    present = [f"key_{i}" for i in range(20)]
    err = BatchValidationError("TestTrainer", ["missing"], present)
    msg = str(err)
    # Should mention the overflow count
    assert "more" in msg


def test_valid_batch_passes_without_error():
    from lighttrain.trainers._utils import validate_batch
    validate_batch({"input_ids": 1, "log_probs_old": 2, "advantages_buf": 3},
                   ["input_ids", "log_probs_old", "advantages_buf"], "PPOTrainer")


# ---------------------------------------------------------------------------
# Per-trainer missing-key tests
# ---------------------------------------------------------------------------

def test_preference_missing_chosen_input_ids_raises():
    trainer = _make_dpo_trainer()
    B, T, V = 2, 5, 16
    batch = {
        # "chosen_input_ids" intentionally omitted
        "chosen_labels": torch.randint(0, V, (B, T)),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_labels": torch.randint(0, V, (B, T)),
        "aux.ref.chosen_logprobs": torch.randn(B),
        "aux.ref.rejected_logprobs": torch.randn(B),
    }
    with pytest.raises(BatchValidationError, match="PreferenceTrainer"):
        trainer._preference_step(batch)


def test_preference_missing_rejected_labels_raises():
    trainer = _make_dpo_trainer()
    B, T, V = 2, 5, 16
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_labels": torch.randint(0, V, (B, T)),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        # "rejected_labels" intentionally omitted
        "aux.ref.chosen_logprobs": torch.randn(B),
        "aux.ref.rejected_logprobs": torch.randn(B),
    }
    with pytest.raises(BatchValidationError, match="PreferenceTrainer"):
        trainer._preference_step(batch)


def test_rm_missing_rejected_input_ids_raises():
    trainer = _make_rm_trainer()
    B, T, V = 2, 5, 16
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        # "rejected_input_ids" intentionally omitted
    }
    with pytest.raises(BatchValidationError, match="RewardModelTrainer"):
        trainer._reward_step(batch)


def test_ppo_missing_advantages_buf_raises():
    trainer = _make_ppo_trainer()
    B, T, V = 2, 4, 16
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        # "advantages_buf" intentionally omitted
    }
    with pytest.raises(BatchValidationError, match="PPOTrainer"):
        trainer._ppo_step(batch)


def test_ppo_missing_input_ids_raises():
    trainer = _make_ppo_trainer()
    B, T = 2, 4
    batch = {
        # "input_ids" intentionally omitted
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }
    with pytest.raises(BatchValidationError, match="PPOTrainer"):
        trainer._ppo_step(batch)


def test_grpo_missing_group_ids_raises():
    trainer = _make_grpo_trainer()
    B, T, V = 2, 4, 16
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.ones(B),
        # "group_ids" intentionally omitted
    }
    with pytest.raises(BatchValidationError, match="GRPOTrainer"):
        trainer._grpo_step(batch)


def test_grpo_missing_rewards_raises():
    trainer = _make_grpo_trainer()
    B, T, V = 2, 4, 16
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "group_ids": torch.zeros(B, dtype=torch.long),
        # "rewards" intentionally omitted
    }
    with pytest.raises(BatchValidationError, match="GRPOTrainer"):
        trainer._grpo_step(batch)
