"""RewardModelTrainer tests (M6)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.trainers.rm import LinearValueHead, RewardModelTrainer


# ---- Minimal helpers ------------------------------------------------------

class _FakeCfg:
    hidden_size = 8


class _TinyBackbone(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.config = _FakeCfg()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)                     # (B, T, D)
        logits = self.proj(h)
        return ModelOutput(
            outputs={"logits": logits},
            hidden_states=[h],                      # list of (B, T, D)
        )


class _FakeDataModule:
    def train_loader(self):
        B, T, V = 2, 5, 16
        def _gen():
            while True:
                yield {
                    "chosen_input_ids": torch.randint(0, V, (B, T)),
                    "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
                    "rejected_input_ids": torch.randint(0, V, (B, T)),
                    "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
                }
        return _gen()


class _FakeEngine:
    mixed_precision = "no"

    def to_device(self, x):
        return x


def _make_trainer(**kw) -> RewardModelTrainer:
    model = _TinyBackbone()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return RewardModelTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDataModule(),
        optimizer=optim,
        model=model,
        max_steps=5,
        **kw,
    )


# ---- Tests ----------------------------------------------------------------

def test_linear_value_head_shape():
    vhead = LinearValueHead(hidden_size=8)
    h = torch.randn(3, 6, 8)
    out = vhead(h)
    assert out.shape == (3,)


def test_linear_value_head_no_bias():
    vhead = LinearValueHead(hidden_size=8)
    assert vhead.linear.bias is None


def test_rm_trainer_registers():
    from lighttrain.registry import get as resolve
    cls = resolve("trainer", "reward_model")
    assert cls is RewardModelTrainer


def test_rm_reward_step_returns_loss():
    trainer = _make_trainer()
    B, T, V = 2, 5, 16
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    metrics = trainer._reward_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] > 0.0


def test_rm_preference_step_backward_compat():
    """_preference_step alias still callable (backward compat)."""
    trainer = _make_trainer()
    B, T, V = 2, 5, 16
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    metrics = trainer._preference_step(batch)
    assert "loss" in metrics


def test_rm_value_head_auto_detected():
    trainer = _make_trainer()
    B, T, V = 2, 5, 16
    trainer._score(torch.randint(0, V, (B, T)), torch.ones(B, T, dtype=torch.long))
    assert trainer._value_head is not None
    assert trainer._value_head.linear.in_features == 8


def test_rm_margin_shifts_loss():
    trainer_no = _make_trainer(margin=0.0)
    trainer_mg = _make_trainer(margin=5.0)
    B, T, V = 2, 5, 16
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    torch.manual_seed(7)
    m_no = trainer_no._reward_step(batch)["loss"]
    torch.manual_seed(7)
    m_mg = trainer_mg._reward_step(batch)["loss"]
    assert m_mg >= m_no - 1e-5   # larger margin → harder task → ≥ loss
