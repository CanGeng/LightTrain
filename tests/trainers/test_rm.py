"""Tests for RewardModelTrainer (relocated from tests/test_trainer_rm.py and
the RM smoke layer of tests/test_trainer_step_protocol.py).

RewardModelTrainer is an inline trainer (consumes_objective=False): it computes
its own Bradley-Terry pairwise loss over chosen/rejected hidden states via a
LinearValueHead, rather than going through the loss: seam.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.trainers.rm import LinearValueHead, RewardModelTrainer
from lighttrain.protocols import ModelOutput, StepOutput


class _FakeCfg:
    hidden_size = 8


class _TinyBackbone(nn.Module):
    """Backbone that returns hidden_states (needed by RewardModelTrainer)."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.config = _FakeCfg()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)}, hidden_states=[h])


class _FakeEngine:
    mixed_precision = "no"

    def to_device(self, x):
        return x


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


def _make_rm(**kw) -> RewardModelTrainer:
    model = _TinyBackbone()
    return RewardModelTrainer(
        engine=_FakeEngine(),
        data_module=_RmDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=5,
        **kw,
    )


# ===========================================================================
# LinearValueHead
# ===========================================================================


def test_linear_value_head_last_token_reduces_to_per_sequence():
    """last-token reduction maps (B, T, D) hidden states to a (B,) scalar score."""
    vhead = LinearValueHead(hidden_size=8, bias=False, reduction="last")
    out = vhead(torch.randn(3, 6, 8))
    assert out.shape == (3,)


def test_linear_value_head_no_bias_when_disabled():
    vhead = LinearValueHead(hidden_size=8, bias=False, reduction="last")
    assert vhead.linear.bias is None


# ===========================================================================
# Registry
# ===========================================================================


def test_reward_model_resolves_from_registry():
    from lighttrain.registry import get as resolve

    assert resolve("trainer", "reward_model") is RewardModelTrainer


# ===========================================================================
# _reward_step
# ===========================================================================


def test_rm_reward_step_returns_positive_loss():
    metrics = _make_rm()._reward_step(_rm_batch())
    assert "loss" in metrics
    assert metrics["loss"] > 0.0


def test_rm_reward_step_returns_reward_metric_keys():
    result = _make_rm()._reward_step(_rm_batch())
    assert {"reward_chosen", "reward_rejected", "reward_margin"} <= result.keys()


def test_rm_has_reward_step_method():
    assert hasattr(RewardModelTrainer, "_reward_step")


def test_rm_preference_step_aliases_reward_step():
    """_preference_step alias routes to _reward_step (backward compat) — proven
    by the presence of RM-specific reward keys (the base preference path would
    not produce them)."""
    result = _make_rm()._preference_step(_rm_batch())
    assert "reward_chosen" in result


def test_rm_value_head_auto_detected_from_hidden_size():
    trainer = _make_rm()
    B, T, V = 2, 5, 16
    trainer._score(torch.randint(0, V, (B, T)), torch.ones(B, T, dtype=torch.long))
    assert trainer._value_head is not None
    assert trainer._value_head.linear.in_features == 8


def test_rm_larger_margin_does_not_lower_loss():
    """A larger margin makes the separation task harder ⇒ loss must not drop."""
    trainer_no = _make_rm(margin=0.0)
    trainer_mg = _make_rm(margin=5.0)
    batch = _rm_batch()
    torch.manual_seed(7)
    m_no = trainer_no._reward_step(batch)["loss"]
    torch.manual_seed(7)
    m_mg = trainer_mg._reward_step(batch)["loss"]
    assert m_mg >= m_no - 1e-5


# ===========================================================================
# fit + train_step (inline objective)
# ===========================================================================


def test_rm_fit_runs_without_recipe_loss():
    """Regression: reward_model is inline (consumes_objective=False); its fit must
    NOT be blocked by the inherited preference 'no loss configured' guard — it
    computes its own Bradley-Terry loss and the runtime leaves ctx.loss_fn=None."""
    trainer = _make_rm()
    assert trainer.consumes_objective is False
    assert trainer.ctx.loss_fn is None
    metrics = trainer.fit(steps=2)
    assert "loss" in metrics


def test_rm_train_step_returns_reward_metrics_stepoutput():
    out = _make_rm().train_step(_rm_batch())
    assert isinstance(out, StepOutput)
    assert "reward_chosen" in out.metrics
    assert "reward_rejected" in out.metrics
    assert "reward_margin" in out.metrics
