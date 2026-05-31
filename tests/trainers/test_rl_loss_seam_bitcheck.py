"""Keystone step-3 checks for GRPO / PPO:

1. Bit-check: the RLUpdateRule->apply_update backward unification + loss-seam
   refactor leave the per-step update numerics EXACTLY unchanged (golden
   sequences captured from the pre-migration code).
2. Loss seam: the recipe-provided ``ctx.loss_fn`` is now actually used (it was
   silently overwritten by a hardcoded ``self._loss_fn`` before). A sentinel
   loss set on ctx must drive the step; with none set, the RL default is used.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.trainers.grpo import GRPOTrainer
from lighttrain.trainers.ppo import PPOTrainer

GRPO_GOLDEN = [0.26884612, 0.26883215, 0.26881814, 0.26880413, 0.2687901]
PPO_GOLDEN = [-0.33317095, -0.3332406, -0.33331046, -0.3333804, -0.33345056]


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        return ModelOutput(outputs={"logits": self.proj(self.emb(input_ids))})


class _FakeEngine:
    pass


class _FakeDM:
    def train_loader(self):
        while True:
            yield {"input_ids": torch.zeros(2, 4, dtype=torch.long)}


def _reward(ids, batch):
    n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
    return [1.0] * n


def _grpo_batch(V=16, T=4, B=4):
    g = torch.Generator().manual_seed(42)
    return {
        "input_ids": torch.randint(0, V, (B, T), generator=g),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T), generator=g),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.tensor([1.0, 0.5, 0.5, -0.5]),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }


def _ppo_batch(V=16, T=4, B=2):
    g = torch.Generator().manual_seed(7)
    return {
        "input_ids": torch.randint(0, V, (B, T), generator=g),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T), generator=g),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }


def _make_grpo(model):
    return GRPOTrainer(
        engine=_FakeEngine(), data_module=_FakeDM(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-2), model=model,
        max_steps=5, group_size=2, ppo_epochs=1, mini_batch_size=2,
        max_new_tokens=4, reward_fn=_reward, clip_eps=0.2, beta_kl=0.0,
    )


def _make_ppo(model):
    return PPOTrainer(
        engine=_FakeEngine(), data_module=_FakeDM(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-2), model=model,
        max_steps=5, rollout_steps=2, ppo_epochs=1, mini_batch_size=2,
        max_new_tokens=4, reward_fn=_reward, clip_eps=0.2, vf_coef=0.0, ent_coef=0.01,
    )


def test_grpo_update_path_bit_identical():
    torch.manual_seed(3)
    t = _make_grpo(_TinyLM())
    got = [round(float(t._step(_grpo_batch()).loss), 8) for _ in range(5)]
    assert got == GRPO_GOLDEN


def test_ppo_update_path_bit_identical():
    torch.manual_seed(5)
    t = _make_ppo(_TinyLM())
    got = [round(float(t._step(_ppo_batch()).loss), 8) for _ in range(5)]
    assert got == PPO_GOLDEN


class _SentinelLoss:
    """A loss that ignores inputs and returns a fixed value — proves it was
    actually invoked (i.e. the recipe's ctx.loss_fn drives the step)."""

    def __call__(self, model_output, batch, ctx):
        return {"loss": torch.tensor(42.0, requires_grad=True)}


@pytest.mark.parametrize("make", [_make_grpo, _make_ppo], ids=["grpo", "ppo"])
def test_recipe_loss_fn_actually_selects_rl_loss(make):
    """Gate 2: setting ctx.loss_fn (the `loss:` seam) must drive the RL step —
    previously self._loss_fn was hardcoded and the recipe loss ignored."""
    t = make(_TinyLM())
    t.ctx.loss_fn = _SentinelLoss()
    batch = _grpo_batch() if make is _make_grpo else _ppo_batch()
    out = t._step(batch)
    assert float(out.loss) == 42.0


@pytest.mark.parametrize("make", [_make_grpo, _make_ppo], ids=["grpo", "ppo"])
def test_rl_default_loss_used_when_recipe_omits_loss(make):
    """Fallback: with no recipe loss (ctx.loss_fn None or the cross_entropy
    runtime default), the RL trainer uses its own surrogate default."""
    from lighttrain.losses.core import CrossEntropyLoss

    t = make(_TinyLM())
    batch = _grpo_batch() if make is _make_grpo else _ppo_batch()
    # None → default
    t.ctx.loss_fn = None
    out_none = t._step(batch)
    assert float(out_none.loss) != 42.0 and torch.isfinite(torch.tensor(float(out_none.loss)))
    # cross_entropy sentinel → default (not the misapplied CE)
    t.ctx.loss_fn = CrossEntropyLoss()
    out_ce = t._step(batch)
    assert torch.isfinite(torch.tensor(float(out_ce.loss)))
