"""Keystone step-3 checks for GRPO / PPO:

1. Bit-check: the RLUpdateRule->apply_update backward unification + loss-seam
   refactor leave the per-step update numerics EXACTLY unchanged (golden
   sequences captured from the pre-migration code).
2. Objective seam: the canonical source of the per-step loss is now
   ``trainer.objective`` (the runtime binds it; RL trainers fall back to their
   own ``default_objective()`` surrogate). A recipe-provided objective drives
   the step; with none, the RL default surrogate is used. (No more
   ``ctx.loss_fn`` type-sniffing / cross_entropy sentinel.)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.trainers.grpo import GRPOTrainer
from lighttrain.builtin_plugins.trainers.ppo import PPOTrainer
from lighttrain.optim.architectures.profile import LossOnlyObjective
from lighttrain.protocols import ModelOutput

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
    # tolerance (not 8-decimal exact): golden captured on one platform; a
    # different CPU/BLAS drifts ~1e-7, while a real change is >>1e-5.
    assert got == pytest.approx(GRPO_GOLDEN, rel=1e-5, abs=1e-7)


def test_ppo_update_path_bit_identical():
    torch.manual_seed(5)
    t = _make_ppo(_TinyLM())
    got = [round(float(t._step(_ppo_batch()).loss), 8) for _ in range(5)]
    assert got == pytest.approx(PPO_GOLDEN, rel=1e-5, abs=1e-7)


class _SentinelLoss:
    """A loss that ignores inputs and returns a fixed value — proves it was
    actually invoked (i.e. the recipe's ctx.loss_fn drives the step)."""

    def __call__(self, model_output, batch, ctx):
        return {"loss": torch.tensor(42.0, requires_grad=True)}


@pytest.mark.parametrize("make", [_make_grpo, _make_ppo], ids=["grpo", "ppo"])
def test_recipe_objective_drives_rl_step(make):
    """Gate 2: a recipe-provided ``trainer.objective`` (the canonical seam) must
    drive the RL step — bound by the runtime, here set directly."""
    t = make(_TinyLM())
    t.objective = LossOnlyObjective(_SentinelLoss())
    batch = _grpo_batch() if make is _make_grpo else _ppo_batch()
    out = t._step(batch)
    assert float(out.loss) == 42.0


@pytest.mark.parametrize("make", [_make_grpo, _make_ppo], ids=["grpo", "ppo"])
def test_rl_default_objective_used_when_unbound(make):
    """Fallback: with no objective bound (``trainer.objective is None``), the RL
    trainer resolves its own ``default_objective()`` surrogate — no CE sniffing."""
    t = make(_TinyLM())
    assert t.objective is None  # not wired by the runtime here
    # default_objective wraps the trainer's own surrogate loss.
    assert isinstance(t.default_objective(), LossOnlyObjective)
    batch = _grpo_batch() if make is _make_grpo else _ppo_batch()
    out = t._step(batch)
    assert float(out.loss) != 42.0
    assert torch.isfinite(torch.tensor(float(out.loss)))
    # the step resolved + bound the default objective on first use.
    assert isinstance(t.objective, LossOnlyObjective)
