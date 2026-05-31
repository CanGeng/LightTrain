"""GRPOTrainer tests (M6)."""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.trainers.grpo import GRPOTrainer


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _FakeDataModule:
    def train_loader(self):
        V, T, B = 16, 4, 2
        while True:
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }


class _FakeEngine:
    mixed_precision = "no"


def _dummy_reward(ids, batch):
    B = ids.shape[0] if isinstance(ids, torch.Tensor) else 1
    return [1.0] * B


def _make_grpo(**kw) -> GRPOTrainer:
    V = 16
    model = _TinyLM(V=V)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return GRPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDataModule(),
        optimizer=opt,
        model=model,
        max_steps=2,
        group_size=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_dummy_reward,
        **kw,
    )


def test_grpo_registers():
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "grpo") is GRPOTrainer


def test_grpo_step_finite_loss():
    trainer = _make_grpo()
    V, T, B = 16, 4, 4
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.tensor([1.0, 0.5, 0.5, -0.5]),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }
    metrics = trainer._grpo_step(batch)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(float(metrics["loss"])))


def test_grpo_group_size_stored():
    trainer = _make_grpo()   # group_size=2 set in _make_grpo
    assert trainer.group_size == 2


def test_grpo_clip_eps_propagated():
    # clip_eps now feeds the default RL loss (used when the recipe omits a
    # `loss:` block); the loss: seam wins when present (keystone step 3).
    trainer = _make_grpo(clip_eps=0.15)
    assert trainer._default_loss.clip_eps == 0.15


# ---- callback wiring fix (bug fix verification) ----------------------------

def test_grpo_step_fires_full_callback_chain():
    """_grpo_step must dispatch the complete standard event chain."""
    fired = []

    class _Recorder:
        def on_step_begin(self, **kw): fired.append("on_step_begin")
        def on_backward_pre(self, **kw): fired.append("on_backward_pre")
        def on_backward_post(self, **kw): fired.append("on_backward_post")
        def on_clip_grad(self, **kw): fired.append("on_clip_grad")
        def on_optimizer_step_pre(self, **kw): fired.append("on_optimizer_step_pre")
        def on_optimizer_step_post(self, **kw): fired.append("on_optimizer_step_post")
        def on_zero_grad(self, **kw): fired.append("on_zero_grad")
        def on_step_end(self, **kw): fired.append("on_step_end")

    trainer = _make_grpo(callbacks=[_Recorder()])
    V, T, B = 16, 4, 4
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.tensor([1.0, 0.5, 0.5, -0.5]),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }
    trainer._grpo_step(batch)

    for event in [
        "on_step_begin", "on_backward_pre", "on_backward_post",
        "on_clip_grad", "on_optimizer_step_pre", "on_optimizer_step_post",
        "on_zero_grad", "on_step_end",
    ]:
        assert event in fired, f"{event} not fired by _grpo_step"
