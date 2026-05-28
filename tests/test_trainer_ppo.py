"""PPOTrainer smoke tests (M6)."""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.trainers.ppo import PPOTrainer


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
                "labels": torch.randint(0, V, (B, T)),
            }


class _FakeEngine:
    mixed_precision = "no"


def _dummy_reward_fn(response_ids, batch):
    B = response_ids.shape[0] if isinstance(response_ids, torch.Tensor) else len(response_ids)
    return [0.5] * B


def _make_ppo(**kw) -> PPOTrainer:
    V = 16
    model = _TinyLM(V=V)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return PPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDataModule(),
        optimizer=opt,
        model=model,
        max_steps=2,
        rollout_steps=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_dummy_reward_fn,
        **kw,
    )


def test_ppo_registers():
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "ppo") is PPOTrainer


def test_ppo_step_returns_finite_loss():
    trainer = _make_ppo()
    V, T, B = 16, 4, 2
    # Manually call _ppo_step with a crafted mini-batch
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),   # (B,) — expanded to (B, T) inside
    }
    metrics = trainer._ppo_step(batch)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(float(metrics["loss"])))


def test_ppo_ref_policy_frozen_after_freeze():
    trainer = _make_ppo()
    trainer._ref_policy = __import__("lighttrain.rl.ref_policy", fromlist=["freeze_as_ref"]).freeze_as_ref(trainer.model)
    for p in trainer._ref_policy.model.parameters():
        assert not p.requires_grad


def test_ppo_target_kl_early_stop_attribute():
    trainer = _make_ppo(target_kl=0.01)
    assert trainer.target_kl == 0.01


def test_ppo_clip_eps_propagated():
    trainer = _make_ppo(clip_eps=0.3)
    assert trainer._loss_fn.clip_eps == 0.3


# ---- callback wiring fix (bug fix verification) ----------------------------

def test_ppo_step_fires_full_callback_chain():
    """_ppo_step must dispatch the full standard event chain including missing items."""
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

    trainer = _make_ppo(callbacks=[_Recorder()])
    V, T, B = 16, 4, 2
    batch = {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }
    trainer._ppo_step(batch)

    for event in [
        "on_step_begin", "on_backward_pre", "on_backward_post",
        "on_clip_grad", "on_optimizer_step_pre", "on_optimizer_step_post",
        "on_zero_grad", "on_step_end",
    ]:
        assert event in fired, f"{event} not fired by _ppo_step"
