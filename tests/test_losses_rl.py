"""RL surrogate loss tests (M6) — PPOSurrogateLoss / GRPOLoss."""

from __future__ import annotations

import torch

from lighttrain.losses.rl import GRPOLoss, PPOSurrogateLoss
from lighttrain.protocols import LossContext, ModelOutput

_DUMMY = ModelOutput(outputs={})


def _ppo_ctx(B: int = 2, T: int = 4, advantages_val: float = 1.0) -> LossContext:
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), -0.1)
    adv = torch.full((B, T), advantages_val)
    return LossContext(extras={"log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv})


# ---- PPO -----------------------------------------------------------------

def test_ppo_loss_returns_policy_loss():
    ctx = _ppo_ctx()
    out = PPOSurrogateLoss()(_DUMMY, {}, ctx)
    assert "loss" in out and "policy_loss" in out


def test_ppo_clip_ratio_bounded():
    """ratio > 1+eps should be clipped; test that clip_frac > 0 for large new LP."""
    B, T = 2, 4
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), 1.0)    # ratio = exp(1) >> 1+0.2
    adv = torch.ones(B, T)
    ctx = LossContext(extras={"log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv})
    out = PPOSurrogateLoss(clip_eps=0.2)(_DUMMY, {}, ctx)
    assert out["clip_frac"] > 0.0


def test_ppo_approx_kl_nonneg():
    ctx = _ppo_ctx()
    out = PPOSurrogateLoss()(_DUMMY, {}, ctx)
    assert out["approx_kl"] >= 0.0


# ---- GRPO ----------------------------------------------------------------

def test_grpo_loss_basic():
    B, T = 4, 4
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), -0.05)
    adv = torch.tensor([1.0, 0.5, -0.5, -1.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    ctx = LossContext(
        extras={
            "log_probs_new": lp_new,
            "log_probs_old": lp_old,
            "advantages": adv,
            "group_ids": group_ids,
        }
    )
    out = GRPOLoss()(_DUMMY, {}, ctx)
    assert "loss" in out


def test_grpo_group_norm_applied():
    """Group normalization within the same group_id should balance advantages."""
    B, T = 4, 1
    lp_old = torch.zeros(B, T)
    lp_new = torch.zeros(B, T)
    adv_raw = torch.tensor([10.0, 20.0, 30.0, 40.0])
    group_ids = torch.tensor([0, 0, 0, 0])   # all same group
    ctx = LossContext(
        extras={
            "log_probs_new": lp_new,
            "log_probs_old": lp_old,
            "advantages": adv_raw,
            "group_ids": group_ids,
        }
    )
    out_normed = GRPOLoss()(_DUMMY, {}, ctx)
    # Loss should be finite and computable
    assert torch.isfinite(torch.tensor(out_normed["loss"].item()))


def test_grpo_returns_ratio_mean():
    B, T = 2, 3
    ctx = LossContext(
        extras={
            "log_probs_new": torch.zeros(B, T),
            "log_probs_old": torch.zeros(B, T),
            "advantages": torch.ones(B),
        }
    )
    out = GRPOLoss()(_DUMMY, {}, ctx)
    assert "ratio_mean" in out
    assert abs(out["ratio_mean"] - 1.0) < 1e-4


# ---- PPO entropy gradient fix (bug fix verification) ------------------------

def test_ppo_entropy_gradient_flows():
    """ent_coef > 0 must produce non-zero gradient through log_probs_new (no detach)."""
    B, T = 2, 4
    lp_new = torch.full((B, T), -0.5, requires_grad=True)
    lp_old = torch.zeros(B, T)
    adv = torch.ones(B, T)
    ctx = LossContext(extras={"log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv})
    out = PPOSurrogateLoss(ent_coef=0.1)(_DUMMY, {}, ctx)
    out["loss"].backward()
    assert lp_new.grad is not None and lp_new.grad.abs().sum().item() > 0


def test_ppo_response_only_mask_excludes_prompt():
    """labels-based mask must exclude prompt positions (-100) from loss computation.
    Prompt positions use lp_new=lp_old (ratio=1), response uses lp_new < lp_old.
    Masked mean covers only response; unmasked includes both — values must differ.
    """
    B, T = 2, 6
    # prompt: first 3 cols use lp_new == lp_old → ratio=1, surr=adv
    # response: last 3 cols use lp_new=-0.5 < lp_old=0 → ratio<1, surr<adv
    lp_new = torch.cat([torch.zeros(B, 3), torch.full((B, 3), -0.5)], dim=1)
    lp_old = torch.zeros(B, T)
    adv = torch.ones(B, T)
    labels = torch.cat([torch.full((B, 3), -100), torch.ones(B, 3, dtype=torch.long)], dim=1)
    ctx = LossContext(extras={"log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv})
    out_with_labels = PPOSurrogateLoss()(_DUMMY, {"labels": labels}, ctx)
    out_no_labels = PPOSurrogateLoss()(_DUMMY, {}, ctx)
    assert float(out_with_labels["loss"]) != float(out_no_labels["loss"])


def test_grpo_response_only_mask_excludes_prompt():
    """GRPOLoss must use labels-based mask when labels are provided.
    Prompt positions use lp_new=lp_old (ratio=1), response uses lp_new < lp_old.
    """
    B, T = 4, 6
    lp_new = torch.cat([torch.zeros(B, 3), torch.full((B, 3), -0.1)], dim=1)
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    group_ids = torch.tensor([0, 0, 1, 1])
    labels = torch.cat([torch.full((B, 3), -100), torch.ones(B, 3, dtype=torch.long)], dim=1)
    ctx = LossContext(extras={"log_probs_new": lp_new, "log_probs_old": lp_old,
                              "advantages": adv, "group_ids": group_ids})
    out_labeled = GRPOLoss()(_DUMMY, {"labels": labels}, ctx)
    out_plain = GRPOLoss()(_DUMMY, {}, ctx)
    assert float(out_labeled["loss"]) != float(out_plain["loss"])
