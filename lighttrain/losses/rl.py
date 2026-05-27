"""RL losses — PPOSurrogateLoss / GRPOLoss.

Batch contract
--------------
Both losses read pre-computed tensors injected by the RL trainer into
``ctx.extras`` before the loss call:

    log_probs_new    (B, T)  — per-token log-probs under current policy
    log_probs_old    (B, T)  — per-token log-probs from rollout collection
    advantages       (B, T)  — GAE or group-normalized advantages
    attention_mask   (B, T)  — 1 for real tokens, 0 for padding

PPOSurrogateLoss additionally needs:

    values           (B, T)  — value-head estimates
    returns          (B, T)  — GAE returns (advantages + values_old)

GRPOLoss additionally needs:

    group_ids        (B,)    — integer group index per sample (for normalization)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..protocols import LossContext, ModelOutput
from ..registry import register


def _require_extra(ctx: LossContext, key: str, caller: str) -> torch.Tensor:
    val = ctx.extras.get(key)
    if val is None:
        raise KeyError(
            f"{caller} needs ctx.extras['{key}']. "
            "Did the RL trainer inject RL tensors before calling loss?"
        )
    return val


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    x = x * mask.to(x.dtype)
    return x.sum() / mask.sum().clamp_min(1).to(x.dtype)


# ---------------------------------------------------------------------------
# PPO Surrogate
# ---------------------------------------------------------------------------


@register("loss", "ppo_surrogate")
class PPOSurrogateLoss:
    """Clipped PPO surrogate objective (Schulman et al., 2017).

    Policy loss:
        ratio   = exp(log_probs_new - log_probs_old)
        surr1   = ratio * advantages
        surr2   = clip(ratio, 1-ε, 1+ε) * advantages
        L_π     = -mean(min(surr1, surr2))

    Value loss (optional if ``vf_coef > 0``):
        L_V = clip(Δv²) where Δv = values - returns

    Entropy bonus:
        L_H = -mean(entropy)

    Total = L_π + vf_coef * L_V - ent_coef * H
    """

    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        vf_clip_range: float | None = None,
    ) -> None:
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.vf_clip_range = float(vf_clip_range) if vf_clip_range is not None else None

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,
    ) -> dict[str, Any]:
        log_probs_new = _require_extra(ctx, "log_probs_new", "PPOSurrogateLoss")
        log_probs_old = _require_extra(ctx, "log_probs_old", "PPOSurrogateLoss")
        advantages = _require_extra(ctx, "advantages", "PPOSurrogateLoss")
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()

        # Broadcast (B,) advantages to (B, T) if needed
        if advantages.dim() == 1 and log_probs_new.dim() == 2:
            advantages = advantages.unsqueeze(1).expand_as(log_probs_new)

        # Policy loss
        log_ratio = log_probs_new - log_probs_old
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -_masked_mean(torch.min(surr1, surr2), mask)

        # Approx KL for monitoring
        approx_kl = _masked_mean((log_probs_old - log_probs_new).detach(), mask)
        clip_frac = _masked_mean(
            ((ratio - 1.0).abs() > self.clip_eps).float().detach(), mask
        )

        # Value loss
        value_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.vf_coef > 0 and "values" in ctx.extras and "returns" in ctx.extras:
            values = ctx.extras["values"]
            returns = ctx.extras["returns"]
            values_old = ctx.extras.get("values_old", values.detach())
            vf_loss_unclipped = (values - returns).pow(2)
            if self.vf_clip_range is not None:
                values_clipped = values_old + (values - values_old).clamp(
                    -self.vf_clip_range, self.vf_clip_range
                )
                vf_loss_clipped = (values_clipped - returns).pow(2)
                vf_loss_unclipped = torch.max(vf_loss_unclipped, vf_loss_clipped)
            value_loss = _masked_mean(vf_loss_unclipped, mask)

        # Entropy bonus (approximate via -log_probs mean if full probs unavailable)
        entropy_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.ent_coef > 0:
            entropy_loss = _masked_mean(-log_probs_new.detach(), mask)

        total = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        return {
            "loss": total,
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_loss.detach()),
            "approx_kl": float(approx_kl.detach()),
            "clip_frac": float(clip_frac.detach()),
        }


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


@register("loss", "grpo")
class GRPOLoss:
    """Group Relative Policy Optimization (Shao et al., 2024).

    Normalizes advantages within each group (G responses per prompt), removes
    the value model, and applies a clipped surrogate:

        advantages_norm = (advantages - group_mean) / (group_std + ε)
        ratio           = exp(log_probs_new - log_probs_old)
        L_GRPO          = -mean(clip(ratio, 1-ε, 1+ε) * advantages_norm)
                        + β_kl * KL(π_θ || π_ref)

    ``group_ids`` (B,) must be in ctx.extras to identify which samples belong
    to the same prompt group. For a batch of B=G*N with G groups of N responses
    each, ``group_ids`` = [0,0,...,0, 1,1,...,1, ...].
    """

    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        beta_kl: float = 0.0,
        advantage_eps: float = 1e-6,
    ) -> None:
        self.clip_eps = float(clip_eps)
        self.beta_kl = float(beta_kl)
        self.advantage_eps = float(advantage_eps)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,
    ) -> dict[str, Any]:
        log_probs_new = _require_extra(ctx, "log_probs_new", "GRPOLoss")
        log_probs_old = _require_extra(ctx, "log_probs_old", "GRPOLoss")
        advantages_raw = _require_extra(ctx, "advantages", "GRPOLoss")
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()

        # Sequence-level mean log-probs for advantage normalization per group
        seq_logps_new = _masked_mean(log_probs_new, mask)  # (B,) if looped, scalar here
        # Per-token advantages (broadcast from sequence level if needed)
        advantages = advantages_raw
        if advantages.dim() == 1 and log_probs_new.dim() == 2:
            advantages = advantages.unsqueeze(1).expand_as(log_probs_new)

        # Group-level advantage normalization
        group_ids = ctx.extras.get("group_ids")
        if group_ids is not None and group_ids.numel() > 1:
            advantages = self._normalize_by_group(advantages, group_ids)

        log_ratio = log_probs_new - log_probs_old
        ratio = log_ratio.exp()
        surr = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -_masked_mean(surr, mask)

        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.beta_kl > 0 and "log_probs_ref" in ctx.extras:
            log_probs_ref = ctx.extras["log_probs_ref"]
            kl = _masked_mean(log_probs_new - log_probs_ref, mask)
            kl_loss = self.beta_kl * kl

        total = policy_loss + kl_loss
        return {
            "loss": total,
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl_loss.detach()),
            "ratio_mean": float(ratio.mean().detach()),
        }

    def _normalize_by_group(
        self, advantages: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        out = advantages.clone()
        for gid in group_ids.unique():
            sel = group_ids == gid
            adv_g = advantages[sel]
            mean = adv_g.mean()
            std = adv_g.std(unbiased=False).clamp_min(self.advantage_eps)
            out[sel] = (adv_g - mean) / std
        return out


__all__ = [
    "GRPOLoss",
    "PPOSurrogateLoss",
]
