"""Generalized Advantage Estimation (GAE, Schulman et al. 2016).

Computes per-step advantages and returns given rewards and value estimates
from a rollout buffer.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float | torch.Tensor = 0.0,
    dones: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Parameters
    ----------
    rewards : (B, T) or (T,)
        Per-step reward signals. For episode-level rewards, broadcast to
        all response tokens before calling.
    values : (B, T) or (T,)
        Value-head estimates V(s_t).
    gamma : float
        Discount factor.
    lam : float
        GAE lambda (bias-variance trade-off; 0 = TD(0), 1 = full MC).
    last_value : float or (B,) tensor
        Bootstrap value V(s_{T+1}). Zero for terminal episodes.
    dones : (B, T) or (T,) boolean tensor, optional
        ``True`` at the last step of an episode (masks the bootstrap).

    Returns
    -------
    advantages : same shape as ``rewards``
    returns : same shape as ``rewards``
        ``returns = advantages + values`` (targets for the value function).
    """
    squeeze = rewards.dim() == 1
    if squeeze:
        rewards = rewards.unsqueeze(0)
        values = values.unsqueeze(0)
        if dones is not None:
            dones = dones.unsqueeze(0)

    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    if isinstance(last_value, torch.Tensor):
        lv = last_value.to(device=device, dtype=dtype)
        if lv.dim() == 0:
            lv = lv.expand(B)
    else:
        lv = torch.full((B,), float(last_value), device=device, dtype=dtype)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=device, dtype=dtype)

    for t in reversed(range(T)):
        next_val = lv if t == T - 1 else values[:, t + 1]
        # Mask next_val to 0 at episode boundaries.
        if dones is not None:
            done_mask = dones[:, t].float()
            next_val = next_val * (1.0 - done_mask)
            last_gae = last_gae * (1.0 - done_mask)

        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values

    if squeeze:
        advantages = advantages.squeeze(0)
        returns = returns.squeeze(0)

    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    *,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standardize advantages to zero mean, unit variance.

    Parameters
    ----------
    advantages : arbitrary shape
    eps : numerical stability epsilon
    mask : optional boolean mask; only masked-in positions contribute to
           mean/std estimation.
    """
    if mask is not None:
        flat = advantages[mask.bool()]
    else:
        flat = advantages.flatten()
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(eps)
    return (advantages - mean) / std


__all__ = ["compute_gae", "normalize_advantages"]
