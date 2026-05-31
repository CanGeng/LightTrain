"""Pluggable value / reward heads (registry category ``value_head``).

A single parameterised linear head subsumes the two formerly-duplicated
``LinearValueHead`` definitions in ppo.py and rm.py:

  * PPO critic — per-token value V(s_t): ``bias=True, zero_init=True,
    reduction="per_token"`` → (B, T).
  * Reward model — sequence reward from the last token: ``bias=False,
    zero_init=False, reduction="last"`` → (B,).

Resolve it from a recipe ``value_head:`` block; each trainer defaults to its
own historical config so behaviour is unchanged when the block is omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register


@register("value_head", "linear")
class LinearValueHead(nn.Module):
    """Linear projection of hidden states to a scalar value/reward."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bias: bool = True,
        zero_init: bool = False,
        reduction: str = "per_token",
    ) -> None:
        super().__init__()
        if reduction not in ("per_token", "last"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.reduction = reduction
        self.linear = nn.Linear(int(hidden_size), 1, bias=bias)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states`` (B, T, H) → (B, T) per-token, or (B,) for ``last``."""
        if self.reduction == "last":
            hidden_states = hidden_states[:, -1, :]
        return self.linear(hidden_states).squeeze(-1)


__all__ = ["LinearValueHead"]
