"""Rollout buffer for on-policy RL training.

The buffer accumulates episodes collected by :class:`~lighttrain.builtin_plugins.rl.rollout.RolloutEngine`
and exposes them as mini-batches for PPO/GRPO inner epochs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Episode:
    """Single rollout episode (one prompt → one response).

    Attributes
    ----------
    input_ids : (T,) — prompt + response token ids
    attention_mask : (T,) — 1 for real tokens
    labels : (T,) — response tokens; prompt positions = -100
    reward : scalar float
    log_probs : (T,) — per-token log-probs under the policy at collection time
    values : (T,) or None — value-head estimates (PPO only)
    group_id : int — which prompt group this response belongs to (GRPO)
    extras : dict — arbitrary metadata (e.g. prompt text, response text)
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    reward: float
    log_probs: torch.Tensor
    values: torch.Tensor | None = None
    group_id: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class RolloutBuffer:
    """Stores :class:`Episode` objects collected during rollout and serves mini-batches.

    Usage::

        buf = RolloutBuffer(max_size=1024)
        for ep in rollout_engine.rollout(...):
            buf.add(ep)
        for batch in buf.batches(batch_size=32):
            # train on batch
            ...
        buf.clear()
    """

    def __init__(self, max_size: int = 4096) -> None:
        self.max_size = int(max_size)
        self._episodes: list[Episode] = []

    # ------------------------------------------------------------------ mutate

    def add(self, episode: Episode) -> None:
        if len(self._episodes) >= self.max_size:
            # Drop oldest when full.
            self._episodes.pop(0)
        self._episodes.append(episode)

    def clear(self) -> None:
        self._episodes.clear()

    # ------------------------------------------------------------------ query

    def __len__(self) -> int:
        return len(self._episodes)

    def is_empty(self) -> bool:
        return len(self._episodes) == 0

    # ------------------------------------------------------------------ batch

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        advantages: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield padded mini-batches from the buffer.

        Parameters
        ----------
        batch_size :
            Number of episodes per mini-batch.
        shuffle :
            Randomly permute episodes before batching.
        advantages : (N,) optional pre-computed advantages (one per episode).
        returns : (N,) optional pre-computed returns.
        """
        n = len(self._episodes)
        indices = torch.randperm(n) if shuffle else torch.arange(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size].tolist()
            episodes = [self._episodes[i] for i in batch_idx]
            yield self._collate(
                episodes,
                batch_idx=batch_idx,
                advantages=advantages,
                returns=returns,
            )

    def _collate(
        self,
        episodes: list[Episode],
        *,
        batch_idx: list[int],
        advantages: torch.Tensor | None,
        returns: torch.Tensor | None,
    ) -> dict[str, Any]:
        max_len = max(ep.input_ids.size(0) for ep in episodes)

        def _pad(t: torch.Tensor, pad_val: float, target_len: int) -> torch.Tensor:
            diff = target_len - t.size(0)
            if diff == 0:
                return t
            return torch.cat([t, torch.full((diff,), pad_val, dtype=t.dtype, device=t.device)])

        input_ids = torch.stack([_pad(ep.input_ids, 0, max_len) for ep in episodes])
        attention_mask = torch.stack([_pad(ep.attention_mask, 0, max_len) for ep in episodes])
        labels = torch.stack([_pad(ep.labels, -100, max_len) for ep in episodes])
        log_probs_old = torch.stack([_pad(ep.log_probs, 0.0, max_len) for ep in episodes])
        rewards = torch.tensor([ep.reward for ep in episodes])
        group_ids = torch.tensor([ep.group_id for ep in episodes])

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "log_probs_old": log_probs_old,
            "rewards": rewards,
            "group_ids": group_ids,
        }

        if any(ep.values is not None for ep in episodes):
            values_list = [
                _pad(ep.values, 0.0, max_len) if ep.values is not None
                else torch.zeros(max_len)
                for ep in episodes
            ]
            batch["values_old"] = torch.stack(values_list)

        if advantages is not None:
            batch["advantages_buf"] = advantages[batch_idx]
        if returns is not None:
            batch["returns_buf"] = returns[batch_idx]

        return batch

    # ------------------------------------------------------------------ tensor views

    def all_rewards(self) -> torch.Tensor:
        """Return (N,) tensor of all episode rewards."""
        return torch.tensor([ep.reward for ep in self._episodes])

    def all_values(self) -> torch.Tensor | None:
        """Return (N, T_max) padded values or None if no value estimates."""
        if not self._episodes or self._episodes[0].values is None:
            return None
        max_len = max(ep.values.size(0) for ep in self._episodes if ep.values is not None)
        out = []
        for ep in self._episodes:
            v = ep.values if ep.values is not None else torch.zeros(max_len)
            diff = max_len - v.size(0)
            if diff > 0:
                v = torch.cat([v, torch.zeros(diff)])
            out.append(v)
        return torch.stack(out)


__all__ = ["Episode", "RolloutBuffer"]
