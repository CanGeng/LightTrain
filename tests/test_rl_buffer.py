"""RolloutBuffer tests (M6)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.rl.buffers import Episode, RolloutBuffer


def _ep(reward: float = 1.0, T: int = 4, group_id: int = 0) -> Episode:
    return Episode(
        input_ids=torch.zeros(T, dtype=torch.long),
        attention_mask=torch.ones(T, dtype=torch.long),
        labels=torch.zeros(T, dtype=torch.long),
        reward=reward,
        log_probs=torch.zeros(T),
        values=torch.zeros(T),
        group_id=group_id,
    )


def test_buffer_add_and_len():
    buf = RolloutBuffer(max_size=16)
    for _ in range(5):
        buf.add(_ep())
    assert len(buf) == 5


def test_buffer_clear():
    buf = RolloutBuffer()
    buf.add(_ep())
    buf.clear()
    assert len(buf) == 0


def test_buffer_all_rewards():
    buf = RolloutBuffer()
    buf.add(_ep(reward=1.0))
    buf.add(_ep(reward=2.0))
    rewards = buf.all_rewards()
    assert rewards.shape == (2,)
    assert float(rewards.sum()) == pytest.approx(3.0)


def test_buffer_batches_yields_batches():
    buf = RolloutBuffer()
    for i in range(6):
        buf.add(_ep(reward=float(i)))
    batches = list(buf.batches(batch_size=3, shuffle=False))
    assert len(batches) == 2
    for b in batches:
        assert "input_ids" in b
        assert b["input_ids"].shape[0] == 3


import pytest  # noqa: E402
