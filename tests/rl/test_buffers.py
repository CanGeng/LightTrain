"""Adversarial tests for lighttrain.rl.buffers (Episode / RolloutBuffer)."""

from __future__ import annotations

import torch

from lighttrain.rl.buffers import Episode, RolloutBuffer


def _ep(input_len: int, reward: float, group_id: int = 0) -> Episode:
    return Episode(
        input_ids=torch.arange(input_len),
        attention_mask=torch.ones(input_len),
        labels=torch.full((input_len,), -100),
        reward=reward,
        log_probs=torch.zeros(input_len),
        group_id=group_id,
    )


def test_buffer_round_trip_preserves_episode_data():
    """Goal: pushing N episodes and reading them back gives the exact input.

    Input: 3 episodes with distinct rewards.
    Analytical: buf.all_rewards() must equal the original reward list.
    """
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.add(_ep(4, 2.5))
    buf.add(_ep(4, -1.0))
    torch.testing.assert_close(
        buf.all_rewards(), torch.tensor([1.0, 2.5, -1.0]), atol=1e-6, rtol=1e-5
    )
    assert len(buf) == 3


def test_buffer_overflow_drops_oldest_fifo():
    """Goal: at capacity, oldest episode is evicted (FIFO).

    Input: max_size=2, push 3 episodes.
    Analytical: rewards retained should be [2nd, 3rd], not [1st, 2nd].
    """
    buf = RolloutBuffer(max_size=2)
    buf.add(_ep(3, 1.0))
    buf.add(_ep(3, 2.0))
    buf.add(_ep(3, 3.0))
    assert len(buf) == 2
    torch.testing.assert_close(
        buf.all_rewards(), torch.tensor([2.0, 3.0]), atol=1e-6, rtol=1e-5
    )


def test_buffer_batches_pads_and_yields_correct_size():
    """Goal: ``buf.batches(batch_size=B)`` yields padded mini-batches whose
            input_ids tensor has shape (B, max_seq_in_batch) — and the values
            at non-pad positions exactly match the episode's input_ids.
    """
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))  # input_ids = [0,1,2,3]
    buf.add(_ep(5, 2.0))  # input_ids = [0,1,2,3,4]
    batches = list(buf.batches(batch_size=2, shuffle=False))
    assert len(batches) == 1
    b = batches[0]
    # batch_size=2 episodes, max_len = 5 → shape (2, 5)
    assert b["input_ids"].shape == (2, 5)
    # First episode (len=4) gets padded with 0 at index 4.
    torch.testing.assert_close(
        b["input_ids"][0], torch.tensor([0, 1, 2, 3, 0]), atol=0, rtol=0
    )
    # Second episode unchanged.
    torch.testing.assert_close(
        b["input_ids"][1], torch.tensor([0, 1, 2, 3, 4]), atol=0, rtol=0
    )
    # rewards correctly assembled from the two episodes
    torch.testing.assert_close(
        b["rewards"], torch.tensor([1.0, 2.0]), atol=1e-6, rtol=1e-5
    )
