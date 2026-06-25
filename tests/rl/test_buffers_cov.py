"""Edge-case coverage for lighttrain.builtin_plugins.rl.buffers.

Lines driven to covered (previously uncovered):
  77  — is_empty() True/False branches
  146 — any(ep.values is not None ...) branch inside _collate
  151 — batch["values_old"] assignment when some episodes have values
  154 — batch["advantages_buf"] = advantages[batch_idx]
  156 — batch["returns_buf"] = returns[batch_idx]
  168 — all_values() early return: empty buffer or first episode has values=None
  169 — all_values() None-values branch (first episode has no values)
  170 — max_len computation
  171 — loop over episodes in all_values()
  172 — v = ep.values if ep.values is not None else zeros
  173 — diff = max_len - v.size(0)
  174/175/176 — diff > 0 padding branch
  177 — out.append(v)
  178 — torch.stack(out)
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.rl.buffers import Episode, RolloutBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ep(
    input_len: int,
    reward: float,
    *,
    group_id: int = 0,
    with_values: bool = False,
    values_len: int | None = None,
) -> Episode:
    """Build a minimal Episode.  ``with_values`` attaches a values tensor."""
    vlen = values_len if values_len is not None else input_len
    return Episode(
        input_ids=torch.arange(input_len),
        attention_mask=torch.ones(input_len),
        labels=torch.full((input_len,), -100),
        reward=reward,
        log_probs=torch.zeros(input_len),
        values=torch.full((vlen,), float(reward)) if with_values else None,
        group_id=group_id,
    )


# ---------------------------------------------------------------------------
# is_empty (line 77)
# ---------------------------------------------------------------------------

def test_invariant_is_empty_true_on_fresh_buffer():
    """A freshly created buffer with no episodes reports is_empty() == True."""
    buf = RolloutBuffer(max_size=10)
    assert buf.is_empty() is True


def test_invariant_is_empty_false_after_add():
    """After adding one episode, is_empty() must return False."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    assert buf.is_empty() is False


def test_invariant_is_empty_true_after_clear():
    """is_empty() reverts to True after clear()."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.clear()
    assert buf.is_empty() is True


# ---------------------------------------------------------------------------
# _collate: values_old branch (lines 145-151)
# ---------------------------------------------------------------------------

def test_invariant_values_old_present_when_any_episode_has_values():
    """When at least one episode carries values, _collate must add 'values_old'."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0, with_values=True))
    buf.add(_ep(4, 2.0, with_values=True))
    batches = list(buf.batches(batch_size=2, shuffle=False))
    assert len(batches) == 1
    b = batches[0]
    assert "values_old" in b
    assert b["values_old"].shape == (2, 4)


def test_invariant_values_old_absent_when_no_episode_has_values():
    """When no episode has values, 'values_old' must not appear in the batch."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.add(_ep(4, 2.0))
    batches = list(buf.batches(batch_size=2, shuffle=False))
    assert "values_old" not in batches[0]


def test_invariant_values_old_mixed_some_none_episodes():
    """If only some episodes have values, None episodes become zero rows in values_old."""
    buf = RolloutBuffer(max_size=10)
    # episode 0 has values; episode 1 does not
    buf.add(_ep(4, 1.0, with_values=True))
    buf.add(_ep(4, 2.0, with_values=False))
    batches = list(buf.batches(batch_size=2, shuffle=False))
    b = batches[0]
    assert "values_old" in b
    # None episode should be represented as a zero row
    # shape: (2, 4)
    assert b["values_old"].shape == (2, 4)


def test_invariant_values_old_padded_to_max_len():
    """Episodes with different lengths get values_old padded correctly."""
    torch.manual_seed(42)
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(3, 1.0, with_values=True))  # shorter
    buf.add(_ep(5, 2.0, with_values=True))  # longer
    batches = list(buf.batches(batch_size=2, shuffle=False))
    b = batches[0]
    assert "values_old" in b
    assert b["values_old"].shape == (2, 5)
    # Padded positions (index 3 and 4 for first episode) should be zeros
    torch.testing.assert_close(
        b["values_old"][0, 3:], torch.zeros(2), atol=1e-6, rtol=0
    )


# ---------------------------------------------------------------------------
# _collate: advantages_buf (line 153-154)
# ---------------------------------------------------------------------------

def test_invariant_advantages_buf_sliced_correctly():
    """advantages[batch_idx] lands in 'advantages_buf' with the right values."""
    torch.manual_seed(0)
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.add(_ep(4, 2.0))
    buf.add(_ep(4, 3.0))
    advantages = torch.tensor([10.0, 20.0, 30.0])
    batches = list(buf.batches(batch_size=3, shuffle=False, advantages=advantages))
    b = batches[0]
    assert "advantages_buf" in b
    assert b["advantages_buf"].shape == (3,)
    # With shuffle=False, indices are [0,1,2] → values are [10,20,30]
    torch.testing.assert_close(
        b["advantages_buf"], torch.tensor([10.0, 20.0, 30.0]), atol=1e-6, rtol=0
    )


def test_invariant_advantages_buf_absent_when_none():
    """When advantages=None, 'advantages_buf' must not appear in the batch."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    batches = list(buf.batches(batch_size=1, shuffle=False, advantages=None))
    assert "advantages_buf" not in batches[0]


# ---------------------------------------------------------------------------
# _collate: returns_buf (line 155-156)
# ---------------------------------------------------------------------------

def test_invariant_returns_buf_sliced_correctly():
    """returns[batch_idx] lands in 'returns_buf' with the right values."""
    torch.manual_seed(0)
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.add(_ep(4, 2.0))
    returns = torch.tensor([100.0, 200.0])
    batches = list(buf.batches(batch_size=2, shuffle=False, returns=returns))
    b = batches[0]
    assert "returns_buf" in b
    torch.testing.assert_close(
        b["returns_buf"], torch.tensor([100.0, 200.0]), atol=1e-6, rtol=0
    )


def test_invariant_returns_buf_absent_when_none():
    """When returns=None, 'returns_buf' must not appear in the batch."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    batches = list(buf.batches(batch_size=1, shuffle=False, returns=None))
    assert "returns_buf" not in batches[0]


def test_invariant_advantages_and_returns_both_present():
    """Both advantages_buf and returns_buf appear when both tensors are provided."""
    torch.manual_seed(0)
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0))
    buf.add(_ep(4, 2.0))
    adv = torch.tensor([1.5, 2.5])
    ret = torch.tensor([3.5, 4.5])
    batches = list(buf.batches(batch_size=2, shuffle=False, advantages=adv, returns=ret))
    b = batches[0]
    assert "advantages_buf" in b
    assert "returns_buf" in b


# ---------------------------------------------------------------------------
# all_values (lines 168-178)
# ---------------------------------------------------------------------------

def test_invariant_all_values_returns_none_for_empty_buffer():
    """all_values() on an empty buffer returns None (line 168 early-return)."""
    buf = RolloutBuffer(max_size=10)
    assert buf.all_values() is None


def test_invariant_all_values_returns_none_when_first_episode_has_no_values():
    """all_values() returns None if the first episode carries values=None (line 168-169)."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0, with_values=False))
    buf.add(_ep(4, 2.0, with_values=False))
    assert buf.all_values() is None


def test_invariant_all_values_stacks_uniform_length_episodes():
    """all_values() returns (N, T) tensor when all episodes have equal-length values."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(4, 1.0, with_values=True))
    buf.add(_ep(4, 2.0, with_values=True))
    result = buf.all_values()
    assert result is not None
    assert result.shape == (2, 4)
    # First episode reward = 1.0 → values tensor is filled with 1.0
    torch.testing.assert_close(result[0], torch.ones(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[1], torch.full((4,), 2.0), atol=1e-6, rtol=0)


def test_invariant_all_values_pads_shorter_episodes():
    """all_values() pads shorter episodes with zeros to reach max_len (lines 174-176)."""
    buf = RolloutBuffer(max_size=10)
    buf.add(_ep(3, 1.0, with_values=True, values_len=3))
    buf.add(_ep(5, 2.0, with_values=True, values_len=5))
    result = buf.all_values()
    assert result is not None
    assert result.shape == (2, 5)
    # First episode values len=3, padded with zeros at positions 3 and 4
    torch.testing.assert_close(
        result[0, 3:], torch.zeros(2), atol=1e-6, rtol=0
    )
    # Second episode fully filled with 2.0
    torch.testing.assert_close(result[1], torch.full((5,), 2.0), atol=1e-6, rtol=0)


def test_invariant_all_values_no_padding_needed_for_equal_length():
    """When all episodes have the same length, diff == 0 branch in all_values is taken."""
    buf = RolloutBuffer(max_size=10)
    for r in [0.5, 1.5, 2.5]:
        buf.add(_ep(4, r, with_values=True, values_len=4))
    result = buf.all_values()
    assert result is not None
    assert result.shape == (3, 4)


def test_pin_current_behavior_all_values_uses_first_episode_values_check():
    """Pin: all_values() inspects only the FIRST episode's values field to decide
    whether to return None or compute the stack.

    NOTE: this is a debatable design choice — if later episodes have values but
    the first does not, all_values() returns None and silently ignores the rest.
    This test pins the current implemented behavior so any change is visible.
    """
    buf = RolloutBuffer(max_size=10)
    # First episode has no values; second does.
    buf.add(_ep(4, 1.0, with_values=False))
    buf.add(_ep(4, 2.0, with_values=True))
    # Current code: checks only _episodes[0].values — returns None
    result = buf.all_values()
    assert result is None


# ---------------------------------------------------------------------------
# Batches with shuffle=True (exercises randperm path + ensures advantages/returns
# are correctly indexed even when order is shuffled)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 99])
def test_invariant_shuffled_advantages_match_episode_order(seed: int):
    """With shuffle=True, advantages_buf[i] must correspond to the episode at
    batch position i — i.e. indexing with the permuted batch_idx is correct."""
    torch.manual_seed(seed)
    n = 5
    rewards = [float(i) for i in range(n)]
    advantages = torch.arange(n, dtype=torch.float)
    buf = RolloutBuffer(max_size=20)
    for r in rewards:
        buf.add(_ep(4, r))
    # Single batch of all episodes (shuffle permutes the order)
    batches = list(buf.batches(batch_size=n, shuffle=True, advantages=advantages))
    b = batches[0]
    # The rewards and advantages must be aligned: for each position i,
    # advantages_buf[i] == rewards[i]
    for i in range(n):
        assert b["advantages_buf"][i].item() == pytest.approx(b["rewards"][i].item())


# ---------------------------------------------------------------------------
# Edge cases for multi-batch iteration
# ---------------------------------------------------------------------------

def test_invariant_multi_batch_advantages_are_non_overlapping():
    """With multiple mini-batches, each element from advantages appears exactly once."""
    torch.manual_seed(7)
    n = 6
    buf = RolloutBuffer(max_size=20)
    for i in range(n):
        buf.add(_ep(3, float(i)))
    advantages = torch.arange(n, dtype=torch.float)
    collected = []
    for b in buf.batches(batch_size=2, shuffle=False, advantages=advantages):
        collected.append(b["advantages_buf"])
    all_adv = torch.cat(collected)
    assert all_adv.shape == (n,)
    # All values from 0..5 appear (no duplicates, no gaps)
    torch.testing.assert_close(all_adv.sort().values, advantages, atol=1e-6, rtol=0)
