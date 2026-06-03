"""Reachability tests for the hardcoding-audit fixes (F1–F5).

Each finding turned a written-in value into a configurable seam; these tests
assert the new knob is *reachable* and that omitting it preserves old behaviour.
The bit-identity of the defaults is guarded by the existing golden tests
(tests/trainers/test_rl_loss_seam_bitcheck.py); here we check the seams are open.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import lighttrain.rl.reward_adapters  # noqa: F401 — registers reward_adapter/pointwise
import lighttrain.rl.value_heads  # noqa: F401 — registers value_head/linear
from lighttrain.protocols import ModelOutput


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        return ModelOutput(outputs={"logits": self.proj(self.emb(input_ids))})


class _FakeEngine:
    pass


class _FakeDM:
    def train_loader(self):
        while True:
            yield {"input_ids": torch.zeros(2, 4, dtype=torch.long)}


def _reward(ids, batch):
    n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
    return [1.0] * n


# --------------------------------------------------------------------------- #
# F1 — rollout backend resolved via registry + temperature/top_p reachable
# --------------------------------------------------------------------------- #

def _make_ppo(**over):
    from lighttrain.trainers.ppo import PPOTrainer
    m = _TinyLM()
    return PPOTrainer(engine=_FakeEngine(), data_module=_FakeDM(),
                      optimizer=torch.optim.SGD(m.parameters(), lr=1e-2), model=m,
                      max_steps=1, rollout_steps=2, ppo_epochs=1, mini_batch_size=2,
                      max_new_tokens=4, reward_fn=_reward, **over)


def _make_grpo(**over):
    from lighttrain.trainers.grpo import GRPOTrainer
    m = _TinyLM()
    return GRPOTrainer(engine=_FakeEngine(), data_module=_FakeDM(),
                       optimizer=torch.optim.SGD(m.parameters(), lr=1e-2), model=m,
                       max_steps=1, group_size=2, ppo_epochs=1, mini_batch_size=2,
                       max_new_tokens=4, reward_fn=_reward, **over)


def test_f1_sampling_knobs_reach_backend():
    from lighttrain.rl.rollout import HFGenerateBackend
    for make in (_make_ppo, _make_grpo):
        t = make(temperature=0.7, top_p=0.9, do_sample=True)
        be = t._rollout_engine.backend
        assert isinstance(be, HFGenerateBackend)          # resolved via rl_backend registry
        assert be.temperature == pytest.approx(0.7)
        assert be.top_p == pytest.approx(0.9)


def test_f1_defaults_reproduce_old_backend():
    for make in (_make_ppo, _make_grpo):
        be = make()._rollout_engine.backend
        assert be.temperature == 1.0 and be.top_p == 1.0 and be.do_sample is True


# --------------------------------------------------------------------------- #
# F2 — judge->reward via registrable adapter (pointwise); no isinstance whitelist
# --------------------------------------------------------------------------- #

def test_f2_pointwise_adapter_matches_old_reward_fn():
    from lighttrain.builtin_plugins.judges.judge import VerifierJudge
    from lighttrain.registry import get as _get

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    judge = VerifierJudge(verify_fn=lambda p, r: 1.0)
    adapter = _get("reward_adapter", "pointwise")(judge=judge, tokenizer=_Tok())
    pid = torch.zeros(3, 2, dtype=torch.long)
    rid = torch.zeros(3, 2, dtype=torch.long)
    # same as old inline _reward_fn: judge.score(zip(prompts, responses))
    assert adapter(pid, rid) == judge.score(list(zip(["0 0"] * 3, ["0 0"] * 3)))


def test_f2_custom_pointwise_judge_no_longer_rejected():
    # A judge that is NOT VerifierJudge but declares reward_kind="pointwise"
    # would previously hit the isinstance whitelist; now it resolves an adapter.
    from lighttrain.registry import get as _get

    class _MyJudge:
        reward_kind = "pointwise"
        def score(self, items, ctx=None):
            return [0.5 for _ in items]

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x"

    adapter = _get("reward_adapter", _MyJudge.reward_kind)(judge=_MyJudge(), tokenizer=_Tok())
    assert adapter(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)) == [0.5, 0.5]


def test_f2_pairwise_adapter_is_deferred_seam_open_not_implemented():
    """pairwise reward is a deliberately deferred new feature: the seam (registry
    category) is open and pointwise ships, but no `pairwise` adapter is
    registered, so resolving one raises (clean missing-registration, not a
    hardcoded whitelist). Also: PairwiseLLMJudge declares reward_kind='pairwise'."""
    from lighttrain.builtin_plugins.judges.judge import PairwiseLLMJudge
    from lighttrain.registry import get as _get, list_entries
    from lighttrain.registry._exceptions import NotRegisteredError

    assert "pointwise" in list_entries("reward_adapter")
    assert "pairwise" not in list_entries("reward_adapter")  # deferred
    assert getattr(PairwiseLLMJudge, "reward_kind", None) == "pairwise"
    with pytest.raises(NotRegisteredError):
        _get("reward_adapter", "pairwise")


# --------------------------------------------------------------------------- #
# F3 — rm grad_clip is a knob (default 0.0 preserves no-clip)
# --------------------------------------------------------------------------- #

def test_f3_rm_grad_clip_knob():
    from lighttrain.trainers.rm import RewardModelTrainer
    m = _TinyLM()
    base = dict(engine=_FakeEngine(), data_module=_FakeDM(),
                optimizer=torch.optim.SGD(m.parameters(), lr=1e-2), model=m, max_steps=1)
    # default is now 1.0 (matches ppo/grpo/preference); legacy no-clip via 0.0
    assert RewardModelTrainer(**base).grad_clip == 1.0
    assert RewardModelTrainer(**{**base, "grad_clip": 0.0}).grad_clip == 0.0


# --------------------------------------------------------------------------- #
# F4 — value_head pluggable; ppo zero-init vs rm default-init preserved
# --------------------------------------------------------------------------- #

def test_f4_value_head_registered_and_resolvable():
    from lighttrain.registry import get as _get
    head = _get("value_head", "linear")(8, bias=True, zero_init=True, reduction="last")
    assert head.reduction == "last"


def test_f4_ppo_default_head_is_zero_init():
    from lighttrain.rl.value_heads import LinearValueHead
    h = LinearValueHead(8, bias=True, zero_init=True, reduction="per_token")
    assert torch.allclose(h.linear.weight, torch.zeros_like(h.linear.weight))
    assert torch.allclose(h.linear.bias, torch.zeros_like(h.linear.bias))


def test_f4_rm_default_head_is_default_init_not_zero():
    # The watch-point: rm's head must keep DEFAULT init (not zero) and no bias.
    from lighttrain.rl.value_heads import LinearValueHead
    torch.manual_seed(0)
    h = LinearValueHead(8, bias=False, zero_init=False, reduction="last")
    assert h.linear.bias is None
    assert not torch.allclose(h.linear.weight, torch.zeros_like(h.linear.weight))


# --------------------------------------------------------------------------- #
# F5 — RolloutBuffer max_size is a knob
# --------------------------------------------------------------------------- #

def test_f5_buffer_max_size_knob():
    assert _make_ppo(buffer_max_size=16)._buffer.max_size == 16
    assert _make_grpo(buffer_max_size=16)._buffer.max_size == 16
    # defaults preserve the old literals (_make_ppo uses rollout_steps=2)
    assert _make_ppo()._buffer.max_size == 2 * 4
    assert _make_grpo()._buffer.max_size == 2048
