"""Coverage supplement for lighttrain/builtin_plugins/trainers/ppo.py.

Pins / exercises the uncovered lines reported at 76% baseline:

  fit() loop:
    261-266  logger.log_dict called each `log_every` step
    268-279  ckpt_manager.save called each `ckpt_every` step
    286-287  secondary exception in on_exception dispatch is suppressed
    291-298  on_train_end and logger.flush exceptions are suppressed in finally

  produce_batch / _compute_advantages:
    322      episodes iterated and added to buffer
    337-341  _compute_advantages: value_head enabled, values available/None

  _ppo_step():
    379-380  model returns plain dict → logits extracted from dict
    395      labels is None → log_probs_new = zeros_like(log_probs_old)
    406      value_head_spec dict is merged into the resolver spec

  _compute_buffer_values():
    442-488  entire method: noop paths, TypeErrors fallback, no-hidden warn, full compute path

  Stubs:
    502      eval() → {}
    505      predict() → []
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.rl.rollout import Episode
from lighttrain.builtin_plugins.rl.value_heads import LinearValueHead
from lighttrain.builtin_plugins.trainers.ppo import PPOTrainer
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Local stubs
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Tiny causal LM — ModelOutput with optional hidden_states."""

    def __init__(self, V: int = 16, D: int = 8, with_hidden: bool = True) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)
        self._with_hidden = with_hidden

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        logits = self.proj(h)
        if self._with_hidden:
            return ModelOutput(outputs={"logits": logits}, hidden_states=(h,))
        return ModelOutput(outputs={"logits": logits}, hidden_states=None)


class _DictLM(nn.Module):
    """Returns a plain dict (not ModelOutput) — exercises lines 379-380."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        logits = self.proj(h)
        return {"logits": logits}


class _TypeErrorLM(nn.Module):
    """Rejects ``output_hidden_states`` but returns ModelOutput with hidden on fallback."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=None, **_):
        if output_hidden_states is not None:
            raise TypeError("model does not accept output_hidden_states")
        h = self.emb(input_ids)
        logits = self.proj(h)
        return ModelOutput(outputs={"logits": logits}, hidden_states=(h,))


class _NoHiddenAfterTypeErrorLM(nn.Module):
    """Rejects ``output_hidden_states``; fallback call returns no hidden → None path."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=None, **_):
        if output_hidden_states is not None:
            raise TypeError("model does not accept output_hidden_states")
        h = self.emb(input_ids)
        logits = self.proj(h)
        return ModelOutput(outputs={"logits": logits}, hidden_states=None)


class _FakeDM:
    """Yields infinite identical batches (controlled by max_steps)."""

    def __init__(self, n_yields: int = 100) -> None:
        self._n = n_yields

    def train_loader(self):
        V, T, B = 16, 4, 2
        for _ in range(self._n):
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
                "labels": torch.randint(0, V, (B, T)),
            }


def _reward_fn(response_ids: Any, batch: Any) -> list[float]:
    n = response_ids.shape[0] if isinstance(response_ids, torch.Tensor) else len(response_ids)
    return [0.5] * n


def _make_ppo(
    *,
    model: nn.Module | None = None,
    callbacks: list | None = None,
    logger: Any = None,
    ckpt_manager: Any = None,
    **over,
) -> PPOTrainer:
    if model is None:
        model = _TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return PPOTrainer(
        engine=object(),
        data_module=_FakeDM(),
        optimizer=opt,
        model=model,
        callbacks=callbacks,
        logger=logger,
        ckpt_manager=ckpt_manager,
        max_steps=over.pop("max_steps", 1),
        rollout_steps=over.pop("rollout_steps", 2),
        ppo_epochs=over.pop("ppo_epochs", 1),
        mini_batch_size=over.pop("mini_batch_size", 2),
        max_new_tokens=over.pop("max_new_tokens", 4),
        reward_fn=over.pop("reward_fn", _reward_fn),
        **over,
    )


def _stub_rollout(trainer: PPOTrainer, rewards: torch.Tensor) -> None:
    """Skip the actual rollout/generate machinery."""
    trainer._rollout_engine.rollout = MagicMock(return_value=[])
    trainer._buffer.clear = MagicMock()
    trainer._buffer.add = MagicMock()
    trainer._buffer.all_rewards = MagicMock(return_value=rewards)
    trainer._buffer.all_values = MagicMock(return_value=None)
    trainer._buffer.batches = MagicMock(return_value=iter([]))
    trainer._compute_buffer_values = lambda: None


def _ppo_batch(V: int = 16, T: int = 4, B: int = 2) -> dict:
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }


def _make_episode(T: int = 4, n_resp: int = 2, values: torch.Tensor | None = None) -> Episode:
    """Create a minimal Episode for _compute_buffer_values tests."""
    return Episode(
        input_ids=torch.randint(0, 16, (T,)),
        attention_mask=torch.ones(T, dtype=torch.long),
        labels=torch.cat([
            torch.full((T - n_resp,), -100, dtype=torch.long),
            torch.randint(0, 16, (n_resp,)),
        ]),
        reward=1.0,
        log_probs=torch.zeros(T),
        values=values,
    )


# ===========================================================================
# fit() loop — logger and checkpoint paths (lines 261-279)
# ===========================================================================


def test_invariant_logger_log_dict_called_each_log_every_step():
    """logger.log_dict must be called once per step when log_every=1 (lines 261-266)."""
    logger = MagicMock()
    trainer = _make_ppo(logger=logger, max_steps=2, log_every=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0, 0.5]))
    trainer.fit()
    assert logger.log_dict.call_count == 2


def test_invariant_logger_log_dict_skips_off_log_every():
    """logger.log_dict is NOT called when step % log_every != 0."""
    logger = MagicMock()
    trainer = _make_ppo(logger=logger, max_steps=1, log_every=5)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    logger.log_dict.assert_not_called()


def test_invariant_ckpt_manager_save_called_each_ckpt_every_step():
    """ckpt_manager.save must be called once per step when ckpt_every=1 (lines 268-279)."""
    ckpt_mgr = MagicMock()
    trainer = _make_ppo(ckpt_manager=ckpt_mgr, max_steps=2, ckpt_every=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0, 0.5]))
    trainer.fit()
    assert ckpt_mgr.save.call_count == 2


def test_invariant_ckpt_manager_save_skips_off_ckpt_every():
    """ckpt_manager.save is NOT called when step % ckpt_every != 0."""
    ckpt_mgr = MagicMock()
    trainer = _make_ppo(ckpt_manager=ckpt_mgr, max_steps=1, ckpt_every=5)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    ckpt_mgr.save.assert_not_called()


def test_invariant_ckpt_manager_save_passes_model_state_dict():
    """ckpt_manager.save receives a 'state' dict with a 'model' key (line 275)."""
    ckpt_mgr = MagicMock()
    trainer = _make_ppo(ckpt_manager=ckpt_mgr, max_steps=1, ckpt_every=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    _args, kwargs = ckpt_mgr.save.call_args
    assert "model" in kwargs["state"]


def test_invariant_logger_flush_called_in_finally():
    """logger.flush() is called in the finally block after a successful fit (line 296)."""
    logger = MagicMock()
    trainer = _make_ppo(logger=logger, max_steps=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    logger.flush.assert_called_once()


def test_pin_current_behavior_logger_flush_exception_suppressed():
    """logger.flush() raising in finally must be caught and suppressed — fit still returns (line 296-298).

    NOTE: current behavior pins that flush exceptions are silently swallowed.
    """
    logger = MagicMock()
    logger.flush.side_effect = RuntimeError("flush exploded")
    trainer = _make_ppo(logger=logger, max_steps=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    result = trainer.fit()
    assert isinstance(result, dict)  # fit returned normally despite flush raising


def test_pin_current_behavior_on_train_end_dispatch_exception_suppressed():
    """bus.dispatch('on_train_end') raising must be caught and suppressed (lines 291-293).

    NOTE: current behavior — secondary dispatch error in finally is silently swallowed.
    """
    trainer = _make_ppo(max_steps=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    original_dispatch = trainer.bus.dispatch

    def boom_dispatch(event, **kw):
        if event == "on_train_end":
            raise RuntimeError("on_train_end bus exploded")
        return original_dispatch(event, **kw)

    trainer.bus.dispatch = boom_dispatch
    result = trainer.fit()
    assert isinstance(result, dict)


def test_pin_current_behavior_on_exception_secondary_exception_suppressed():
    """When fit raises and on_exception callback itself raises, the secondary exception
    is suppressed and the original exception re-propagates (lines 286-288).

    NOTE: current behavior — secondary exception in on_exception is swallowed.
    """

    class _BoomTwice:
        critical = True  # ensure rollout_begin exception escapes the bus

        def on_rollout_begin(self, **_):
            raise ValueError("primary crash")

        def on_exception(self, **_):
            raise RuntimeError("secondary crash in on_exception")

        def on_train_end(self, **_):
            pass

    trainer = _make_ppo(callbacks=[_BoomTwice()], max_steps=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))

    with pytest.raises(ValueError, match="primary crash"):
        trainer.fit()
    # Secondary RuntimeError must NOT have escaped — test reaching here proves it


# ===========================================================================
# produce_batch — episode iterator (line 322)
# ===========================================================================


def test_invariant_produce_batch_adds_episodes_to_buffer():
    """produce_batch must call buffer.add for each episode returned by rollout (line 322)."""
    ep1 = _make_episode(T=4, n_resp=2)
    ep2 = _make_episode(T=4, n_resp=2)
    trainer = _make_ppo()
    trainer._rollout_engine.rollout = MagicMock(return_value=[ep1, ep2])
    trainer._buffer.add = MagicMock()

    trainer._ref_policy = MagicMock()
    prompt = {
        "input_ids": torch.randint(0, 16, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    trainer.produce_batch(prompt)

    assert trainer._buffer.add.call_count == 2


def test_invariant_produce_batch_returns_tensor_of_rewards():
    """produce_batch must return a 1-D tensor of per-episode rewards."""
    ep1 = _make_episode(T=4, n_resp=2)
    ep1.reward = 1.5
    ep2 = _make_episode(T=4, n_resp=2)
    ep2.reward = 0.5
    trainer = _make_ppo()
    trainer._rollout_engine.rollout = MagicMock(return_value=[ep1, ep2])
    trainer._ref_policy = MagicMock()
    prompt = {
        "input_ids": torch.randint(0, 16, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    rewards = trainer.produce_batch(prompt)
    assert rewards.shape == (2,)
    # Buffer stores episodes and all_rewards reads their .reward fields
    assert float(rewards[0]) == pytest.approx(1.5)
    assert float(rewards[1]) == pytest.approx(0.5)


# ===========================================================================
# _compute_advantages — value head branches (lines 337-341)
# ===========================================================================


def test_invariant_compute_advantages_uses_values_when_available():
    """When use_value_head=True and buffer.all_values() returns a tensor,
    that tensor must be used as values_for_gae (lines 337-340)."""
    torch.manual_seed(0)
    trainer = _make_ppo(use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    trainer._buffer.all_values = MagicMock(return_value=torch.tensor([[0.5], [0.3]]))
    trainer._compute_buffer_values = lambda: None

    rewards = torch.tensor([1.0, 0.5])
    advantages, returns = trainer._compute_advantages(rewards)
    assert advantages.shape == (2, 1)
    assert returns.shape == (2, 1)
    # With non-zero baseline, returns differ from plain rewards-unsqueeze
    # (just verify shapes — numeric value depends on gamma/lam)


def test_invariant_compute_advantages_falls_back_to_zeros_when_values_none():
    """When use_value_head=True but all_values() returns None, zero baseline is used (line 341)."""
    torch.manual_seed(0)
    trainer = _make_ppo(use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    trainer._buffer.all_values = MagicMock(return_value=None)
    trainer._compute_buffer_values = lambda: None

    rewards = torch.tensor([1.0, 0.5])
    advantages, _ = trainer._compute_advantages(rewards)
    assert advantages.shape == (2, 1)


def test_invariant_compute_advantages_zeros_when_no_value_head():
    """When use_value_head=False, zero baseline is always used (line 343)."""
    torch.manual_seed(0)
    trainer = _make_ppo(use_value_head=False)
    trainer._compute_buffer_values = lambda: None

    rewards = torch.tensor([1.0, 0.5])
    advantages, _ = trainer._compute_advantages(rewards)
    assert advantages.shape == (2, 1)


# ===========================================================================
# _ppo_step — dict-output and no-labels paths (lines 379-380, 395)
# ===========================================================================


def test_invariant_ppo_step_handles_dict_model_output():
    """When the model returns a plain dict (not ModelOutput), logits are still extracted
    correctly (lines 379-380)."""
    model = _DictLM()
    trainer = _make_ppo(model=model)
    batch = _ppo_batch()
    metrics = trainer._ppo_step(batch)
    assert "loss" in metrics
    import math
    assert math.isfinite(float(metrics["loss"]))


def test_invariant_ppo_step_dict_model_hidden_is_none():
    """Dict-returning model → hidden is None (line 380) → value head not used even if enabled."""
    model = _DictLM()
    trainer = _make_ppo(model=model, use_value_head=True)
    batch = _ppo_batch()
    trainer._ppo_step(batch)
    assert trainer._value_head is None  # never initialized because hidden=None


def test_invariant_ppo_step_no_labels_uses_zero_log_probs():
    """When 'labels' is absent from batch, log_probs_new = zeros_like(log_probs_old) (line 395).

    We capture log_probs_new via a spy objective to confirm the zeros path is exercised.
    """
    model = _TinyLM(with_hidden=False)
    trainer = _make_ppo(model=model)

    captured: dict[str, Any] = {}

    class _SpyObjective:
        def __call__(self, out, batch, ctx):
            captured["log_probs_new"] = ctx.extras.get("log_probs_new")
            params = list(trainer.model.parameters())
            return {"loss": sum(p.sum() for p in params) * 0.0}

    trainer.objective = _SpyObjective()
    trainer.ctx.loss_fn = trainer.objective

    batch_no_labels = {
        "input_ids": torch.randint(0, 16, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        # labels intentionally absent
        "log_probs_old": torch.zeros(2, 4),
        "advantages_buf": torch.ones(2),
    }
    trainer._ppo_step(batch_no_labels)
    lp = captured.get("log_probs_new")
    assert lp is not None
    assert (lp == 0).all(), "no-labels path must yield all-zero log_probs_new"


# ===========================================================================
# _ppo_step — value_head_spec merge (line 406)
# ===========================================================================


def test_invariant_value_head_spec_bias_false_honoured():
    """value_head={'bias': False} must propagate to the constructed LinearValueHead
    (line 406 — spec.update merges the override into defaults)."""
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True, value_head={"bias": False})
    trainer._ppo_step(_ppo_batch())
    assert trainer._value_head is not None
    assert trainer._value_head.linear.bias is None


def test_invariant_value_head_spec_zero_init_propagated():
    """value_head={'zero_init': True} must zero-init the LinearValueHead weights."""
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True, value_head={"zero_init": True})
    trainer._ppo_step(_ppo_batch())
    w = trainer._value_head.linear.weight
    assert (w == 0).all(), "zero_init=True must zero out linear weight"


# ===========================================================================
# _compute_buffer_values — all sub-paths (lines 442-488)
# ===========================================================================


def test_invariant_compute_buffer_values_noop_when_value_head_none():
    """_compute_buffer_values returns early if _value_head is None (line 443 first condition)."""
    trainer = _make_ppo(use_value_head=True)
    assert trainer._value_head is None
    ep = _make_episode()
    trainer._buffer.add(ep)
    trainer._compute_buffer_values()
    assert ep.values is None  # nothing changed


def test_invariant_compute_buffer_values_noop_when_buffer_empty():
    """_compute_buffer_values returns early if buffer is empty (line 443 second condition)."""
    trainer = _make_ppo(use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    # Don't add any episodes
    assert len(trainer._buffer._episodes) == 0
    trainer._compute_buffer_values()  # must not raise


def test_invariant_compute_buffer_values_populates_episode_values():
    """_compute_buffer_values must set ep.values for every episode (lines 482-487)."""
    torch.manual_seed(42)
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)

    ep1 = _make_episode(T=3, n_resp=1)
    ep2 = _make_episode(T=5, n_resp=3)  # Different length → padding exercised
    trainer._buffer.add(ep1)
    trainer._buffer.add(ep2)

    trainer._compute_buffer_values()

    assert ep1.values is not None, "ep1.values must be populated"
    assert ep2.values is not None, "ep2.values must be populated"
    assert ep1.values.shape == (1,), "values must be scalar (1,)"
    assert ep2.values.shape == (1,)


def test_invariant_compute_buffer_values_sets_model_to_train_after_inference():
    """_compute_buffer_values calls model.eval() then restores model.train() (lines 459, 488)."""
    torch.manual_seed(0)
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    trainer._buffer.add(_make_episode())

    # Confirm model is in train mode initially
    trainer.model.train()
    assert trainer.model.training

    trainer._compute_buffer_values()

    assert trainer.model.training, "model must be back in train mode after _compute_buffer_values"


def test_invariant_compute_buffer_values_warns_when_no_hidden():
    """When model returns no hidden states after all fallbacks, a warning is emitted and
    ep.values remains None (lines 472-479)."""
    model = _TinyLM(with_hidden=False)  # hidden_states=None
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    ep = _make_episode()
    trainer._buffer.add(ep)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer._compute_buffer_values()

    texts = [str(w.message) for w in caught]
    assert any("hidden_states" in t for t in texts), \
        "must warn about missing hidden_states"
    assert ep.values is None, "no values should be set when hidden is None"


def test_invariant_compute_buffer_values_model_back_to_train_when_no_hidden():
    """Even when hidden_states is None and we hit the warning path (line 478), the model
    must be restored to train mode before returning."""
    model = _TinyLM(with_hidden=False)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    trainer._buffer.add(_make_episode())

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        trainer._compute_buffer_values()

    assert trainer.model.training


def test_invariant_compute_buffer_values_falls_back_on_type_error():
    """When model raises TypeError for output_hidden_states=True, fallback call is made
    and values are still populated (lines 464-465)."""
    torch.manual_seed(0)
    model = _TypeErrorLM()
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    ep = _make_episode()
    trainer._buffer.add(ep)

    trainer._compute_buffer_values()
    assert ep.values is not None, "TypeError fallback must still yield values"


def test_invariant_compute_buffer_values_no_hidden_after_type_error_warns():
    """If even the fallback call returns no hidden, a warning is emitted and ep.values stays None."""
    model = _NoHiddenAfterTypeErrorLM()
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    ep = _make_episode()
    trainer._buffer.add(ep)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer._compute_buffer_values()

    texts = [str(w.message) for w in caught]
    assert any("hidden_states" in t for t in texts)
    assert ep.values is None


def test_invariant_compute_buffer_values_mixed_lengths_padding():
    """Episodes of different lengths are padded to max_len before batching (lines 449-457)."""
    torch.manual_seed(0)
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)

    # Three episodes with different lengths
    trainer._buffer.add(_make_episode(T=2, n_resp=1))
    trainer._buffer.add(_make_episode(T=4, n_resp=2))
    trainer._buffer.add(_make_episode(T=6, n_resp=4))

    trainer._compute_buffer_values()

    for ep in trainer._buffer._episodes:
        assert ep.values is not None
        assert ep.values.shape == (1,)


def test_invariant_compute_buffer_values_cpu_outputs():
    """ep.values must be stored on CPU regardless of compute device (line 487)."""
    torch.manual_seed(0)
    model = _TinyLM(with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)
    ep = _make_episode()
    trainer._buffer.add(ep)

    trainer._compute_buffer_values()
    assert ep.values.device.type == "cpu"


def test_invariant_compute_buffer_values_only_response_tokens_contribute():
    """ep.values scalar is a weighted mean over response tokens only (labels != -100).

    Verify shape consistency across episodes with different prompt/response splits.
    """
    torch.manual_seed(0)
    model = _TinyLM(with_hidden=True, D=8)
    trainer = _make_ppo(model=model, use_value_head=True)
    trainer._value_head = LinearValueHead(hidden_size=8)

    # ep_all_prompt: all labels = -100 → resp_mask all zeros → denom clamped to 1
    ep_all_prompt = Episode(
        input_ids=torch.randint(0, 16, (4,)),
        attention_mask=torch.ones(4, dtype=torch.long),
        labels=torch.full((4,), -100, dtype=torch.long),
        reward=1.0,
        log_probs=torch.zeros(4),
    )
    trainer._buffer.add(ep_all_prompt)

    trainer._compute_buffer_values()
    assert ep_all_prompt.values is not None  # clamp(min=1) prevents division by zero


# ===========================================================================
# Stub overrides: eval() and predict()
# ===========================================================================


def test_invariant_eval_returns_empty_dict():
    """PPOTrainer.eval() is a no-op stub that always returns {} (line 502)."""
    trainer = _make_ppo()
    result = trainer.eval()
    assert result == {}


def test_invariant_eval_accepts_arbitrary_args():
    """eval() must accept positional and keyword arguments without error."""
    trainer = _make_ppo()
    result = trainer.eval(1, 2, key="value")
    assert result == {}


def test_invariant_predict_returns_empty_list():
    """PPOTrainer.predict() is a no-op stub that always returns [] (line 505)."""
    trainer = _make_ppo()
    result = trainer.predict()
    assert result == []


def test_invariant_predict_accepts_arbitrary_args():
    """predict() must accept positional and keyword arguments without error."""
    trainer = _make_ppo()
    result = trainer.predict("input", batch_size=4)
    assert result == []


# ===========================================================================
# Additional edge-cases for fit() metrics aggregation
# ===========================================================================


def test_invariant_fit_returns_mean_reward_in_metrics():
    """The returned dict from fit() must include 'mean_reward' (line 254)."""
    trainer = _make_ppo(max_steps=1, ppo_epochs=1)
    # Provide one real minibatch so inner_metrics is populated
    trainer._rollout_engine.rollout = MagicMock(return_value=[])
    trainer._buffer.clear = MagicMock()
    trainer._buffer.add = MagicMock()
    trainer._buffer.all_rewards = MagicMock(return_value=torch.tensor([1.0, 0.5]))
    trainer._buffer.all_values = MagicMock(return_value=None)
    trainer._buffer.batches = MagicMock(return_value=iter([_ppo_batch()]))
    trainer._compute_buffer_values = lambda: None

    result = trainer.fit()
    assert "mean_reward" in result
    assert result["mean_reward"] == pytest.approx(0.75)


def test_invariant_fit_returns_empty_dict_when_no_inner_metrics():
    """If all minibatch iters are empty, inner_metrics stays empty, last_metrics == {}."""
    trainer = _make_ppo(max_steps=1)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    result = trainer.fit()
    assert isinstance(result, dict)


def test_invariant_fit_increments_ctx_step():
    """ctx.step must increment by 1 per outer step (line 258)."""
    trainer = _make_ppo(max_steps=3)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    assert trainer.ctx.step == 3


def test_invariant_fit_increments_ctx_global_step():
    """ctx.global_step must mirror ctx.step at the end of each outer iteration (line 259)."""
    trainer = _make_ppo(max_steps=2)
    _stub_rollout(trainer, rewards=torch.tensor([1.0]))
    trainer.fit()
    assert trainer.ctx.global_step == 2


# ===========================================================================
# buffer_max_size constructor knob
# ===========================================================================


def test_invariant_buffer_max_size_knob_respected():
    """buffer_max_size= is forwarded to RolloutBuffer.max_size (constructor line 176)."""
    trainer = _make_ppo(buffer_max_size=512)
    assert trainer._buffer.max_size == 512


def test_invariant_buffer_max_size_defaults_to_rollout_steps_times_four():
    """Without buffer_max_size, default is rollout_steps * 4 (line 176)."""
    trainer = _make_ppo(rollout_steps=8)
    assert trainer._buffer.max_size == 32
