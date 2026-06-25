"""Adversarial tests for PPOTrainer — fit lifecycle, value head lazy init,
engine bypass, target_kl, signal handling.

PPOTrainer's ``fit()`` orchestrates: ref-policy freeze → outer rollout loop
(rollout → GAE → inner PPO epochs → step advance). The ``_ppo_step`` path
calls ``self._rl_rule.step(self.model, batch, self.ctx)`` directly,
bypassing the engine entirely (engine is the StandardUpdateRule path).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.trainers.ppo import LinearValueHead, PPOTrainer
from lighttrain.callbacks.base import Signal
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# fixtures specific to PPO
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Tiny causal LM that returns ModelOutput with optional hidden_states."""

    def __init__(self, V: int = 16, D: int = 8, with_hidden: bool = False) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)
        self._with_hidden = with_hidden

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        logits = self.proj(h)
        if self._with_hidden:
            return ModelOutput(outputs={"logits": logits}, hidden_states=(h,))
        return ModelOutput(outputs={"logits": logits})


class _FakeEngine:
    """Sentinel engine — PPOTrainer must NOT call ``engine.step`` for RL.

    We wrap it as a MagicMock in tests that need to spy on call count.
    """


class _FakeDM:
    """Yields a single prompt batch then stops (controlled by max_steps)."""

    def __init__(self, n_yields: int = 100):
        self._n = n_yields

    def train_loader(self):
        V, T, B = 16, 4, 2
        for _ in range(self._n):
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
                "labels": torch.randint(0, V, (B, T)),
            }


def _reward_fn(response_ids, batch) -> list[float]:
    B = response_ids.shape[0] if isinstance(response_ids, torch.Tensor) else len(response_ids)
    return [0.5] * B


def _make_ppo(
    *,
    model: nn.Module | None = None,
    callbacks: list | None = None,
    engine: Any = None,
    **over,
) -> PPOTrainer:
    if model is None:
        model = _TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return PPOTrainer(
        engine=engine if engine is not None else _FakeEngine(),
        data_module=_FakeDM(),
        optimizer=opt,
        model=model,
        callbacks=callbacks,
        max_steps=over.pop("max_steps", 1),
        rollout_steps=over.pop("rollout_steps", 2),
        ppo_epochs=over.pop("ppo_epochs", 1),
        mini_batch_size=over.pop("mini_batch_size", 2),
        max_new_tokens=over.pop("max_new_tokens", 4),
        reward_fn=over.pop("reward_fn", _reward_fn),
        **over,
    )


def _stub_rollout_phase(trainer: PPOTrainer, rewards: torch.Tensor) -> None:
    """Replace the rollout machinery so fit() runs end-to-end without
    needing a real model.generate.

    - rollout returns [] (no episodes added)
    - buffer.all_rewards returns the supplied tensor
    - buffer.all_values returns None
    - buffer.batches returns empty iter (skip inner-epoch body)
    - _compute_buffer_values is a no-op (value head path not exercised here)
    """
    trainer._rollout_engine.rollout = MagicMock(return_value=[])  # type: ignore[method-assign]
    trainer._buffer.clear = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.add = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.all_rewards = MagicMock(return_value=rewards)  # type: ignore[method-assign]
    trainer._buffer.all_values = MagicMock(return_value=None)  # type: ignore[method-assign]
    trainer._buffer.batches = MagicMock(return_value=iter([]))  # type: ignore[method-assign]
    trainer._compute_buffer_values = lambda: None  # type: ignore[method-assign]


def _ppo_batch(V: int = 16, T: int = 4, B: int = 2) -> dict:
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "advantages_buf": torch.ones(B),
    }


# ===========================================================================
# fit() lifecycle
# ===========================================================================


def test_ppo_fit_lifecycle_strict_order():
    """Goal: pin the exact temporal order of fit-level events.

    Construction: max_steps=1, mock rollout phase (no episodes, empty
    batches), and record bus events with an OrderedRecorder. Inner-step
    events (on_step_begin, etc.) are excluded by the empty inner loop.

    Expected fit-level events in order:
      [on_train_start, on_epoch_begin, on_rollout_begin, on_rollout_end,
       on_reward_computed, on_train_end]
    """
    events: list[str] = []

    class _Rec:
        def on_train_start(self, **_): events.append("on_train_start")
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")
        def on_rollout_begin(self, **_): events.append("on_rollout_begin")
        def on_rollout_end(self, **_): events.append("on_rollout_end")
        def on_reward_computed(self, **_): events.append("on_reward_computed")
        def on_exception(self, **_): events.append("on_exception")
        def on_train_end(self, **_): events.append("on_train_end")

    trainer = _make_ppo(callbacks=[_Rec()], max_steps=1)
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    trainer.fit()

    assert events == [
        "on_train_start",
        "on_epoch_begin",
        "on_rollout_begin",
        "on_rollout_end",
        "on_reward_computed",
        "on_train_end",
    ]


def test_ppo_fit_emits_epoch_end_then_epoch_begin_on_iterator_exhaustion():
    """Goal: when train_loader's iterator raises StopIteration, fit must
    dispatch on_epoch_end FIRST (closing the old epoch), bump ctx.epoch,
    then on_epoch_begin (opening the new one), then re-pull.

    Construction: max_steps=2 with a DM that yields only ONCE before
    StopIteration — second iter goes through the exhaustion path.

    Catches a refactor that drops on_epoch_end on iterator exhaustion
    (lines 200-204 in ppo.py).
    """
    events: list[str] = []

    class _Rec:
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")

    class _OneBatchIter:
        """An iterable whose ``__iter__`` yields exactly one batch each time."""

        def __iter__(self_inner):
            V, T, B = 16, 4, 2
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }

    class _OneShotDM:
        def train_loader(self):
            # Returns a re-iterable object — each iter() yields ONE batch.
            return _OneBatchIter()

    dm = _OneShotDM()
    model = _TinyLM()
    trainer = PPOTrainer(
        engine=_FakeEngine(),
        data_module=dm,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Rec()],
        max_steps=2,
        rollout_steps=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_reward_fn,
    )
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    trainer.fit()

    # Sequence must contain: on_epoch_begin (epoch 0), on_epoch_end, on_epoch_begin (epoch 1)
    assert events[:3] == ["on_epoch_begin", "on_epoch_end", "on_epoch_begin"]


def test_ppo_fit_raises_when_model_is_none():
    """Pin constructor / fit guard at ppo.py:175-176."""
    model = _TinyLM()
    trainer = PPOTrainer(
        engine=_FakeEngine(),
        data_module=_FakeDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        rollout_steps=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_reward_fn,
    )
    trainer.model = None
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.fit()


def test_ppo_fit_raises_when_optimizer_is_none():
    """Pin guard at ppo.py:177-178."""
    trainer = _make_ppo()
    trainer.optimizer = None
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        trainer.fit()


def test_ppo_fit_raises_when_reward_fn_is_none():
    """Pin guard at ppo.py:179-180."""
    trainer = _make_ppo()
    trainer.reward_fn = None
    with pytest.raises(RuntimeError, match="reward_fn is not set"):
        trainer.fit()


# ===========================================================================
# Engine bypass — core contract for RL trainers
# ===========================================================================


def test_ppo_step_bypasses_standard_engine_step():
    """Goal: PPOTrainer's ``_ppo_step`` must NOT call ``engine.step`` —
    it goes directly through ``self._rl_rule.step``.

    Construction: pass a MagicMock as the engine. Run ``_ppo_step`` once
    with a hand-crafted batch.

    Catches a refactor that "unifies" trainers by routing PPO through the
    engine. That would silently apply ``StandardUpdateRule``, which calls
    ``model(**batch)`` itself and computes the wrong gradients (PPO needs
    pre-computed log_probs from ctx.extras, not a fresh forward).
    """
    engine_mock = MagicMock()
    trainer = _make_ppo(engine=engine_mock)
    batch = _ppo_batch()

    trainer._ppo_step(batch)

    engine_mock.step.assert_not_called()


# ===========================================================================
# Value head lazy init
# ===========================================================================


def test_ppo_value_head_lazy_init_only_when_use_value_head_true():
    """Goal: with use_value_head=False, _value_head stays None even after
    a step that runs forward.

    Catches a refactor that always builds the value head.
    """
    trainer = _make_ppo(use_value_head=False)
    assert trainer._value_head is None
    trainer._ppo_step(_ppo_batch())
    assert trainer._value_head is None


def test_ppo_value_head_lazy_init_uses_hidden_dim_from_output():
    """Goal: when use_value_head=True and the model returns hidden_states,
    the first _ppo_step constructs a LinearValueHead with in_features ==
    hidden_size.

    Construction: model returns ModelOutput with hidden_states (D=8).
    Expected: trainer._value_head.linear.in_features == 8.
    """
    model = _TinyLM(D=8, with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    assert trainer._value_head is None

    trainer._ppo_step(_ppo_batch())

    assert trainer._value_head is not None
    assert isinstance(trainer._value_head, LinearValueHead)
    assert trainer._value_head.linear.in_features == 8


def test_ppo_value_head_lazy_init_registers_params_with_optimizer():
    """Goal: lazy init must register value head params with the optimizer
    so they actually train — otherwise optimizer.step is a no-op on them.

    Catches a refactor that drops the ``_register_new_params`` call.
    """
    model = _TinyLM(D=8, with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)
    n_groups_before = len(trainer.optimizer.param_groups)

    trainer._ppo_step(_ppo_batch())

    assert trainer._value_head is not None
    assert len(trainer.optimizer.param_groups) == n_groups_before + 1
    # And the head params must actually be in some param_group
    head_param_ids = {id(p) for p in trainer._value_head.parameters()}
    in_optim_ids = {
        id(p) for g in trainer.optimizer.param_groups for p in g["params"]
    }
    assert head_param_ids.issubset(in_optim_ids)


def test_ppo_value_head_not_re_initialized_on_second_step():
    """Goal: the second call to ``_ppo_step`` must reuse the existing value
    head (same Python object), not allocate a new one.

    Catches a refactor that drops the ``if self._value_head is None`` gate.
    """
    model = _TinyLM(D=8, with_hidden=True)
    trainer = _make_ppo(model=model, use_value_head=True)

    trainer._ppo_step(_ppo_batch())
    head_id_after_step1 = id(trainer._value_head)

    trainer._ppo_step(_ppo_batch())
    head_id_after_step2 = id(trainer._value_head)

    assert head_id_after_step1 == head_id_after_step2


def test_ppo_value_head_skipped_when_hidden_is_none():
    """Goal: use_value_head=True but model returns no hidden_states ⇒
    _value_head stays None and the step succeeds (uses zero baseline).

    Construction: TinyLM with with_hidden=False; use_value_head=True.
    Expected: ``trainer._value_head is None`` after step.
    """
    model = _TinyLM(D=8, with_hidden=False)
    trainer = _make_ppo(model=model, use_value_head=True)

    metrics = trainer._ppo_step(_ppo_batch())

    assert trainer._value_head is None
    assert "loss" in metrics  # step still completed


# ===========================================================================
# target_kl early stop
# ===========================================================================


def test_ppo_target_kl_early_stops_inner_loop():
    """Goal: when ``train_step`` reports approx_kl > target_kl, the inner
    PPO epoch loop must break out (early stop).

    Construction:
      - target_kl=0.01
      - patch ``trainer.train_step`` so first call returns approx_kl=0.005
        (below threshold), second call returns approx_kl=0.99 (above).
      - mock buffer.batches to return 4 minibatches per epoch.

    Expected: train_step is called exactly twice (broke after 2nd hit).

    Catches a refactor that compares against the wrong field or always
    runs the full inner loop.
    """
    from lighttrain.protocols import StepOutput

    call_log: list[float] = []

    def _fake_train_step(batch):
        # First two calls: under threshold, then over. The minibatch loop
        # should break after the over-threshold step (idx=1).
        kl = 0.005 if len(call_log) < 1 else 0.99
        call_log.append(kl)
        return StepOutput(loss=1.0, metrics={"loss": 1.0, "approx_kl": kl})

    trainer = _make_ppo(target_kl=0.01, max_steps=1, ppo_epochs=2)
    # 4 minibatches per epoch — but early stop should kick in after 2 steps.
    trainer._rollout_engine.rollout = MagicMock(return_value=[])
    trainer._buffer.clear = MagicMock()
    trainer._buffer.add = MagicMock()
    trainer._buffer.all_rewards = MagicMock(return_value=torch.tensor([1.0, 0.5]))
    trainer._buffer.all_values = MagicMock(return_value=None)
    # Provide many minibatches across two epochs
    trainer._buffer.batches = MagicMock(
        side_effect=[iter([_ppo_batch() for _ in range(4)]), iter([_ppo_batch() for _ in range(4)])]
    )
    trainer._compute_buffer_values = lambda: None
    trainer.train_step = _fake_train_step

    trainer.fit()

    # Two train_step calls: 1st = 0.005 (under), 2nd = 0.99 (over → break).
    # The break should exit both inner loops because early_stop=True at the
    # epoch level (lines 271-273 in ppo.py).
    assert len(call_log) == 2


# ===========================================================================
# loss_signal clearing
# ===========================================================================


def test_ppo_step_clears_loss_signal_extras_per_call():
    """Goal: ``_step`` must pop ``ctx.extras['loss_signal']`` before invoking
    the underlying ``_ppo_step`` (so a stale signal from the previous
    iteration cannot persist into the next).

    Catches a refactor that drops line 459 in ppo.py.
    """
    trainer = _make_ppo()
    trainer.ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)
    assert "loss_signal" in trainer.ctx.extras

    trainer._step(_ppo_batch())

    # _ppo_step's downstream RLUpdateRule will only re-set loss_signal if
    # a callback returns a non-CONTINUE signal; with no callbacks attached,
    # the slot must be cleared and not re-added.
    assert "loss_signal" not in trainer.ctx.extras


# ===========================================================================
# Per-step callback chain (strict order)
# ===========================================================================


def test_ppo_step_fires_full_callback_chain_in_strict_order():
    """Goal: tighten the legacy ``event in fired`` test to an ordered list
    over the per-step events emitted by RLUpdateRule under PPO.

    Expected list (RL has no on_forward_pre / on_forward_post since the
    trainer owns the forward):
      [on_step_begin, on_loss_computed, on_backward_pre, on_backward_post,
       on_clip_grad, on_optimizer_step_pre, on_optimizer_step_post,
       on_zero_grad, on_step_end]
    """
    events: list[str] = []

    class _Rec:
        def on_step_begin(self, **_): events.append("on_step_begin")
        def on_loss_computed(self, **_): events.append("on_loss_computed")
        def on_backward_pre(self, **_): events.append("on_backward_pre")
        def on_backward_post(self, **_): events.append("on_backward_post")
        def on_clip_grad(self, **_): events.append("on_clip_grad")
        def on_optimizer_step_pre(self, **_): events.append("on_optimizer_step_pre")
        def on_optimizer_step_post(self, **_): events.append("on_optimizer_step_post")
        def on_zero_grad(self, **_): events.append("on_zero_grad")
        def on_step_end(self, **_): events.append("on_step_end")

    trainer = _make_ppo(callbacks=[_Rec()])
    trainer._ppo_step(_ppo_batch())

    expected = [
        "on_step_begin",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
        "on_step_end",
    ]
    assert events == expected


# ===========================================================================
# Exception path
# ===========================================================================


def test_ppo_fit_dispatches_on_exception_on_raise():
    """Goal: when something inside the fit loop raises, on_exception fires
    and the exception re-propagates.

    Construction: a callback whose on_rollout_begin raises. Asserts:
      - exception bubbles out of fit
      - on_exception was dispatched
      - on_train_end still fires (finally block)
    """
    events: list[str] = []

    class _BoomCb:
        critical = True  # so the raise bubbles through the bus
        def on_rollout_begin(self, **_): raise ValueError("simulated crash")
        def on_exception(self, **_): events.append("on_exception")
        def on_train_end(self, **_): events.append("on_train_end")

    trainer = _make_ppo(callbacks=[_BoomCb()])
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    with pytest.raises(ValueError, match="simulated crash"):
        trainer.fit()

    assert "on_exception" in events
    assert "on_train_end" in events


def test_ppo_fit_dispatches_on_train_end_even_on_exception():
    """Companion: on_train_end is in the finally block — even when fit
    raises, it must run before the re-raise.

    Catches a refactor that moves on_train_end out of finally.
    """
    fired = []

    class _BoomCb:
        critical = True
        def on_rollout_begin(self, **_): raise RuntimeError("kaboom")
        def on_train_end(self, **_): fired.append("on_train_end")

    trainer = _make_ppo(callbacks=[_BoomCb()])
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0]))

    with pytest.raises(RuntimeError, match="kaboom"):
        trainer.fit()

    assert fired == ["on_train_end"]


# ===========================================================================
# stop_requested
# ===========================================================================


def test_ppo_stop_requested_breaks_outer_loop():
    """Goal: setting ``trainer._stop_requested = True`` from any callback
    must end the outer loop. With max_steps=10 and stop after step 1,
    we expect ctx.step == 1.
    """

    class _Stopper:
        def on_reward_computed(self, ctx=None, **_):
            # Trigger stop after the first outer iteration completes
            trainer._stop_requested = True

    trainer = _make_ppo(callbacks=[_Stopper()], max_steps=10)
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0]))

    trainer.fit()

    assert trainer.ctx.step == 1


# ===========================================================================
# Registry + constructor-config invariants (merged from
# tests/test_trainer_ppo.py)
# ===========================================================================


def test_ppo_resolves_from_registry():
    """The 'ppo' trainer name resolves to PPOTrainer through the registry."""
    from lighttrain.registry import get as resolve

    assert resolve("trainer", "ppo") is PPOTrainer


def test_ppo_step_returns_finite_loss():
    """A hand-crafted PPO minibatch produces a finite 'loss' metric."""
    trainer = _make_ppo()
    metrics = trainer._ppo_step(_ppo_batch())
    assert "loss" in metrics
    import math

    assert math.isfinite(float(metrics["loss"]))


def test_ppo_ref_policy_params_frozen_after_freeze_as_ref():
    """freeze_as_ref produces a ref policy whose params have requires_grad=False."""
    from lighttrain.builtin_plugins.rl.ref_policy import freeze_as_ref

    trainer = _make_ppo()
    trainer._ref_policy = freeze_as_ref(trainer.model)
    for p in trainer._ref_policy.model.parameters():
        assert not p.requires_grad


def test_ppo_target_kl_stored_on_trainer():
    """target_kl passed to the constructor is retained as an attribute."""
    trainer = _make_ppo(target_kl=0.01)
    assert trainer.target_kl == 0.01


def test_ppo_clip_eps_propagated_to_default_loss():
    """clip_eps feeds the default RL loss used when the recipe omits a `loss:`
    block (the loss: seam wins when present — keystone step 3)."""
    trainer = _make_ppo(clip_eps=0.3)
    assert trainer._default_loss.clip_eps == 0.3
