"""Adversarial tests for GRPOTrainer — fit lifecycle, engine bypass,
loss_signal clearing, callback chain order.

GRPO is similar to PPO in structure (same RL flow), but has no value head
and no target_kl early stop. Group-relative advantage normalization happens
inside GRPOLoss, not the trainer.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.losses.rl import GRPOLoss
from lighttrain.builtin_plugins.rl.ref_policy import freeze_as_ref
from lighttrain.builtin_plugins.trainers.grpo import GRPOTrainer, _effective_beta_kl
from lighttrain.callbacks.base import Signal
from lighttrain.optim.architectures.profile import LossOnlyObjective
from lighttrain.protocols import ModelOutput


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _FakeEngine:
    pass


class _FakeDM:
    def train_loader(self):
        V, T, B = 16, 4, 2
        while True:
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }


def _reward(ids, batch):
    n = ids.shape[0] if isinstance(ids, torch.Tensor) else len(ids)
    return [1.0] * n


def _make_grpo(*, model=None, callbacks=None, engine=None, **over) -> GRPOTrainer:
    if model is None:
        model = _TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return GRPOTrainer(
        engine=engine if engine is not None else _FakeEngine(),
        data_module=_FakeDM(),
        optimizer=opt,
        model=model,
        callbacks=callbacks,
        max_steps=over.pop("max_steps", 1),
        group_size=over.pop("group_size", 2),
        ppo_epochs=over.pop("ppo_epochs", 1),
        mini_batch_size=over.pop("mini_batch_size", 2),
        max_new_tokens=over.pop("max_new_tokens", 4),
        reward_fn=over.pop("reward_fn", _reward),
        **over,
    )


def _stub_rollout_phase(trainer: GRPOTrainer, rewards: torch.Tensor) -> None:
    trainer._rollout_engine.rollout = MagicMock(return_value=[])  # type: ignore[method-assign]
    trainer._buffer.clear = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.add = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.all_rewards = MagicMock(return_value=rewards)  # type: ignore[method-assign]
    trainer._buffer.batches = MagicMock(return_value=iter([]))  # type: ignore[method-assign]


def _grpo_batch(V: int = 16, T: int = 4, B: int = 4) -> dict:
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
        "log_probs_old": torch.zeros(B, T),
        "rewards": torch.tensor([1.0, 0.5, 0.5, -0.5]),
        "group_ids": torch.tensor([0, 0, 1, 1]),
    }


# ===========================================================================
# fit lifecycle
# ===========================================================================


def test_grpo_fit_lifecycle_strict_order():
    """Goal: pin exact temporal order of fit-level events.

    Expected:
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
        def on_train_end(self, **_): events.append("on_train_end")

    trainer = _make_grpo(callbacks=[_Rec()], max_steps=1)
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


def test_grpo_fit_emits_epoch_end_then_epoch_begin_on_iterator_exhaustion():
    """Pin lines 139-143 in grpo.py: on iterator exhaustion,
    on_epoch_end → epoch++ → on_epoch_begin in that order.
    """
    events: list[str] = []

    class _Rec:
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")

    class _OneBatchIter:
        def __iter__(self_inner):
            V, T, B = 16, 4, 2
            yield {
                "input_ids": torch.randint(0, V, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.long),
            }

    class _OneShotDM:
        def train_loader(self):
            return _OneBatchIter()

    model = _TinyLM()
    trainer = GRPOTrainer(
        engine=_FakeEngine(),
        data_module=_OneShotDM(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Rec()],
        max_steps=2,
        group_size=2,
        ppo_epochs=1,
        mini_batch_size=2,
        max_new_tokens=4,
        reward_fn=_reward,
    )
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0, 0.5]))

    trainer.fit()

    assert events[:3] == ["on_epoch_begin", "on_epoch_end", "on_epoch_begin"]


def test_grpo_fit_raises_when_model_is_none():
    trainer = _make_grpo()
    trainer.model = None
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.fit()


def test_grpo_fit_raises_when_optimizer_is_none():
    trainer = _make_grpo()
    trainer.optimizer = None
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        trainer.fit()


def test_grpo_fit_raises_when_reward_fn_is_none():
    trainer = _make_grpo()
    trainer.reward_fn = None
    with pytest.raises(RuntimeError, match="reward_fn is not set"):
        trainer.fit()


# ===========================================================================
# Engine bypass
# ===========================================================================


def test_grpo_step_bypasses_standard_engine_step():
    """Goal: GRPOTrainer must NOT call engine.step — same contract as PPO.

    Catches a refactor that routes GRPO through the engine, which would
    silently apply StandardUpdateRule and re-forward the model.
    """
    engine_mock = MagicMock()
    trainer = _make_grpo(engine=engine_mock)

    trainer._grpo_step(_grpo_batch())

    engine_mock.step.assert_not_called()


# ===========================================================================
# loss_signal clearing
# ===========================================================================


def test_grpo_step_clears_loss_signal_extras_per_call():
    """Goal: line 283 in grpo.py — ``_step`` pops ``loss_signal`` from
    ctx.extras before delegating to ``_grpo_step``.
    """
    trainer = _make_grpo()
    trainer.ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)

    trainer._step(_grpo_batch())

    assert "loss_signal" not in trainer.ctx.extras


# ===========================================================================
# Per-step callback chain (strict order)
# ===========================================================================


def test_grpo_step_fires_full_callback_chain_in_strict_order():
    """Goal: ordered list of per-step events fired by RLUpdateRule under GRPO.
    Same expected sequence as PPO (both use RLUpdateRule).
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

    trainer = _make_grpo(callbacks=[_Rec()])
    trainer._grpo_step(_grpo_batch())

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
# GRPO-specific contract
# ===========================================================================


def test_grpo_has_no_value_head_attribute():
    """Goal: pin contract — GRPO does NOT have a value head. Introducing
    one would change advantage computation semantics (GRPO uses
    group-relative normalization, not GAE).
    """
    trainer = _make_grpo()
    assert not hasattr(trainer, "_value_head")
    assert not hasattr(trainer, "_use_value_head")


def test_grpo_stop_requested_breaks_outer_loop():
    """Goal: ``trainer._stop_requested = True`` from any callback breaks the
    outer loop early.
    """

    class _Stopper:
        def __init__(self):
            self.trainer = None
        def on_reward_computed(self, **_):
            assert self.trainer is not None
            self.trainer._stop_requested = True

    stopper = _Stopper()
    trainer = _make_grpo(callbacks=[stopper], max_steps=10)
    stopper.trainer = trainer
    _stub_rollout_phase(trainer, rewards=torch.tensor([1.0]))

    trainer.fit()

    assert trainer.ctx.step == 1


# ===========================================================================
# KL reference-policy wiring (L-P0f) — the trainer must inject per-token
# log_probs_ref so a configured beta_kl actually applies the KL penalty.
# The k3 KL math itself is covered in tests/losses/test_rl.py — here we pin
# the *wiring* (shape, gating, stale-clearing, fail-loud paths).
# ===========================================================================


def _stub_rollout_with_batch(trainer: GRPOTrainer, rewards: torch.Tensor, batch: dict) -> None:
    """Like _stub_rollout_phase but feeds one real minibatch into the inner loop."""
    trainer._rollout_engine.rollout = MagicMock(return_value=[])  # type: ignore[method-assign]
    trainer._buffer.clear = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.add = MagicMock()  # type: ignore[method-assign]
    trainer._buffer.all_rewards = MagicMock(return_value=rewards)  # type: ignore[method-assign]
    trainer._buffer.batches = MagicMock(return_value=iter([batch]))  # type: ignore[method-assign]


def test_effective_beta_kl_unwraps_objective_and_defaults_to_zero():
    """_effective_beta_kl honors loss-seam: unwrap LossOnlyObjective.loss_fn,
    read beta_kl; a loss without the knob → 0.0 (never builds an unusable ref)."""
    assert _effective_beta_kl(GRPOLoss(beta_kl=0.3)) == pytest.approx(0.3)
    assert _effective_beta_kl(
        LossOnlyObjective(GRPOLoss(beta_kl=0.3), loss_family="rl")
    ) == pytest.approx(0.3)

    def _custom_loss(model_output, batch, ctx):  # no beta_kl attribute
        return {"loss": torch.tensor(0.0)}

    assert _effective_beta_kl(LossOnlyObjective(_custom_loss)) == 0.0


def test_grpo_step_injects_per_token_log_probs_ref_shape():
    """beta_kl>0 + ref built → ctx.extras['log_probs_ref'] is (B, T) (== log_probs_new),
    first column 0, and the kl metric is finite."""
    trainer = _make_grpo(beta_kl=1.0)
    trainer._ref_policy = freeze_as_ref(trainer.model)

    batch = _grpo_batch()  # B=4, T=4
    raw = trainer._grpo_step(batch)

    ref = trainer.ctx.extras["log_probs_ref"]
    assert ref.shape == (4, 4)
    assert torch.all(ref[:, 0] == 0.0)
    assert math.isfinite(raw["kl"])


def test_grpo_step_kl_positive_with_distinct_ref():
    """A reference that differs from the current policy yields kl_loss > 0.

    (A just-frozen ref equals the policy → KL=0 by construction, so we inject a
    separately-initialized model as the reference to exercise a real penalty.)
    """
    trainer = _make_grpo(beta_kl=1.0)
    trainer._ref_policy = freeze_as_ref(_TinyLM())  # independent init → differs

    raw = trainer._grpo_step(_grpo_batch())

    assert raw["kl"] > 0.0


def test_grpo_step_fail_loud_when_beta_kl_but_no_ref():
    """Direct _grpo_step (no fit()) with beta_kl>0 and no ref → GRPOLoss raises
    instead of silently dropping the KL term."""
    trainer = _make_grpo(beta_kl=0.1)
    assert trainer._ref_policy is None
    with pytest.raises(RuntimeError, match="log_probs_ref"):
        trainer._grpo_step(_grpo_batch())


def test_grpo_step_clears_stale_log_probs_ref():
    """Once a step injects log_probs_ref, a later step without a ref must not
    leave the stale value in ctx.extras (GRPOLoss only checks key presence)."""
    trainer = _make_grpo()  # beta_kl=0 → GRPOLoss won't require the key
    trainer._ref_policy = freeze_as_ref(trainer.model)
    trainer._grpo_step(_grpo_batch())
    assert "log_probs_ref" in trainer.ctx.extras

    trainer._ref_policy = None
    trainer._grpo_step(_grpo_batch())
    assert "log_probs_ref" not in trainer.ctx.extras


def test_grpo_step_fail_loud_when_kl_enabled_but_no_labels():
    """KL enabled (ref built) but a batch without labels → fail loud rather than
    forming a meaningless 'ref vs zeros' KL."""
    trainer = _make_grpo(beta_kl=1.0)
    trainer._ref_policy = freeze_as_ref(trainer.model)
    batch = _grpo_batch()
    del batch["labels"]
    with pytest.raises(RuntimeError, match="labels"):
        trainer._grpo_step(batch)


def test_grpo_fit_builds_ref_only_when_beta_kl_positive():
    """The fit() gate builds the reference policy iff effective beta_kl>0."""
    batch = _grpo_batch()

    t_kl = _make_grpo(beta_kl=0.5, max_steps=1)
    _stub_rollout_with_batch(t_kl, rewards=batch["rewards"], batch=batch)
    t_kl.fit()
    assert t_kl._ref_policy is not None

    t_no = _make_grpo(beta_kl=0.0, max_steps=1)
    _stub_rollout_with_batch(t_no, rewards=batch["rewards"], batch=_grpo_batch())
    t_no.fit()
    assert t_no._ref_policy is None


def test_grpo_step_beta_kl_zero_does_not_inject_ref():
    """Behavior-neutral baseline: default beta_kl=0 builds no ref, injects no
    log_probs_ref, and computes a zero KL term (unchanged from pre-fix)."""
    trainer = _make_grpo()  # beta_kl=0
    raw = trainer._grpo_step(_grpo_batch())
    assert trainer._ref_policy is None
    assert "log_probs_ref" not in trainer.ctx.extras
    assert raw["kl"] == 0.0


# ===========================================================================
# A2 — lora_base_as_ref per-token KL (the LoRA-base reference path is now wired).
# ===========================================================================


class _LoRALikeLM(_TinyLM):
    """_TinyLM + PEFT-style adapter toggles (no-ops here) so the LoRA-base
    reference path can disable/enable adapters around the base forward."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__(V, D)
        self.disable_calls = 0
        self.enable_calls = 0

    def disable_adapter_layers(self) -> None:
        self.disable_calls += 1

    def enable_adapter_layers(self) -> None:
        self.enable_calls += 1


def test_grpo_step_lora_base_per_token_kl_runs():
    """A2 e2e: lora_base_as_ref ref injects a (B, T) per-token log_probs_ref via
    adapter toggling; KL metric is finite and the adapters are restored."""
    model = _LoRALikeLM()
    trainer = _make_grpo(model=model, beta_kl=1.0, lora_base_as_ref=True)
    trainer._ref_policy = freeze_as_ref(model, lora_base_as_ref=True)

    raw = trainer._grpo_step(_grpo_batch())  # B=4, T=4

    ref = trainer.ctx.extras["log_probs_ref"]
    assert ref.shape == (4, 4)
    assert torch.all(ref[:, 0] == 0.0)
    assert math.isfinite(raw["kl"])
    # adapters toggled once each and left enabled
    assert model.disable_calls == 1
    assert model.enable_calls == 1


def test_grpo_fit_lora_base_as_ref_with_kl_builds_ref():
    """A2: fit() with beta_kl>0 + lora_base_as_ref=True no longer raises (the
    guard is removed); it builds a lora-base reference policy."""
    model = _LoRALikeLM()
    batch = _grpo_batch()
    t = _make_grpo(model=model, beta_kl=0.5, lora_base_as_ref=True, max_steps=1)
    _stub_rollout_with_batch(t, rewards=batch["rewards"], batch=batch)

    t.fit()  # must NOT raise

    assert t._ref_policy is not None
    assert t._ref_policy.lora_base_as_ref is True


# ===========================================================================
# Registry + constructor-config invariants (merged from
# tests/test_trainer_grpo.py)
# ===========================================================================


def test_grpo_resolves_from_registry():
    """The 'grpo' trainer name resolves to GRPOTrainer through the registry."""
    from lighttrain.registry import get as resolve

    assert resolve("trainer", "grpo") is GRPOTrainer


def test_grpo_step_returns_finite_loss():
    """A hand-crafted GRPO minibatch produces a finite 'loss' metric."""
    trainer = _make_grpo()
    metrics = trainer._grpo_step(_grpo_batch())
    assert "loss" in metrics
    assert math.isfinite(float(metrics["loss"]))


def test_grpo_group_size_stored_on_trainer():
    """group_size passed to the constructor is retained as an attribute."""
    trainer = _make_grpo(group_size=2)
    assert trainer.group_size == 2


def test_grpo_clip_eps_propagated_to_default_loss():
    """clip_eps feeds the default RL loss used when the recipe omits a `loss:`
    block (the loss: seam wins when present — keystone step 3)."""
    trainer = _make_grpo(clip_eps=0.15)
    assert trainer._default_loss.clip_eps == 0.15
