"""Adversarial tests for RLUpdateRule — backward/clip/step rule for
GRPO/PPO/Preference trainers.

Key contracts (different from StandardUpdateRule):
  - Does NOT call ``model(**batch)`` — trainer is responsible for forward.
  - Has no gradient accumulation, no RETRY_STEP.
  - Reads RL-specific tensors (log_probs_new/old, advantages, …) from
    ``ctx.extras`` (NOT from the batch dict).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.builtin_plugins.update_rules.rl import RLUpdateRule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _OrderedRecorder:
    """Records the temporal order of bus events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def _h(name):
        def _f(self, **_kw):
            self.events.append(name)

        return _f

    on_step_begin = _h("on_step_begin")
    on_forward_pre = _h("on_forward_pre")
    on_forward_post = _h("on_forward_post")
    on_loss_computed = _h("on_loss_computed")
    on_backward_pre = _h("on_backward_pre")
    on_backward_post = _h("on_backward_post")
    on_clip_grad = _h("on_clip_grad")
    on_optimizer_step_pre = _h("on_optimizer_step_pre")
    on_optimizer_step_post = _h("on_optimizer_step_post")
    on_zero_grad = _h("on_zero_grad")
    on_scheduler_step = _h("on_scheduler_step")
    on_step_end = _h("on_step_end")


class _SpyModel(nn.Module):
    """Wraps a tiny Linear AND records every forward call so we can pin
    that the RLUpdateRule does NOT call ``model(**batch)``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)
        self.forward_calls = 0

    def forward(self, **kw):
        self.forward_calls += 1
        x = kw.get("x", torch.randn(2, 4))
        return ModelOutput(outputs={"logits": self.linear(x)})


def _build_ctx(
    *,
    callbacks: list | None = None,
    loss_fn=None,
    accelerator=None,
    grad_sync=None,
    parallel_ctx=None,
    scheduler=None,
) -> tuple[StepContext, _SpyModel, torch.optim.Optimizer]:
    model = _SpyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=EventBus(callbacks or []),
        loss_fn=loss_fn,
        scheduler=scheduler,
        accelerator=accelerator,
        grad_sync=grad_sync,
        parallel_ctx=parallel_ctx,
    )
    return ctx, model, optim


def _grad_loss_fn(model_output, batch, ctx):
    """Compute a real-gradient loss using model parameters from ctx.extras.

    We pull the model directly so backward gives us a real grad.
    """
    model = ctx.extras["model"]
    return {"loss": (model.linear.weight ** 2).sum()}


# ===========================================================================
# lifecycle order
# ===========================================================================


def test_rl_full_step_emits_events_in_exact_order():
    """Goal: pin the exact event sequence for a successful RL step.

    Input: scheduler with step_per_batch=True.
    Expected list:
      [on_step_begin, on_loss_computed, on_backward_pre, on_backward_post,
       on_clip_grad, on_optimizer_step_pre, on_optimizer_step_post,
       on_zero_grad, on_scheduler_step, on_step_end]

    Note: RLUpdateRule does NOT fire on_forward_pre / on_forward_post —
    trainer owns the forward, rule only owns backward+optimizer.

    Catches event reordering AND a refactor that adds spurious
    on_forward_post events (would conflate the trainer's contract with
    the rule's).
    """
    rec = _OrderedRecorder()
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    ctx, model, _ = _build_ctx(
        callbacks=[rec], loss_fn=_grad_loss_fn, scheduler=scheduler
    )
    ctx.extras["model"] = model

    RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)

    expected = [
        "on_step_begin",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
        "on_scheduler_step",
        "on_step_end",
    ]
    assert rec.events == expected


def test_rl_does_not_call_model_forward():
    """Goal: pin the contract — RLUpdateRule must NOT invoke
    ``model(**batch)`` (the trainer already did the forward).

    Input: spy model whose forward increments a counter. loss_fn doesn't
    invoke the model either.
    Expected: model.forward_calls == 0 after step.

    Catches a refactor that "harmonizes" RL with Standard by adding a
    forward pass — would compute log_probs twice and corrupt PPO ratios.
    """
    ctx, model, _ = _build_ctx(loss_fn=_grad_loss_fn)
    ctx.extras["model"] = model

    RLUpdateRule().step(model, {}, ctx)

    assert model.forward_calls == 0


def test_rl_step_reads_log_probs_from_ctx_extras():
    """Goal: pin the contract that PPO/GRPO log-probs flow through
    ``ctx.extras["log_probs_new"]`` / ``["log_probs_old"]``, NOT the
    batch dict.

    Positive sub-test: a stub loss_fn asserts both keys are present and
    equal to the tensors the (mock) trainer pushed.

    Negative sub-test: a stub loss_fn that reads ``ctx.extras["missing_key"]``
    directly raises KeyError — the error surfaces unchanged to the caller
    (not wrapped or swallowed).
    """
    # Positive case
    lp_new = torch.randn(2, 4, requires_grad=True)
    lp_old = torch.randn(2, 4)

    seen: dict[str, torch.Tensor] = {}

    def _reader_loss(_out, _batch, ctx):
        seen["new"] = ctx.extras["log_probs_new"]
        seen["old"] = ctx.extras["log_probs_old"]
        # produce a real graph so backward works
        return {"loss": (ctx.extras["log_probs_new"] ** 2).sum()}

    ctx, model, _ = _build_ctx(loss_fn=_reader_loss)
    ctx.extras["model"] = model
    ctx.extras["log_probs_new"] = lp_new
    ctx.extras["log_probs_old"] = lp_old

    RLUpdateRule().step(model, {}, ctx)

    torch.testing.assert_close(seen["new"], lp_new, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(seen["old"], lp_old, atol=1e-5, rtol=1e-4)

    # Negative case — missing key raises clearly
    def _bad_reader_loss(_out, _batch, ctx):
        return {"loss": ctx.extras["missing_key"]}

    ctx2, model2, _ = _build_ctx(loss_fn=_bad_reader_loss)
    ctx2.extras["model"] = model2

    with pytest.raises(KeyError, match="missing_key"):
        RLUpdateRule().step(model2, {}, ctx2)


# ===========================================================================
# skip path
# ===========================================================================


class _SkipperCb:
    def on_loss_computed(self, **_):
        return Signal.SKIP_STEP


class _StopperCb:
    def on_loss_computed(self, **_):
        return Signal.STOP_TRAINING


def test_rl_skip_path_no_backward_no_optimizer_step():
    """Goal: SKIP_STEP from on_loss_computed must abort both backward and
    optimizer.step.

    Input: a loss whose .backward raises — proves backward isn't called.
    Expected: clean return, no exception.

    Catches a refactor that swallows the skip signal silently.
    """

    class _ExplodingLoss:
        def __init__(self):
            self._t = torch.tensor(0.5)

        def detach(self):
            return self._t.detach()

        def item(self):
            return 0.5

        def backward(self):
            raise RuntimeError("backward must not be called on skip path")

    def _loss(_out, _batch, _ctx):
        return {"loss": _ExplodingLoss()}

    ctx, model, optim = _build_ctx(callbacks=[_SkipperCb()], loss_fn=_loss)
    ctx.extras["model"] = model

    step_spy = MagicMock(wraps=optim.step)
    optim.step = step_spy

    metrics = RLUpdateRule().step(model, {}, ctx)

    assert metrics["skipped"] == 1.0
    step_spy.assert_not_called()


def test_rl_step_end_always_fires_on_skip():
    """Goal: on_step_end must fire on the skip path too — downstream loggers
    rely on it.

    Catches a refactor that drops the on_step_end dispatch inside the skip
    early-return at lines 92-100.
    """
    rec = _OrderedRecorder()
    ctx, model, _ = _build_ctx(
        callbacks=[rec, _SkipperCb()], loss_fn=_grad_loss_fn
    )
    ctx.extras["model"] = model

    RLUpdateRule().step(model, {}, ctx)

    assert "on_step_end" in rec.events
    # And the skip-path events ONLY (no backward / clip / optimizer)
    forbidden = {
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
    }
    assert forbidden.isdisjoint(set(rec.events))


def test_rl_stop_training_surfaces_via_ctx_extras():
    """Goal: STOP_TRAINING signal surfaces via ctx.extras['loss_signal']
    as ``int(Signal.STOP_TRAINING)``.

    Catches a refactor that drops line 90 ``ctx.extras["loss_signal"] = int(sig)``
    — trainer's outer fit loop relies on this int to break the loop.
    """
    ctx, model, _ = _build_ctx(callbacks=[_StopperCb()], loss_fn=_grad_loss_fn)
    ctx.extras["model"] = model

    RLUpdateRule().step(model, {}, ctx)

    assert ctx.extras["loss_signal"] == int(Signal.STOP_TRAINING)


# ===========================================================================
# three-way backward branch
# ===========================================================================


def test_rl_three_way_branch_grad_sync(fake_dist_env):
    """grad_sync wins over both other options; backward / clip / optimizer_step
    all route through it.

    Input: ctx.grad_sync = fake recording obj.
    Expected: grad_sync recorded backward + clip_grad_norm + optimizer_step.
    """
    grad_sync, pctx = fake_dist_env()
    ctx, model, _ = _build_ctx(
        loss_fn=_grad_loss_fn, grad_sync=grad_sync, parallel_ctx=pctx
    )
    ctx.extras["model"] = model

    RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)

    names = [c[0] for c in grad_sync.calls]
    assert "backward" in names
    assert "clip_grad_norm" in names
    assert "optimizer_step" in names


def test_rl_three_way_branch_accelerator(fake_accelerator):
    """accelerator path is used when grad_sync is None and accelerator
    has 'backward' attribute.

    Input: ctx.accelerator = fake_accelerator (has backward + clip_grad_norm_).
    Expected: fake_accelerator.calls contains 'backward' + 'clip_grad_norm_'.
    """
    ctx, model, _ = _build_ctx(loss_fn=_grad_loss_fn, accelerator=fake_accelerator)
    ctx.extras["model"] = model

    RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)

    names = [c[0] for c in fake_accelerator.calls]
    assert "backward" in names
    assert "clip_grad_norm_" in names


def test_rl_three_way_branch_bare():
    """Bare path: no grad_sync, no accelerator → ``loss.backward()`` direct.
    Verified by params updating (no exception, weights change).
    """
    ctx, model, _ = _build_ctx(loss_fn=_grad_loss_fn)
    ctx.extras["model"] = model
    before = model.linear.weight.detach().clone()

    RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)
    after = model.linear.weight.detach().clone()

    assert not torch.equal(before, after)


# ===========================================================================
# scheduler gate
# ===========================================================================


@pytest.mark.parametrize("step_per_batch,expected", [(True, 1), (False, 0)])
def test_rl_scheduler_step_per_batch_gate(step_per_batch: bool, expected: int):
    """Goal: scheduler.step fires only when step_per_batch=True.

    Catches a refactor that drops the gate at line 143.
    """
    scheduler = MagicMock()
    scheduler.step_per_batch = step_per_batch
    ctx, model, _ = _build_ctx(loss_fn=_grad_loss_fn, scheduler=scheduler)
    ctx.extras["model"] = model

    RLUpdateRule().step(model, {}, ctx)

    assert scheduler.step.call_count == expected


# ===========================================================================
# RL regression: no RETRY_STEP
# ===========================================================================


def test_rl_regression_no_retry_step_on_rl_path():
    """Goal: RLUpdateRule has NO retry loop. If a callback returns RETRY_STEP,
    the rule must not re-invoke on_loss_computed (no retry semantics).

    Input: callback returns RETRY_STEP from on_loss_computed.
    Expected: on_loss_computed fires exactly once; backward still happens
    (RETRY isn't in the skip set on line 88).

    Catches a refactor that copy-pastes the retry loop from StandardUpdateRule
    — RLUpdateRule explicitly omits it because RL forward is owned by the
    trainer (re-running forward would require trainer cooperation).
    """
    loss_computed_count = [0]

    class _Retrier:
        def on_loss_computed(self, **_):
            loss_computed_count[0] += 1
            return Signal.RETRY_STEP

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()], loss_fn=_grad_loss_fn)
    ctx.extras["model"] = model

    RLUpdateRule().step(model, {}, ctx)

    assert loss_computed_count[0] == 1


# ===========================================================================
# state_dict
# ===========================================================================


def test_rl_state_dict_roundtrip():
    """Pin grad_clip is the only field persisted (vs Standard which also
    persists micro_step).
    """
    rule = RLUpdateRule(grad_clip=0.7)
    sd = rule.state_dict()
    assert sd == {"grad_clip": 0.7}

    rule2 = RLUpdateRule(grad_clip=9.9)
    rule2.load_state_dict(sd)
    assert rule2.grad_clip == 0.7
