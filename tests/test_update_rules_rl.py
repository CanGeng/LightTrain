"""RLUpdateRule tests — backward/callback/SKIP_STEP/grad_clip."""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.update_rules.rl import RLUpdateRule
from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext

# ---- helpers ----------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_ctx(*, callbacks=None, loss_fn=None) -> StepContext:
    ctx = StepContext()
    ctx.bus = EventBus(callbacks or [])
    model = _TinyModel()
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx.model = model
    if loss_fn is not None:
        ctx.loss_fn = loss_fn
    return ctx


def _simple_loss_fn(model):
    """Returns a loss_fn that computes MSE on random data — has a gradient."""
    def _fn(model_output, batch, ctx):
        x = torch.randn(2, 4)
        pred = ctx.extras.get("model", model)(x)
        loss = (pred - 1.0).pow(2).mean()
        return {"loss": loss, "mse": loss}
    return _fn


# ---- registration -----------------------------------------------------------

def test_rl_update_rule_registers():
    from lighttrain.registry import get as resolve
    assert resolve("update_rule", "rl") is RLUpdateRule


# ---- basic step -------------------------------------------------------------

def test_rl_rule_basic_step_returns_metrics():
    model = _TinyModel()
    ctx = _make_ctx()
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx.loss_fn = _simple_loss_fn(model)
    rule = RLUpdateRule(grad_clip=1.0)
    metrics = rule.step(model, {}, ctx)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(float(metrics["loss"])))


# ---- callback sequence ------------------------------------------------------

def test_rl_rule_fires_full_callback_chain():
    fired = []

    class _Recorder:
        def on_step_begin(self, **kw): fired.append("on_step_begin")
        def on_loss_computed(self, **kw): fired.append("on_loss_computed")
        def on_backward_pre(self, **kw): fired.append("on_backward_pre")
        def on_backward_post(self, **kw): fired.append("on_backward_post")
        def on_clip_grad(self, **kw): fired.append("on_clip_grad")
        def on_optimizer_step_pre(self, **kw): fired.append("on_optimizer_step_pre")
        def on_optimizer_step_post(self, **kw): fired.append("on_optimizer_step_post")
        def on_zero_grad(self, **kw): fired.append("on_zero_grad")
        def on_step_end(self, **kw): fired.append("on_step_end")

    model = _TinyModel()
    ctx = _make_ctx(callbacks=[_Recorder()])
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx.loss_fn = _simple_loss_fn(model)
    RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)

    for event in [
        "on_step_begin", "on_loss_computed", "on_backward_pre", "on_backward_post",
        "on_clip_grad", "on_optimizer_step_pre", "on_optimizer_step_post",
        "on_zero_grad", "on_step_end",
    ]:
        assert event in fired, f"{event} not fired"


# ---- SKIP_STEP --------------------------------------------------------------

def test_rl_rule_skip_step_skips_backward():
    backward_called = []

    class _SkipCallback:
        def on_loss_computed(self, **kw):
            return Signal.SKIP_STEP

    class _TrackLoss:
        def backward(self):
            backward_called.append(True)

    model = _TinyModel()
    ctx = _make_ctx(callbacks=[_SkipCallback()])
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def _loss_fn(model_output, batch, ctx):
        _TrackLoss()
        # return a real tensor so float() works
        return {"loss": torch.tensor(0.5)}

    ctx.loss_fn = _loss_fn
    metrics = RLUpdateRule(grad_clip=1.0).step(model, {}, ctx)
    assert not backward_called, "backward must not be called on SKIP_STEP"
    assert float(metrics.get("skipped", 0)) == 1.0


# ---- grad_clip --------------------------------------------------------------

def test_rl_rule_grad_clip_limits_norm():
    """With a very small grad_clip, clipping must fire and the actual param grads
    must be scaled down. Note: clip_grad_norm_ reports the PRE-clip norm."""
    reported_norms = []
    grad_after = []

    class _NormRecorder:
        def on_clip_grad(self, grad_norm=0.0, **kw):
            reported_norms.append(grad_norm)
        def on_optimizer_step_pre(self, **kw):
            # capture grad norm AFTER clipping, BEFORE optimizer.step
            g = model.linear.weight.grad
            if g is not None:
                grad_after.append(float(g.norm()))

    model = _TinyModel()
    ctx = _make_ctx(callbacks=[_NormRecorder()])
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def _big_loss_fn(model_output, batch, ctx):
        loss = (ctx.extras.get("model", model).linear.weight * 1000).sum()
        return {"loss": loss}

    ctx.loss_fn = _big_loss_fn
    RLUpdateRule(grad_clip=0.01).step(model, {}, ctx)
    assert reported_norms, "on_clip_grad must have fired"
    # pre-clip norm is large (reported); actual param grad norm is ≤ grad_clip
    assert reported_norms[0] > 0.01
    assert grad_after and grad_after[0] <= 0.011  # actual grad is clipped


# ---- grad_clip=0 skips clipping ---------------------------------------------

def test_rl_rule_grad_clip_zero_disables_clipping():
    """grad_clip=0 must skip clipping entirely; grad_norm reported as 0.0."""
    reported_norms = []

    class _NormRecorder:
        def on_clip_grad(self, grad_norm=0.0, **kw):
            reported_norms.append(grad_norm)

    model = _TinyModel()
    ctx = _make_ctx(callbacks=[_NormRecorder()])
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx.loss_fn = _simple_loss_fn(model)
    RLUpdateRule(grad_clip=0.0).step(model, {}, ctx)
    assert reported_norms[0] == 0.0


# ---- state_dict roundtrip ---------------------------------------------------

def test_rl_rule_state_dict_roundtrip():
    rule = RLUpdateRule(grad_clip=0.5)
    sd = rule.state_dict()
    rule2 = RLUpdateRule(grad_clip=9.9)
    rule2.load_state_dict(sd)
    assert rule2.grad_clip == 0.5
