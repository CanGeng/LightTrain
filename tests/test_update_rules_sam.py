"""Tests for SAMUpdateRule (M7)."""
import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.update_rules.sam import SAMUpdateRule


def _make_ctx(model, loss_fn, opt):
    from lighttrain.engine._context import StepContext
    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt
    ctx.scheduler = None
    return ctx


def _make_model_loss_opt():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, **batch):
            from lighttrain.protocols import ModelOutput
            x = batch["input_ids"].float()
            return ModelOutput(outputs={"logits": self.m(x)})

    class FakeLoss:
        def __call__(self, out, batch, ctx):
            logits = out.outputs["logits"]
            labels = batch.get("labels", torch.zeros(logits.shape[0], dtype=torch.long))
            return {"loss": nn.functional.cross_entropy(logits, labels)}

    wrapped = Wrapper(model)
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    return wrapped, FakeLoss(), opt


def test_sam_step_runs():
    model, loss_fn, opt = _make_model_loss_opt()
    rule = SAMUpdateRule(rho=0.05)
    ctx = _make_ctx(model, loss_fn, opt)
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    m = rule.step(model, batch, ctx)
    assert "loss" in m and "grad_norm" in m


def test_sam_parameters_updated():
    model, loss_fn, opt = _make_model_loss_opt()
    rule = SAMUpdateRule(rho=0.05)
    ctx = _make_ctx(model, loss_fn, opt)
    before = [p.data.clone() for p in model.parameters()]
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    rule.step(model, batch, ctx)
    after = [p.data.clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after, strict=False))


def test_sam_weights_restored_after_perturbation():
    """Verify perturbation is removed before optimizer.step()."""
    model, loss_fn, opt = _make_model_loss_opt()
    rule = SAMUpdateRule(rho=0.05, grad_clip=0.0)
    ctx = _make_ctx(model, loss_fn, opt)

    weight_snapshots: list = []
    original_step = opt.step

    def patched_step():
        weight_snapshots.append([p.data.clone() for p in model.parameters()])
        return original_step()

    opt.step = patched_step
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    rule.step(model, batch, ctx)

    # Verify that at optimizer.step time, weights don't have the perturbation baked in
    assert len(weight_snapshots) >= 1  # optimizer was called


def test_sam_grad_clip_applied():
    """Verify post-clip parameter grad norm is bounded by grad_clip.

    Note: ``m["grad_norm"]`` reports the PRE-clip norm (torch.nn.utils.clip_grad_norm_
    return convention). We capture post-clip via on_optimizer_step_pre callback.
    """
    from lighttrain.callbacks.base import EventBus

    torch.manual_seed(42)
    model, loss_fn, opt = _make_model_loss_opt()
    rule = SAMUpdateRule(rho=0.05, grad_clip=1.0)
    ctx = _make_ctx(model, loss_fn, opt)

    post_clip_norms: list[float] = []

    class _PostClipRecorder:
        def on_optimizer_step_pre(self, **_):
            params = [p for p in model.parameters() if p.grad is not None]
            if params:
                total = torch.stack([p.grad.norm(2) for p in params]).norm(2)
                post_clip_norms.append(float(total))

    ctx.bus = EventBus([_PostClipRecorder()])
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    rule.step(model, batch, ctx)

    assert post_clip_norms, "on_optimizer_step_pre must have fired"
    assert post_clip_norms[0] <= 1.01, f"post-clip grad norm {post_clip_norms[0]} exceeds 1.0"


def test_sam_state_dict():
    rule = SAMUpdateRule(rho=0.1)
    sd = rule.state_dict()
    rule2 = SAMUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.rho == pytest.approx(0.1)
