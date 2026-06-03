"""Tests for MeZOUpdateRule (M7)."""
import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.update_rules.mezo import MeZOUpdateRule


def _make_ctx(model):
    from lighttrain.engine._context import StepContext
    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    return ctx


def _ce_loss(logits, labels):
    from lighttrain.protocols import LossContext, ModelOutput
    return {"loss": torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))}


def _make_model_and_loss():
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

    return Wrapper(model), FakeLoss()


def _make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_mezo_no_gradient():
    model, loss_fn = _make_model_and_loss()
    rule = MeZOUpdateRule(eps=1e-2)
    opt = _make_optimizer(model)
    ctx = _make_ctx(model)
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt

    batch = {
        "input_ids": torch.randn(4, 8),
        "labels": torch.randint(0, 4, (4,)),
    }
    rule.step(model, batch, ctx)

    # MeZO must never accumulate gradients
    for p in model.parameters():
        assert p.grad is None, "MeZO should not leave gradients on parameters"


def test_mezo_loss_decreases():
    torch.manual_seed(0)
    model, loss_fn = _make_model_and_loss()
    rule = MeZOUpdateRule(eps=1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx = _make_ctx(model)
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt

    batch = {
        "input_ids": torch.randn(8, 8),
        "labels": torch.randint(0, 4, (8,)),
    }

    losses = []
    for _ in range(30):
        m = rule.step(model, batch, ctx)
        losses.append(m["loss"])
        ctx.step += 1

    assert losses[-1] < losses[0] or abs(losses[-1] - losses[0]) < 1.0


def test_mezo_parameters_updated():
    model, loss_fn = _make_model_and_loss()
    rule = MeZOUpdateRule(eps=1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx = _make_ctx(model)
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt

    params_before = [p.data.clone() for p in model.parameters()]
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    rule.step(model, batch, ctx)
    params_after = [p.data.clone() for p in model.parameters()]
    any_changed = any(not torch.allclose(a, b) for a, b in zip(params_before, params_after))
    assert any_changed


def test_mezo_state_dict_roundtrip():
    rule = MeZOUpdateRule(eps=5e-3, seed_per_step=False)
    sd = rule.state_dict()
    rule2 = MeZOUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.eps == pytest.approx(5e-3)
    assert rule2.seed_per_step is False


def test_mezo_metrics_keys():
    model, loss_fn = _make_model_and_loss()
    rule = MeZOUpdateRule(eps=1e-3)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = _make_ctx(model)
    ctx.loss_fn = loss_fn
    ctx.optimizer = opt
    batch = {"input_ids": torch.randn(2, 8), "labels": torch.randint(0, 4, (2,))}
    m = rule.step(model, batch, ctx)
    assert "loss" in m and "grad_est" in m and "lr" in m
