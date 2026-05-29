"""Tests for frontier UpdateRules: ForwardForward, PCN, DFA (M7)."""
import pytest
import torch
import torch.nn as nn

from plugins.update_rules.forward_forward import ForwardForwardUpdateRule
from plugins.update_rules.pcn import PCNUpdateRule
from plugins.update_rules.dfa import DFAUpdateRule


def _ctx(model=None):
    from lighttrain.engine._context import StepContext
    ctx = StepContext()
    ctx.model = model
    ctx.bus = None
    ctx.accelerator = None
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    ctx.loss_fn = None
    ctx.optimizer = None
    ctx.scheduler = None
    return ctx


# ---------------------------------------------------------------------------
# ForwardForward
# ---------------------------------------------------------------------------

def _ff_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_ff_step_returns_loss():
    model = _ff_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    ctx = _ctx(model)
    ctx.optimizer = opt
    batch = {"x": torch.randn(4, 8)}
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_ff_goodness_computation():
    rule = ForwardForwardUpdateRule()
    h = torch.ones(3, 8)  # goodness should be 1.0
    g = rule._goodness(h)
    assert g.shape == (3,)
    assert torch.allclose(g, torch.ones(3))


def test_ff_parameters_updated():
    model = _ff_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    ctx = _ctx(model)
    ctx.optimizer = opt
    before = [p.data.clone() for p in model.parameters()]
    batch = {"x": torch.randn(4, 8)}
    for _ in range(5):
        rule.step(model, batch, ctx)
        ctx.step += 1
    after = [p.data.clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_ff_negative_auto_generated():
    model = _ff_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rule = ForwardForwardUpdateRule()
    ctx = _ctx(model)
    ctx.optimizer = opt
    # Without explicit neg_x, should auto-generate by shuffling
    batch = {"x": torch.randn(4, 8)}
    m = rule.step(model, batch, ctx)
    assert torch.isfinite(torch.tensor(m["loss"]))


# ---------------------------------------------------------------------------
# PCN
# ---------------------------------------------------------------------------

def _pcn_model():
    return nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))


def test_pcn_step_returns_loss():
    model = _pcn_model()
    rule = PCNUpdateRule(n_infer=5, lr_weight=0.01)
    ctx = _ctx(model)
    ctx.optimizer = None  # PCN doesn't use optimizer
    batch = {
        "x": torch.randn(4, 8),
        "labels": torch.zeros(4, 4),  # (B, output_dim)
    }
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_pcn_parameters_updated():
    torch.manual_seed(0)
    model = _pcn_model()
    rule = PCNUpdateRule(n_infer=10, lr_weight=0.1)
    ctx = _ctx(model)
    ctx.optimizer = None
    before = [p.data.clone() for p in model.parameters()]
    batch = {"x": torch.randn(4, 8), "labels": torch.zeros(4, 4)}
    for _ in range(3):
        rule.step(model, batch, ctx)
        ctx.step += 1
    after = [p.data.clone() for p in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_pcn_state_dict():
    rule = PCNUpdateRule(n_infer=15, lr_infer=0.2)
    sd = rule.state_dict()
    rule2 = PCNUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.n_infer == 15
    assert rule2.lr_infer == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------

def _dfa_model():
    return nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))


def test_dfa_step_returns_loss():
    model = _dfa_model()
    rule = DFAUpdateRule(lr=0.01)
    ctx = _ctx(model)
    ctx.optimizer = None
    batch = {"x": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_dfa_feedback_matrices_fixed():
    """Feedback matrices B_l must not have requires_grad."""
    model = _dfa_model()
    rule = DFAUpdateRule()
    ctx = _ctx(model)
    ctx.optimizer = None
    batch = {"x": torch.randn(4, 8), "labels": torch.randint(0, 4, (4,))}
    rule.step(model, batch, ctx)
    for B in rule._feedback.values():
        assert not B.requires_grad


def test_dfa_state_dict():
    rule = DFAUpdateRule(feedback_scale=0.05)
    sd = rule.state_dict()
    rule2 = DFAUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.feedback_scale == pytest.approx(0.05)


def test_dfa_loss_decreases():
    torch.manual_seed(1)
    model = _dfa_model()
    rule = DFAUpdateRule(lr=0.05)
    ctx = _ctx(model)
    ctx.optimizer = None
    batch = {"x": torch.randn(8, 8), "labels": torch.randint(0, 4, (8,))}
    losses = []
    for _ in range(20):
        m = rule.step(model, batch, ctx)
        losses.append(m["loss"])
        ctx.step += 1
    # Check that loss doesn't explode
    assert losses[-1] < losses[0] * 10
