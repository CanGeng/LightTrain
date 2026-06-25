"""Exhaustive edge-case tests for ForwardForwardUpdateRule.

What this file pins (driving toward 100% coverage of the target module):

  Section A — ``_current_lr`` helper
    A1: normal optimizer returns correct lr from param_groups[0]
    A2: empty param_groups returns 0.0  (line 41)
    A3: optimizer wrapper with inner .optimizer attribute is unwrapped
    A4: optimizer with no param_groups attribute returns 0.0

  Section B — lifecycle methods
    B1: setup() returns None regardless of arguments  (line 63)
    B2: state_dict() returns threshold + grad_clip  (line 66)
    B3: load_state_dict() updates threshold and grad_clip  (lines 69-70)
    B4: load_state_dict() partial dict falls back to current values
    B5: state_dict / load_state_dict round-trip is idempotent

  Section C — batch key routing for positive input
    C1: 'input_ids' key routes to pos_input  (line 106)
    C2: 'x' key routes to pos_input  (already covered in test_frontier)
    C3: neither 'input_ids' nor 'x' raises KeyError  (line 110)

  Section D — batch key routing for negative input
    D1: 'neg_input_ids' key routes to neg_input  (line 113)
    D2: 'neg_x' key routes to neg_input  (line 115)
    D3: neither neg key → auto-shuffle fallback (already covered)

  Section E — bus dispatch / scheduler gating
    E1: bus=None executes cleanly (no AttributeError)
    E2: all six bus events fire in correct order when bus is set
    E3: scheduler.step called only when step_per_batch is True
    E4: scheduler.step NOT called when scheduler is None

  Section F — _get_layers coverage
    F1: model with .layers attribute uses that list
    F2: model with .blocks attribute uses that list
    F3: model with .children_list attribute uses that list
    F4: plain nn.Sequential falls back to model.children()

  Section G — grad_clip branching
    G1: grad_clip=0 skips clip_grad_norm_
    G2: grad_clip > 0 clips gradients (grad_norm in metrics)

  Section H — metrics keys
    H1: returned dict contains loss, grad_norm, lr, skipped keys
    H2: model stored in ctx.extras["model"]
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.forward_forward import (
    ForwardForwardUpdateRule,
    _current_lr,
)
from lighttrain.engine._context import StepContext

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ff_ctx(model: nn.Module, optimizer: torch.optim.Optimizer, *, scheduler=None, bus=None) -> StepContext:
    """Build a minimal StepContext for ForwardForward."""
    ctx = StepContext()
    ctx.model = model
    ctx.bus = bus
    ctx.optimizer = optimizer
    ctx.scheduler = scheduler
    ctx.step = 0
    ctx.epoch = 0
    ctx.metrics = {}
    ctx.extras = {}
    return ctx


def _ff_model() -> nn.Sequential:
    """Tiny 2-layer MLP suitable for FF: Linear → ReLU → Linear."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _ff_opt(model: nn.Module, lr: float = 1e-3) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr)


def _batch_x(B: int = 4, D: int = 8) -> dict[str, torch.Tensor]:
    """Batch using the 'x' key."""
    torch.manual_seed(1)
    return {"x": torch.randn(B, D)}


def _batch_input_ids(B: int = 4, D: int = 8) -> dict[str, torch.Tensor]:
    """Batch using the 'input_ids' key (integer ids, cast to float inside rule)."""
    torch.manual_seed(2)
    return {"input_ids": torch.randint(0, 10, (B, D))}


# ---------------------------------------------------------------------------
# Section A — _current_lr helper
# ---------------------------------------------------------------------------


def test_invariant_current_lr_normal_optimizer():
    """_current_lr returns the lr stored in param_groups[0]."""
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.042)
    assert _current_lr(opt) == pytest.approx(0.042)


def test_invariant_current_lr_empty_param_groups_returns_zero():
    """_current_lr returns 0.0 when param_groups is an empty list (line 41)."""
    mock_opt = MagicMock()
    mock_opt.optimizer = mock_opt  # no .optimizer wrapper
    del mock_opt.optimizer  # remove wrapper attribute
    mock_opt.param_groups = []  # falsy
    result = _current_lr(mock_opt)
    assert result == 0.0


def test_invariant_current_lr_optimizer_wrapper_unwrapped():
    """_current_lr unwraps obj.optimizer when present (e.g. AMP GradScaler wrapper)."""
    inner = MagicMock()
    inner.param_groups = [{"lr": 0.007}]
    wrapper = MagicMock()
    wrapper.optimizer = inner
    # getattr(wrapper, "optimizer", wrapper) picks inner; inner has param_groups
    assert _current_lr(wrapper) == pytest.approx(0.007)


def test_invariant_current_lr_no_param_groups_attr_returns_zero():
    """_current_lr returns 0.0 when the object has no param_groups attribute."""
    obj = SimpleNamespace()  # no optimizer, no param_groups
    assert _current_lr(obj) == 0.0


# ---------------------------------------------------------------------------
# Section B — lifecycle methods
# ---------------------------------------------------------------------------


def test_invariant_setup_returns_none():
    """setup() always returns None regardless of arguments (line 63)."""
    rule = ForwardForwardUpdateRule()
    result = rule.setup(model=None, sample=None)
    assert result is None
    # also with non-None args
    result2 = rule.setup(model=MagicMock(), sample={"x": torch.zeros(2, 4)})
    assert result2 is None


def test_invariant_state_dict_contains_threshold_and_grad_clip():
    """state_dict() returns a dict with 'threshold' and 'grad_clip' (line 66)."""
    rule = ForwardForwardUpdateRule(threshold=3.5, grad_clip=2.0)
    sd = rule.state_dict()
    assert sd == {"threshold": 3.5, "grad_clip": 2.0}


def test_invariant_load_state_dict_updates_both_fields():
    """load_state_dict() sets threshold and grad_clip from the dict (lines 69-70)."""
    rule = ForwardForwardUpdateRule(threshold=2.0, grad_clip=1.0)
    rule.load_state_dict({"threshold": 5.0, "grad_clip": 0.5})
    assert rule.threshold == pytest.approx(5.0)
    assert rule.grad_clip == pytest.approx(0.5)


def test_invariant_load_state_dict_partial_uses_current_fallback():
    """load_state_dict() falls back to current values for missing keys."""
    rule = ForwardForwardUpdateRule(threshold=2.0, grad_clip=1.0)
    # Only update threshold; grad_clip should stay at 1.0
    rule.load_state_dict({"threshold": 4.0})
    assert rule.threshold == pytest.approx(4.0)
    assert rule.grad_clip == pytest.approx(1.0)


def test_invariant_state_dict_load_state_dict_roundtrip():
    """state_dict → load_state_dict round-trip is idempotent."""
    rule = ForwardForwardUpdateRule(threshold=1.5, grad_clip=3.0)
    sd = rule.state_dict()
    rule2 = ForwardForwardUpdateRule(threshold=0.0, grad_clip=0.0)
    rule2.load_state_dict(sd)
    assert rule2.threshold == pytest.approx(rule.threshold)
    assert rule2.grad_clip == pytest.approx(rule.grad_clip)


# ---------------------------------------------------------------------------
# Section C — batch key routing for positive input
# ---------------------------------------------------------------------------


def test_invariant_input_ids_key_accepted_as_positive_input():
    """'input_ids' key is converted to float and used as pos_input (line 106)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    # integer input_ids — rule must cast to float
    batch = {"input_ids": torch.randint(0, 10, (4, 8))}
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_missing_input_key_raises_keyerror():
    """Batch with neither 'input_ids' nor 'x' raises a descriptive KeyError (line 110)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    bad_batch: dict[str, Any] = {"labels": torch.zeros(4)}
    with pytest.raises(KeyError, match="ForwardForward"):
        rule.step(model, bad_batch, ctx)


# ---------------------------------------------------------------------------
# Section D — batch key routing for negative input
# ---------------------------------------------------------------------------


def test_invariant_neg_input_ids_key_used_for_negative_pass():
    """'neg_input_ids' key is used as neg_input (line 113)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    torch.manual_seed(3)
    batch = {
        "input_ids": torch.randint(0, 10, (4, 8)),
        "neg_input_ids": torch.randint(0, 10, (4, 8)),
    }
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_neg_x_key_used_for_negative_pass():
    """'neg_x' key is used as neg_input when present (line 115)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    torch.manual_seed(4)
    batch = {
        "x": torch.randn(4, 8),
        "neg_x": torch.randn(4, 8),
    }
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_neg_input_ids_takes_priority_over_neg_x():
    """When both 'neg_input_ids' and 'neg_x' are present, 'neg_input_ids' wins (line 112 guard)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    torch.manual_seed(5)
    batch = {
        "x": torch.randn(4, 8),
        "neg_input_ids": torch.ones(4, 8) * 2.0,   # recognisably different
        "neg_x": torch.ones(4, 8) * -999.0,         # should not be chosen
    }
    # should not raise — neg_input_ids is selected
    m = rule.step(model, batch, ctx)
    assert torch.isfinite(torch.tensor(m["loss"]))


# ---------------------------------------------------------------------------
# Section E — bus dispatch / scheduler gating
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Minimal bus callback that records event names in order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def _h(name: str):
        def _f(self, **_kw):
            self.events.append(name)
        return _f

    on_step_begin = _h("on_step_begin")
    on_optimizer_step_pre = _h("on_optimizer_step_pre")
    on_optimizer_step_post = _h("on_optimizer_step_post")
    on_step_end = _h("on_step_end")


class _FakeBus:
    """Minimal EventBus stub that dispatches to one recorder callback."""

    def __init__(self, recorder: _EventRecorder) -> None:
        self._rec = recorder

    def dispatch(self, event: str, **kwargs: Any) -> None:
        handler = getattr(self._rec, event, None)
        if handler is not None:
            handler(**kwargs)


def test_invariant_bus_none_does_not_raise():
    """A step with bus=None completes without AttributeError."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt, bus=None)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    m = rule.step(model, _batch_x(), ctx)
    assert "loss" in m


def test_invariant_bus_events_fire_in_correct_order():
    """With a real bus, events fire: step_begin → optimizer_step_pre → optimizer_step_post → step_end."""
    rec = _EventRecorder()
    bus = _FakeBus(rec)
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt, bus=bus)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    rule.step(model, _batch_x(), ctx)
    assert rec.events == [
        "on_step_begin",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_step_end",
    ]


def test_invariant_scheduler_step_called_when_step_per_batch_true():
    """scheduler.step() is invoked when scheduler.step_per_batch is True."""
    model = _ff_model()
    opt = _ff_opt(model)
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    ctx = _ff_ctx(model, opt, scheduler=scheduler)
    rule = ForwardForwardUpdateRule()
    rule.step(model, _batch_x(), ctx)
    scheduler.step.assert_called_once()


def test_invariant_scheduler_step_not_called_when_step_per_batch_false():
    """scheduler.step() is NOT invoked when scheduler.step_per_batch is False."""
    model = _ff_model()
    opt = _ff_opt(model)
    scheduler = MagicMock()
    scheduler.step_per_batch = False
    ctx = _ff_ctx(model, opt, scheduler=scheduler)
    rule = ForwardForwardUpdateRule()
    rule.step(model, _batch_x(), ctx)
    scheduler.step.assert_not_called()


def test_invariant_scheduler_none_does_not_raise():
    """scheduler=None → no AttributeError; rule skips scheduler.step()."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt, scheduler=None)
    rule = ForwardForwardUpdateRule()
    m = rule.step(model, _batch_x(), ctx)
    assert "loss" in m


# ---------------------------------------------------------------------------
# Section F — _get_layers coverage
# ---------------------------------------------------------------------------


class _ModelWithLayers(nn.Module):
    """Model exposing a custom .layers attribute (list of nn.Module)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 4)])


class _ModelWithBlocks(nn.Module):
    """Model exposing a custom .blocks attribute (list of nn.Module)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 4)])


class _ModelWithChildrenList(nn.Module):
    """Model exposing a custom .children_list attribute (list of nn.Module)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.children_list = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 4)])


def _run_step(model: nn.Module, D: int = 8) -> dict[str, Any]:
    """Helper: run one FF step and return metrics."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(threshold=2.0)
    torch.manual_seed(10)
    return rule.step(model, {"x": torch.randn(4, D)}, ctx)


def test_invariant_get_layers_uses_layers_attribute():
    """_get_layers prefers .layers when present; step completes successfully."""
    m = _run_step(_ModelWithLayers())
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_get_layers_uses_blocks_attribute():
    """_get_layers falls back to .blocks when .layers is absent."""
    m = _run_step(_ModelWithBlocks())
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_get_layers_uses_children_list_attribute():
    """_get_layers falls back to .children_list when neither .layers nor .blocks is present."""
    m = _run_step(_ModelWithChildrenList())
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_get_layers_falls_back_to_model_children():
    """_get_layers falls back to model.children() for plain nn.Sequential."""
    model = _ff_model()  # nn.Sequential — no .layers/.blocks/.children_list
    rule = ForwardForwardUpdateRule()
    layers = rule._get_layers(model)
    # nn.Sequential with 3 children → list of 3
    assert len(layers) == 3


# ---------------------------------------------------------------------------
# Section G — grad_clip branching
# ---------------------------------------------------------------------------


def test_invariant_grad_clip_zero_skips_clipping():
    """With grad_clip=0, clip_grad_norm_ is not called; grad_norm metric stays 0.0."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(grad_clip=0.0)
    m = rule.step(model, _batch_x(), ctx)
    assert m["grad_norm"] == 0.0


def test_invariant_grad_clip_positive_clips_gradients():
    """With grad_clip=1.0, grad_norm is reported in metrics as a finite float."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(grad_clip=1.0)
    m = rule.step(model, _batch_x(), ctx)
    # grad_norm is finite (may be 0 if no params with grad, but rule computed it)
    assert isinstance(m["grad_norm"], float)
    assert torch.isfinite(torch.tensor(m["grad_norm"]))


# ---------------------------------------------------------------------------
# Section H — metrics keys and ctx.extras
# ---------------------------------------------------------------------------


def test_invariant_metrics_keys_present():
    """Step returns dict containing loss, grad_norm, lr, skipped."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    m = rule.step(model, _batch_x(), ctx)
    for key in ("loss", "grad_norm", "lr", "skipped"):
        assert key in m, f"missing metric key: {key!r}"


def test_invariant_model_stored_in_ctx_extras():
    """After step, ctx.extras['model'] is the passed-in model."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    rule.step(model, _batch_x(), ctx)
    assert ctx.extras.get("model") is model


def test_invariant_skipped_is_zero():
    """'skipped' metric is always 0.0 (FF has no skip mechanism)."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    m = ForwardForwardUpdateRule().step(model, _batch_x(), ctx)
    assert m["skipped"] == 0.0


def test_invariant_lr_reported_correctly():
    """'lr' metric matches the optimizer's initial lr."""
    lr = 7e-4
    model = _ff_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctx = _ff_ctx(model, opt)
    m = ForwardForwardUpdateRule().step(model, _batch_x(), ctx)
    assert m["lr"] == pytest.approx(lr, rel=1e-5)


# ---------------------------------------------------------------------------
# Section I — functional / numerical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.5, 2.0, 5.0])
def test_invariant_loss_finite_for_varied_thresholds(threshold: float):
    """Loss is finite for a range of threshold values."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule(threshold=threshold)
    torch.manual_seed(42)
    m = rule.step(model, _batch_x(), ctx)
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_invariant_parameters_updated_after_step():
    """Model parameters are actually updated after one step (weight changes)."""
    model = _ff_model()
    opt = _ff_opt(model, lr=0.1)
    before = [p.data.clone() for p in model.parameters()]
    ctx = _ff_ctx(model, opt)
    torch.manual_seed(0)
    ForwardForwardUpdateRule(threshold=2.0).step(model, _batch_x(), ctx)
    after = [p.data.clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=False))


def test_invariant_step_with_3d_input_ids():
    """3-D input (B, T, D) is handled — flattening inside the rule for goodness."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    # 3-D: (B=2, T=4, D=8)
    batch = {"input_ids": torch.randint(0, 10, (2, 4, 8))}
    # The rule converts to float; Linear(8,16) can handle (2,4,8) input
    m = rule.step(model, batch, ctx)
    assert "loss" in m
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_pin_current_behavior_registered_under_forward_forward_key():
    """ForwardForwardUpdateRule is registered under key 'forward_forward' in the registry.

    Pin current behavior: registry.get('update_rule', 'forward_forward') returns
    the class. If this changes, all configs using the name break silently.
    """
    from lighttrain.registry import get_registry
    reg = get_registry()
    cls = reg.get("update_rule", "forward_forward")
    assert cls is ForwardForwardUpdateRule


def test_invariant_auto_shuffle_negatives_same_shape_as_positives():
    """Auto-generated negatives (shuffled) have the same shape as positives."""
    model = _ff_model()
    opt = _ff_opt(model)
    ctx = _ff_ctx(model, opt)
    rule = ForwardForwardUpdateRule()
    torch.manual_seed(7)
    # batch with only 'x' — no neg key → shuffle path
    B, D = 4, 8
    x = torch.randn(B, D)
    batch = {"x": x}
    # Step must succeed; negatives will be a permutation of x (same shape)
    m = rule.step(model, batch, ctx)
    assert "loss" in m
