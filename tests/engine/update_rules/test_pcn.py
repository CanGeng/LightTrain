"""Adversarial unit tests for PCNUpdateRule (Predictive Coding Network).

Pins the following previously-uncovered branches:
  - lines 39-43 : _current_lr — wrapper-unwrap, empty param_groups → 0.0
  - line 77     : __init__ raises ValueError on unknown activation
  - line 81     : setup() returns None
  - line 105    : step raises KeyError when neither 'x' nor 'input_ids' in batch
  - line 111    : step raises RuntimeError when model has no nn.Linear layers

Additionally covers general correctness:
  - _current_lr with real SGD optimizer
  - all three valid activations (tanh / relu / none)
  - hyperparameter storage and type coercion
  - state_dict / load_state_dict round-trip
  - bus event dispatching (on_step_begin / on_step_end)
  - supervised clamping (labels same shape as top activation)
  - unsupervised inference (labels is None)
  - input via 'input_ids' key
  - layer bias-update path (bias not None) and no-bias path
  - Hebbian weight updates actually change parameters
  - required metrics keys present in return value
  - PCNUpdateRule registered under ('update_rule', 'pcn')
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.pcn import (
    PCNUpdateRule,
    _current_lr,
)
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext

# ---------------------------------------------------------------------------
# helpers / stubs
# ---------------------------------------------------------------------------


class _TwoLayerMLP(nn.Module):
    """Minimal 2-Linear-layer MLP: input_dim → hidden_dim → out_dim."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        out_dim: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):  # not used by PCN
        return self.fc2(torch.relu(self.fc1(x)))


class _NoLinearModel(nn.Module):
    """Model with no nn.Linear layers — triggers the guard at line 111."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.ReLU()


def _make_ctx(
    *,
    model: nn.Module | None = None,
    lr: float = 1e-2,
    callbacks: list | None = None,
    scheduler=None,
) -> StepContext:
    """Build a minimal StepContext wired to a real SGD optimizer."""
    if model is None:
        model = _TwoLayerMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    bus = EventBus(callbacks or [])
    return StepContext(
        model=model,
        optimizer=optimizer,
        bus=bus,
        scheduler=scheduler,
    )


def _batch_x(B: int = 2, D: int = 4) -> dict[str, torch.Tensor]:
    """Batch containing only the 'x' key."""
    torch.manual_seed(7)
    return {"x": torch.randn(B, D)}


# ---------------------------------------------------------------------------
# _current_lr — lines 39-43
# ---------------------------------------------------------------------------


def test_current_lr_no_param_groups_returns_zero():
    """_current_lr returns 0.0 when inner optimizer has no param_groups (line 41-42)."""

    class _NoGroups:
        param_groups: list = []

    assert _current_lr(_NoGroups()) == 0.0


def test_current_lr_none_param_groups_returns_zero():
    """_current_lr returns 0.0 when getattr(inner, 'param_groups', None) is None (line 41-42)."""

    class _NullGroups:
        pass  # no param_groups attribute at all

    assert _current_lr(_NullGroups()) == 0.0


def test_current_lr_unwraps_inner_optimizer():
    """_current_lr follows the .optimizer wrapper attribute (line 39)."""
    model = nn.Linear(2, 2)
    inner = torch.optim.SGD(model.parameters(), lr=0.456)

    class _Wrapper:
        optimizer = inner

    assert _current_lr(_Wrapper()) == pytest.approx(0.456)


def test_current_lr_with_real_optimizer():
    """_current_lr reads lr from the first param_group of a real SGD (line 43)."""
    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.123)
    assert _current_lr(optim) == pytest.approx(0.123)


def test_current_lr_missing_lr_key_returns_zero():
    """_current_lr returns 0.0 when param_groups[0] has no 'lr' key."""

    class _NoLRGroups:
        param_groups = [{}]

    assert _current_lr(_NoLRGroups()) == 0.0


# ---------------------------------------------------------------------------
# __init__ — validation and attribute storage
# ---------------------------------------------------------------------------


def test_init_unknown_activation_raises_value_error():
    """Unknown activation raises ValueError with the activation name (line 77)."""
    with pytest.raises(ValueError, match="Unknown activation 'sigmoid'"):
        PCNUpdateRule(activation="sigmoid")


@pytest.mark.parametrize("act", ["relu", "tanh", "none"])
def test_init_valid_activations_accepted(act: str):
    """All three valid activation strings are accepted without error."""
    rule = PCNUpdateRule(activation=act)
    assert rule._act is not None


def test_init_stores_hyperparams():
    """n_infer, lr_infer, lr_weight are stored and type-coerced in __init__."""
    rule = PCNUpdateRule(n_infer=5, lr_infer=0.05, lr_weight=0.002, activation="relu")
    assert rule.n_infer == 5
    assert rule.lr_infer == pytest.approx(0.05)
    assert rule.lr_weight == pytest.approx(0.002)


def test_init_n_infer_accepts_float_string_coercion():
    """n_infer is int-cast from its argument (type coercion guard)."""
    rule = PCNUpdateRule(n_infer=3.9)
    assert rule.n_infer == 3


# ---------------------------------------------------------------------------
# setup() — line 81
# ---------------------------------------------------------------------------


def test_setup_returns_none():
    """setup() is a no-op that explicitly returns None (line 81)."""
    rule = PCNUpdateRule()
    result = rule.setup(model=MagicMock(), sample=MagicMock())
    assert result is None


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    """state_dict captures n_infer / lr_infer / lr_weight and load_state_dict restores them."""
    rule = PCNUpdateRule(n_infer=10, lr_infer=0.05, lr_weight=0.003)
    sd = rule.state_dict()
    assert sd["n_infer"] == 10
    assert sd["lr_infer"] == pytest.approx(0.05)
    assert sd["lr_weight"] == pytest.approx(0.003)

    rule2 = PCNUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.n_infer == 10
    assert rule2.lr_infer == pytest.approx(0.05)
    assert rule2.lr_weight == pytest.approx(0.003)


def test_load_state_dict_uses_current_values_for_missing_keys():
    """load_state_dict falls back to current attribute values when keys absent."""
    rule = PCNUpdateRule(n_infer=7, lr_infer=0.07, lr_weight=0.007)
    rule.load_state_dict({})  # empty dict → no changes
    assert rule.n_infer == 7
    assert rule.lr_infer == pytest.approx(0.07)
    assert rule.lr_weight == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# step() — input-validation errors
# ---------------------------------------------------------------------------


def test_step_raises_key_error_when_no_x_or_input_ids():
    """step raises KeyError when batch has neither 'x' nor 'input_ids' (line 105)."""
    rule = PCNUpdateRule()
    ctx = _make_ctx()
    with pytest.raises(KeyError, match="'x' or 'input_ids'"):
        rule.step(
            _TwoLayerMLP(),
            {"labels": torch.zeros(2, 3)},
            ctx,
        )


def test_step_raises_runtime_error_when_no_linear_layers():
    """step raises RuntimeError when model has no nn.Linear layers (line 111)."""
    rule = PCNUpdateRule()
    model = _NoLinearModel()
    # _NoLinearModel has no parameters; build ctx manually so SGD doesn't reject empty params.
    ctx = StepContext(bus=EventBus([]))
    with pytest.raises(RuntimeError, match="no nn.Linear layers"):
        rule.step(model, _batch_x(), ctx)


# ---------------------------------------------------------------------------
# step() — fallback key 'input_ids'
# ---------------------------------------------------------------------------


def test_step_accepts_input_ids_key():
    """step uses 'input_ids' as fallback when 'x' is absent."""
    torch.manual_seed(0)
    rule = PCNUpdateRule(n_infer=1, activation="relu")
    model = _TwoLayerMLP(input_dim=4)
    ctx = _make_ctx(model=model)
    batch = {"input_ids": torch.randn(2, 4)}
    metrics = rule.step(model, batch, ctx)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# step() — bus event dispatching
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records event names dispatched through EventBus."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_step_begin(self, **_kw) -> None:
        self.events.append("on_step_begin")

    def on_step_end(self, **_kw) -> None:
        self.events.append("on_step_end")


def test_step_dispatches_step_begin_and_step_end_via_bus():
    """bus dispatches on_step_begin before on_step_end when bus is not None."""
    rec = _EventRecorder()
    torch.manual_seed(0)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, callbacks=[rec])
    PCNUpdateRule(n_infer=2).step(model, _batch_x(), ctx)
    assert "on_step_begin" in rec.events
    assert "on_step_end" in rec.events
    assert rec.events.index("on_step_begin") < rec.events.index("on_step_end")


def test_step_no_bus_does_not_raise():
    """step proceeds without error when ctx.bus is None (no bus dispatch)."""
    torch.manual_seed(1)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    ctx.bus = None
    metrics = PCNUpdateRule(n_infer=1).step(model, _batch_x(), ctx)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# step() — activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh", "none"])
def test_step_runs_all_activation_branches(activation: str):
    """step completes for each of the three activations and returns float loss."""
    torch.manual_seed(3)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    rule = PCNUpdateRule(n_infer=2, activation=activation)
    metrics = rule.step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — supervised clamping
# ---------------------------------------------------------------------------


def test_step_supervised_labels_same_shape_as_top_activation():
    """When labels.shape == acts[-1].shape, top activation is clamped to labels."""
    torch.manual_seed(5)
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    ctx = _make_ctx(model=model)
    rule = PCNUpdateRule(n_infer=2, activation="tanh")
    B, out_dim = 2, 3
    batch = {
        "x": torch.randn(B, 4),
        "labels": torch.randn(B, out_dim),  # same shape as fc2 output
    }
    metrics = rule.step(model, batch, ctx)
    assert isinstance(metrics["loss"], float)


def test_step_unsupervised_no_labels():
    """When labels is absent, inference runs without supervised clamping."""
    torch.manual_seed(6)
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    ctx = _make_ctx(model=model)
    rule = PCNUpdateRule(n_infer=2, activation="none")
    batch = {"x": torch.randn(2, 4)}
    metrics = rule.step(model, batch, ctx)
    assert isinstance(metrics["loss"], float)


def test_step_labels_shape_mismatch_does_not_clamp():
    """When labels.shape != acts[-1].shape, clamping is skipped (no error)."""
    torch.manual_seed(9)
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    ctx = _make_ctx(model=model)
    rule = PCNUpdateRule(n_infer=1, activation="relu")
    # labels has wrong shape: (2,) vs (2, 3)
    batch = {"x": torch.randn(2, 4), "labels": torch.zeros(2)}
    metrics = rule.step(model, batch, ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — weight update (Hebbian)
# ---------------------------------------------------------------------------


def test_step_updates_weights_hebbianly():
    """PCN modifies layer weights after a supervised step (Hebbian update).

    NOTE (pin_current_behavior): unsupervised PCN initialises activations from a
    forward pass, so prediction errors at initialisation are exactly zero and
    delta_W = 0.  Weights only change when supervised labels clamp the top
    activation to a target that differs from the network's prediction.
    """
    torch.manual_seed(42)
    model = _TwoLayerMLP()
    w_before = model.fc2.weight.detach().clone()
    ctx = _make_ctx(model=model)
    # Supervised batch: random label != predicted top activation → non-zero error
    batch = {"x": torch.randn(2, 4), "labels": torch.randn(2, 3)}
    PCNUpdateRule(n_infer=2, lr_weight=0.1).step(model, batch, ctx)
    assert not torch.equal(model.fc2.weight, w_before)


def test_step_updates_bias_when_present():
    """PCN also updates bias terms when bias is not None (supervised step).

    NOTE (pin_current_behavior): same as test_step_updates_weights_hebbianly —
    only supervised runs produce non-zero errors and thus non-zero bias deltas.
    """
    torch.manual_seed(42)
    model = _TwoLayerMLP(bias=True)
    b_before = model.fc2.bias.detach().clone()
    ctx = _make_ctx(model=model)
    batch = {"x": torch.randn(2, 4), "labels": torch.randn(2, 3)}
    PCNUpdateRule(n_infer=2, lr_weight=0.5).step(model, batch, ctx)
    assert not torch.equal(model.fc2.bias, b_before)


def test_step_works_with_no_bias_layers():
    """step succeeds when Linear layers have bias=False."""
    torch.manual_seed(12)
    model = _TwoLayerMLP(bias=False)
    ctx = _make_ctx(model=model)
    metrics = PCNUpdateRule(n_infer=1).step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — required metrics keys
# ---------------------------------------------------------------------------


def test_step_returns_required_metrics_keys():
    """step returns dict with loss, grad_norm, lr, skipped."""
    torch.manual_seed(13)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    metrics = PCNUpdateRule(n_infer=1).step(model, _batch_x(), ctx)
    for key in ("loss", "grad_norm", "lr", "skipped"):
        assert key in metrics
    assert metrics["grad_norm"] == 0.0
    assert metrics["skipped"] == 0.0


def test_step_lr_in_metrics_equals_lr_weight():
    """metrics['lr'] is set to self.lr_weight (the Hebbian learning rate)."""
    torch.manual_seed(14)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    rule = PCNUpdateRule(n_infer=1, lr_weight=0.042)
    metrics = rule.step(model, _batch_x(), ctx)
    assert metrics["lr"] == pytest.approx(0.042)


# ---------------------------------------------------------------------------
# step() — extras side-effect
# ---------------------------------------------------------------------------


def test_step_sets_model_in_extras():
    """step stores the model reference in ctx.extras['model']."""
    torch.manual_seed(15)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    PCNUpdateRule(n_infer=1).step(model, _batch_x(), ctx)
    assert ctx.extras["model"] is model


# ---------------------------------------------------------------------------
# step() — zero inference steps
# ---------------------------------------------------------------------------


def test_step_zero_inference_steps_runs_without_error():
    """n_infer=0 skips the inference loop entirely and still returns metrics."""
    torch.manual_seed(16)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    metrics = PCNUpdateRule(n_infer=0).step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — multi-layer model (>2 layers)
# ---------------------------------------------------------------------------


def test_step_three_layer_model():
    """step works on a 3-layer model (tests the middle-activation update loop)."""
    torch.manual_seed(17)

    class _ThreeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 6)
            self.fc3 = nn.Linear(6, 3)

    model = _ThreeLayer()
    ctx = _make_ctx(model=model)
    metrics = PCNUpdateRule(n_infer=2).step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pcn_registered_in_update_rule_namespace(clean_registry):
    """PCNUpdateRule is discoverable via the 'update_rule'/'pcn' key."""
    from lighttrain.registry import get_registry

    reg = get_registry()
    cls = reg.get("update_rule", "pcn")
    assert cls is PCNUpdateRule
