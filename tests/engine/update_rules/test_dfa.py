"""Adversarial unit tests for DFAUpdateRule (Direct Feedback Alignment).

Pins the following branches / invariants that were previously uncovered:
  - line 43 : _current_lr returns 0.0 when no param_groups
  - line 70 : __init__ raises ValueError on unknown activation
  - lines 77-80 : _act_deriv for all three activations (relu / tanh / none)
  - line 91 : setup() returns None
  - line 112 : bus.dispatch("on_step_begin") when bus is not None
  - line 118 : step raises KeyError when 'x'/'input_ids' absent from batch
  - line 124 : step raises RuntimeError when model has < 2 Linear layers
  - lines 139-142 : forward pass activation branches (relu / tanh / none)
  - line 152 : cross-layer MSE loss when labels are 2-D
  - line 155 : unsupervised MSE reconstruction when labels is None
  - lines 179-181 : last-layer loss branches (2-D labels / no labels)
  - line 200 : scheduler.step() called when step_per_batch=True
  - line 203 : bus.dispatch("on_step_end") when bus is not None
  - state_dict / load_state_dict round-trip
  - feedback matrix lazily created and cached
  - weight update actually changes parameters
  - metrics keys present in return value
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.dfa import (
    DFAUpdateRule,
    _current_lr,
)
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext

# ---------------------------------------------------------------------------
# helpers
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

    def forward(self, x):  # not called by DFAUpdateRule
        return self.fc2(torch.relu(self.fc1(x)))


class _SingleLinear(nn.Module):
    """Only one Linear layer — used to trigger the < 2 layers guard."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)


def _make_ctx(
    *,
    model: nn.Module | None = None,
    lr: float = 1e-2,
    callbacks: list | None = None,
    scheduler=None,
) -> StepContext:
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
    """Batch with 'x' key."""
    torch.manual_seed(7)
    return {"x": torch.randn(B, D)}


def _batch_with_labels_1d(B: int = 2, D: int = 4, n_classes: int = 3) -> dict:
    torch.manual_seed(7)
    return {
        "x": torch.randn(B, D),
        "labels": torch.randint(0, n_classes, (B,)),
    }


def _batch_with_labels_2d(B: int = 2, D: int = 4, out_dim: int = 3) -> dict:
    torch.manual_seed(7)
    return {
        "x": torch.randn(B, D),
        "labels": torch.randn(B, out_dim),
    }


# ---------------------------------------------------------------------------
# _current_lr helper
# ---------------------------------------------------------------------------


def test_current_lr_no_param_groups_returns_zero():
    """_current_lr returns 0.0 when optimizer exposes no param_groups (line 43)."""

    class _NoGroups:
        param_groups: list = []

    assert _current_lr(_NoGroups()) == 0.0


def test_current_lr_with_real_optimizer():
    """_current_lr reads lr from first param_group of a real SGD optimizer."""
    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.123)
    assert _current_lr(optim) == pytest.approx(0.123)


def test_current_lr_unwraps_inner_optimizer():
    """_current_lr follows the .optimizer attribute (wrapper pattern)."""
    model = nn.Linear(2, 2)
    inner = torch.optim.SGD(model.parameters(), lr=0.456)

    class _Wrapper:
        optimizer = inner

    assert _current_lr(_Wrapper()) == pytest.approx(0.456)


# ---------------------------------------------------------------------------
# __init__ — validation and attribute storage
# ---------------------------------------------------------------------------


def test_init_unknown_activation_raises_value_error():
    """Unknown activation raises ValueError with the activation name (line 70)."""
    with pytest.raises(ValueError, match="Unknown activation 'sigmoid'"):
        DFAUpdateRule(activation="sigmoid")


@pytest.mark.parametrize("act", ["relu", "tanh", "none"])
def test_init_valid_activations_accepted(act: str):
    """All three valid activation strings are accepted without error."""
    rule = DFAUpdateRule(activation=act)
    assert rule.activation == act


def test_init_stores_hyperparams():
    """feedback_scale, lr, activation are stored as instance attributes."""
    rule = DFAUpdateRule(feedback_scale=0.05, activation="tanh", lr=2e-3)
    assert rule.feedback_scale == pytest.approx(0.05)
    assert rule.lr == pytest.approx(2e-3)
    assert rule.activation == "tanh"
    assert rule._feedback == {}


# ---------------------------------------------------------------------------
# _act_deriv — all three branches
# ---------------------------------------------------------------------------


def test_act_deriv_relu_positive():
    """relu derivative is 1 where z > 0 (line 77)."""
    rule = DFAUpdateRule(activation="relu")
    z = torch.tensor([1.0, -1.0, 0.0])
    d = rule._act_deriv(z)
    expected = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(d, expected)


def test_act_deriv_tanh_derivative(
):
    """tanh derivative is 1 - tanh(z)^2 (lines 78-79)."""
    rule = DFAUpdateRule(activation="tanh")
    torch.manual_seed(1)
    z = torch.randn(4)
    d = rule._act_deriv(z)
    expected = 1.0 - torch.tanh(z) ** 2
    torch.testing.assert_close(d, expected, atol=1e-6, rtol=1e-5)


def test_act_deriv_none_returns_ones(
):
    """'none' derivative is all-ones (line 80)."""
    rule = DFAUpdateRule(activation="none")
    z = torch.randn(5)
    d = rule._act_deriv(z)
    torch.testing.assert_close(d, torch.ones_like(z))


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------


def test_setup_returns_none():
    """setup() is a no-op and explicitly returns None (line 91)."""
    rule = DFAUpdateRule()
    result = rule.setup(model=MagicMock(), sample=MagicMock())
    assert result is None


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    """state_dict carries feedback_scale, lr, activation and restores them."""
    rule = DFAUpdateRule(feedback_scale=0.007, activation="tanh", lr=5e-4)
    sd = rule.state_dict()
    assert sd["feedback_scale"] == pytest.approx(0.007)
    assert sd["lr"] == pytest.approx(5e-4)
    assert sd["activation"] == "tanh"

    rule2 = DFAUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2.feedback_scale == pytest.approx(0.007)
    assert rule2.lr == pytest.approx(5e-4)
    assert rule2.activation == "tanh"


def test_load_state_dict_uses_defaults_for_missing_keys():
    """load_state_dict falls back to current values when keys are absent."""
    rule = DFAUpdateRule(feedback_scale=0.02, lr=1e-3, activation="relu")
    rule.load_state_dict({})  # empty dict → no changes
    assert rule.feedback_scale == pytest.approx(0.02)
    assert rule.lr == pytest.approx(1e-3)
    assert rule.activation == "relu"


# ---------------------------------------------------------------------------
# step() — input validation errors (lines 118, 124)
# ---------------------------------------------------------------------------


def test_step_raises_key_error_when_no_x_or_input_ids():
    """step raises KeyError when batch has neither 'x' nor 'input_ids' (line 118)."""
    rule = DFAUpdateRule()
    ctx = _make_ctx()
    with pytest.raises(KeyError, match="'x' or 'input_ids'"):
        rule.step(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3)), {"y": torch.randn(2, 4)}, ctx)


def test_step_raises_runtime_error_when_fewer_than_two_layers():
    """step raises RuntimeError when model has only one Linear layer (line 124)."""
    rule = DFAUpdateRule()
    model = _SingleLinear()
    ctx = _make_ctx(model=model)
    with pytest.raises(RuntimeError, match="at least 2 nn.Linear"):
        rule.step(model, _batch_x(), ctx)


def test_step_accepts_input_ids_key():
    """step accepts 'input_ids' as fallback for 'x' key."""
    torch.manual_seed(0)
    rule = DFAUpdateRule(activation="relu")
    model = _TwoLayerMLP(input_dim=4)
    ctx = _make_ctx(model=model)
    batch = {"input_ids": torch.randn(2, 4)}
    metrics = rule.step(model, batch, ctx)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# step() — bus events (lines 112, 203)
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records event names dispatched through the bus."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_step_begin(self, **_kw) -> None:
        self.events.append("on_step_begin")

    def on_step_end(self, **_kw) -> None:
        self.events.append("on_step_end")


def test_step_dispatches_step_begin_and_step_end_via_bus():
    """bus.dispatch fires on_step_begin and on_step_end when bus is not None (lines 112, 203)."""
    rec = _EventRecorder()
    torch.manual_seed(0)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, callbacks=[rec])
    DFAUpdateRule().step(model, _batch_x(), ctx)
    assert "on_step_begin" in rec.events
    assert "on_step_end" in rec.events
    assert rec.events.index("on_step_begin") < rec.events.index("on_step_end")


# ---------------------------------------------------------------------------
# step() — activation forward-pass branches (lines 139, 140, 142)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh", "none"])
def test_step_runs_all_activation_branches(activation: str):
    """step runs to completion for each of the three activations (lines 139-142)."""
    torch.manual_seed(3)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    rule = DFAUpdateRule(activation=activation)
    metrics = rule.step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — label / loss branches
# ---------------------------------------------------------------------------


def test_step_with_1d_labels_uses_cross_entropy():
    """Integer 1-D labels trigger cross-entropy loss (line 150)."""
    torch.manual_seed(4)
    model = _TwoLayerMLP(out_dim=3)
    ctx = _make_ctx(model=model)
    rule = DFAUpdateRule(activation="relu")
    metrics = rule.step(model, _batch_with_labels_1d(n_classes=3), ctx)
    assert metrics["loss"] > 0.0


def test_step_with_2d_labels_uses_mse_loss():
    """Float 2-D labels trigger MSE loss (line 152 / 179)."""
    torch.manual_seed(5)
    model = _TwoLayerMLP(out_dim=3)
    ctx = _make_ctx(model=model)
    rule = DFAUpdateRule(activation="relu")
    metrics = rule.step(model, _batch_with_labels_2d(out_dim=3), ctx)
    assert isinstance(metrics["loss"], float)


def test_step_without_labels_uses_reconstruction_loss():
    """When labels is None, unsupervised reconstruction MSE is used (lines 155, 181).

    The reconstruction target is acts[-2][:, :out_size], so hidden_dim >= out_size
    is required for shapes to align.  We pick hidden_dim=8, out_dim=3.
    """
    torch.manual_seed(6)
    # hidden_dim=8 >= out_dim=3  => acts[-2] is [B, 8], slice [:, :3] = [B, 3] matches z_last
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    ctx = _make_ctx(model=model)
    rule = DFAUpdateRule(activation="none")
    batch = {"x": torch.randn(2, 4)}
    metrics = rule.step(model, batch, ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# step() — scheduler integration (line 200)
# ---------------------------------------------------------------------------


def test_step_calls_scheduler_step_when_step_per_batch_true():
    """scheduler.step() is called when step_per_batch=True (line 200)."""
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    torch.manual_seed(8)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, scheduler=scheduler)
    DFAUpdateRule().step(model, _batch_x(), ctx)
    scheduler.step.assert_called_once()


def test_step_does_not_call_scheduler_when_step_per_batch_false():
    """scheduler.step() is NOT called when step_per_batch=False."""
    scheduler = MagicMock()
    scheduler.step_per_batch = False
    torch.manual_seed(9)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, scheduler=scheduler)
    DFAUpdateRule().step(model, _batch_x(), ctx)
    scheduler.step.assert_not_called()


def test_step_does_not_call_scheduler_when_scheduler_is_none():
    """When scheduler is None, step proceeds without error (no scheduler.step)."""
    torch.manual_seed(10)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, scheduler=None)
    metrics = DFAUpdateRule().step(model, _batch_x(), ctx)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# step() — feedback matrix management
# ---------------------------------------------------------------------------


def test_feedback_matrices_created_lazily():
    """Feedback matrices are empty before step and populated after (line 86-88)."""
    rule = DFAUpdateRule()
    assert rule._feedback == {}
    torch.manual_seed(11)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx)
    # One feedback matrix per hidden layer (all but the last)
    assert len(rule._feedback) == 1  # 2-layer MLP → 1 hidden layer


def test_feedback_matrices_cached_across_steps():
    """Feedback matrix ids are stable across consecutive steps (no re-creation)."""
    torch.manual_seed(12)
    rule = DFAUpdateRule()
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx)
    ids_after_first = {k: id(v) for k, v in rule._feedback.items()}
    rule.step(model, _batch_x(), ctx)
    ids_after_second = {k: id(v) for k, v in rule._feedback.items()}
    assert ids_after_first == ids_after_second


def test_feedback_matrix_shape_matches_layer_dimensions():
    """Feedback matrix B_l has shape (out_l, out_last)."""
    torch.manual_seed(13)
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    rule = DFAUpdateRule()
    ctx = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx)
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    B_l = list(rule._feedback.values())[0]
    # shape: (hidden_dim, out_dim)
    assert B_l.shape == (layers[0].out_features, layers[-1].out_features)


def test_feedback_matrix_recreated_on_shape_change():
    """pin_current_behavior: feedback matrix is re-created when out_size changes.

    If the out_features of the last layer changes (e.g. model reuse with a
    different head), a new B_l with the new shape is created.
    Note: this is not an advertised API, just pinning current behavior.
    """
    torch.manual_seed(14)
    rule = DFAUpdateRule()
    model = _TwoLayerMLP(input_dim=4, hidden_dim=8, out_dim=3)
    ctx = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx)
    first_shape = list(rule._feedback.values())[0].shape

    # Manually override out_features of last layer to simulate shape change
    with torch.no_grad():
        model.fc2 = nn.Linear(8, 5)  # out_dim changes from 3 to 5
    ctx2 = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx2)
    second_shape = list(rule._feedback.values())[0].shape

    assert first_shape != second_shape
    assert second_shape == (8, 5)


# ---------------------------------------------------------------------------
# step() — weight updates and metrics
# ---------------------------------------------------------------------------


def test_step_updates_hidden_layer_weights():
    """DFA modifies hidden layer weights after step."""
    torch.manual_seed(15)
    model = _TwoLayerMLP()
    w_before = model.fc1.weight.detach().clone()
    ctx = _make_ctx(model=model)
    DFAUpdateRule(lr=1e-2).step(model, _batch_x(), ctx)
    assert not torch.equal(model.fc1.weight, w_before)


def test_step_updates_output_layer_weights():
    """DFA modifies output layer weights (standard grad step) after step."""
    torch.manual_seed(16)
    model = _TwoLayerMLP()
    w_before = model.fc2.weight.detach().clone()
    ctx = _make_ctx(model=model)
    DFAUpdateRule(lr=1e-2).step(model, _batch_x(), ctx)
    assert not torch.equal(model.fc2.weight, w_before)


def test_step_returns_required_metrics_keys():
    """step() returns dict with loss, grad_norm, lr, skipped (coverage of metrics block)."""
    torch.manual_seed(17)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model)
    metrics = DFAUpdateRule().step(model, _batch_x(), ctx)
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "lr" in metrics
    assert "skipped" in metrics
    assert metrics["skipped"] == 0.0


def test_step_lr_from_optimizer_overrides_rule_lr():
    """eff_lr = optimizer lr (if nonzero) rather than the rule's default lr."""
    torch.manual_seed(18)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, lr=0.05)
    metrics = DFAUpdateRule(lr=1e-6).step(model, _batch_x(), ctx)
    assert metrics["lr"] == pytest.approx(0.05)


def test_step_falls_back_to_rule_lr_when_optimizer_lr_is_zero():
    """When optimizer lr=0, rule falls back to self.lr (eff_lr = self.lr)."""
    torch.manual_seed(19)
    model = _TwoLayerMLP()
    ctx = _make_ctx(model=model, lr=0.0)
    rule = DFAUpdateRule(lr=0.007)
    metrics = rule.step(model, _batch_x(), ctx)
    assert metrics["lr"] == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# step() — no-bias layers
# ---------------------------------------------------------------------------


def test_step_works_with_bias_false_layers():
    """step succeeds when Linear layers have no bias (bias=None path at line 171)."""
    torch.manual_seed(20)
    model = _TwoLayerMLP(bias=False)
    ctx = _make_ctx(model=model)
    metrics = DFAUpdateRule().step(model, _batch_x(), ctx)
    assert isinstance(metrics["loss"], float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_dfa_registered_in_update_rule_namespace(clean_registry):
    """DFAUpdateRule is discoverable via the 'update_rule'/'dfa' key."""
    from lighttrain.registry import get_registry

    reg = get_registry()
    cls = reg.get("update_rule", "dfa")
    assert cls is DFAUpdateRule


# ---------------------------------------------------------------------------
# Deeper three-layer model
# ---------------------------------------------------------------------------


def test_step_three_layer_model_generates_two_feedback_matrices():
    """A 3-Linear-layer model produces 2 feedback matrices (one per hidden layer)."""
    torch.manual_seed(21)

    class _ThreeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 6)
            self.fc3 = nn.Linear(6, 3)

    model = _ThreeLayer()
    rule = DFAUpdateRule()
    ctx = _make_ctx(model=model)
    rule.step(model, _batch_x(), ctx)
    assert len(rule._feedback) == 2
