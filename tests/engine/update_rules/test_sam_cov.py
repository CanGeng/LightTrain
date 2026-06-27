"""Coverage-extension tests for lighttrain/builtin_plugins/engine/update_rules/sam.py.

Targets the following previously uncovered lines (module at 93% before this file):
  - 39-42 : _to_metric – float(value) fast-path and TypeError/ValueError fallback
  - 49    : _current_lr – empty param_groups → return 0.0
  - 82    : SAMUpdateRule.setup() → return None
  - 106   : _compute_perturbation() with no gradients → return []
  - 116   : perturbations.append(None) for frozen (no-grad) parameters
  - 153-154: model forward returns non-ModelOutput → wrapped into ModelOutput
  - 159   : loss_fn dict missing 'loss' key → KeyError raised
  - 269   : extra keys beyond 'loss' in loss_dict copied into ctx.metrics
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.sam import (
    SAMUpdateRule,
    _current_lr,
    _to_metric,
)
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# helpers — minimal model + context builders
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """One-layer model whose forward returns a ModelOutput by default."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x, **_):
        return ModelOutput(outputs={"logits": self.linear(x)})


class _TinyModelRawTensor(nn.Module):
    """Forward returns a raw tensor (not ModelOutput) to hit line 153-154."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x, **_):
        return self.linear(x)  # plain Tensor, not ModelOutput


class _TinyModelMapping(nn.Module):
    """Forward returns a plain dict (Mapping) to hit line 154 dict() branch."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x, **_):
        return {"logits": self.linear(x)}  # Mapping, not ModelOutput


def _simple_loss(model_output, batch, ctx):
    pred = model_output.outputs["logits"]
    return {"loss": (pred - 1.0).pow(2).mean()}


def _build_ctx(model=None, *, callbacks=None, loss_fn=None, scheduler=None):
    if model is None:
        model = _TinyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=EventBus(callbacks or []),
        loss_fn=loss_fn or _simple_loss,
        scheduler=scheduler,
    )
    return ctx, model, optim


def _batch():
    torch.manual_seed(7)
    return {"x": torch.randn(2, 4)}


# ===========================================================================
# _to_metric helper — lines 36-42
# ===========================================================================


def test_invariant_to_metric_with_tensor():
    """_to_metric converts a scalar tensor to float via fast path (line 38)."""
    t = torch.tensor(3.14)
    result = _to_metric(t)
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-5) == 3.14


def test_invariant_to_metric_with_plain_float():
    """_to_metric returns float(value) for a plain Python float (line 40)."""
    result = _to_metric(2.71828)
    assert isinstance(result, float)
    assert pytest.approx(result, rel=1e-5) == 2.71828


def test_invariant_to_metric_with_plain_int():
    """_to_metric converts a plain int via the try-float path (line 40)."""
    result = _to_metric(42)
    assert isinstance(result, float)
    assert result == 42.0


def test_invariant_to_metric_non_numeric_returns_nan():
    """_to_metric returns float('nan') for a non-numeric value (lines 41-42).

    The except branch catches TypeError and ValueError so non-convertible
    values do not crash metric collection.
    """
    result = _to_metric("not-a-number")
    assert isinstance(result, float)
    import math
    assert math.isnan(result)


def test_invariant_to_metric_none_returns_nan():
    """_to_metric returns float('nan') for None (TypeError branch, line 41-42)."""
    result = _to_metric(None)
    import math
    assert math.isnan(result)


# ===========================================================================
# _current_lr helper — lines 45-50
# ===========================================================================


def test_invariant_current_lr_empty_groups_returns_zero():
    """_current_lr returns 0.0 when optimizer has no param_groups (line 49).

    This can happen with a fresh / emptied optimizer stub.
    """

    class _EmptyOptim:
        param_groups = []

    result = _current_lr(_EmptyOptim())
    assert result == 0.0


def test_invariant_current_lr_none_groups_returns_zero():
    """_current_lr returns 0.0 when inner has param_groups=None (line 49)."""

    class _NoGroupsOptim:
        param_groups = None

    result = _current_lr(_NoGroupsOptim())
    assert result == 0.0


def test_invariant_current_lr_normal_optimizer():
    """_current_lr returns the lr from the first param group (line 50)."""
    param = nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([param], lr=0.123)
    result = _current_lr(optim)
    assert pytest.approx(result, rel=1e-6) == 0.123


# ===========================================================================
# SAMUpdateRule.setup() — line 82
# ===========================================================================


def test_invariant_setup_returns_none():
    """SAMUpdateRule.setup() is a no-op hook that returns None (line 82)."""
    rule = SAMUpdateRule()
    result = rule.setup(model=object(), sample=object())  # type: ignore[func-returns-value]
    assert result is None


# ===========================================================================
# _compute_perturbation with no gradients — line 106
# ===========================================================================


def test_invariant_compute_perturbation_no_grads_returns_empty():
    """_compute_perturbation returns [] when no parameter has a gradient (line 106).

    This branch is hit when the rule is invoked before any backward pass has
    been run (e.g. the first micro-step in accumulation mode before grads exist).
    """
    rule = SAMUpdateRule()
    model = _TinyModel()
    # Ensure no grads are populated
    for p in model.parameters():
        assert p.grad is None
    result = rule._compute_perturbation(model)
    assert result == []


# ===========================================================================
# _compute_perturbation — line 116 (None appended for frozen params)
# ===========================================================================


class _ModelWithFrozenParam(nn.Module):
    """Model with one trainable and one frozen parameter.

    The frozen param (bias) will have p.grad is None after backward, so
    _compute_perturbation must append None for it (line 116).
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=True)
        nn.init.ones_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.bias.requires_grad_(False)  # freeze bias

    def forward(self, x, **_):
        return ModelOutput(outputs={"logits": self.linear(x)})


def test_invariant_compute_perturbation_frozen_param_appends_none():
    """_compute_perturbation appends None for params with no gradient (line 116).

    Construction: a model with a frozen bias (requires_grad=False). After one
    backward, only the weight gets a grad. The perturbations list should have
    two entries: a tensor for the weight and None for the bias.
    """
    rule = SAMUpdateRule(rho=0.05)
    model = _ModelWithFrozenParam()

    # Simulate a forward+backward so weight gets a grad
    x = torch.randn(2, 4)
    output = model(x=x)
    loss = output.outputs["logits"].sum()
    loss.backward()

    # bias.grad must still be None (frozen)
    assert model.linear.bias.grad is None
    # weight.grad must be populated
    assert model.linear.weight.grad is not None

    perturbations = rule._compute_perturbation(model)

    # Two parameters: weight + bias
    params = list(model.parameters())
    assert len(perturbations) == len(params)

    # The frozen parameter's slot must be None
    frozen_idx = next(i for i, p in enumerate(params) if not p.requires_grad)
    assert perturbations[frozen_idx] is None

    # The trainable parameter's slot must be a non-None tensor
    trainable_idx = next(i for i, p in enumerate(params) if p.requires_grad)
    assert perturbations[trainable_idx] is not None
    assert isinstance(perturbations[trainable_idx], torch.Tensor)


def test_invariant_restore_with_frozen_param_skips_none_slots():
    """_restore skips None slots so frozen params are not modified (related to line 116).

    After compute_perturbation the perturbations list has None for frozen params.
    _restore must skip those without crashing.
    """
    rule = SAMUpdateRule(rho=0.05)
    model = _ModelWithFrozenParam()

    x = torch.randn(2, 4)
    output = model(x=x)
    output.outputs["logits"].sum().backward()

    weight_before = model.linear.weight.data.clone()
    bias_before = model.linear.bias.data.clone()

    perturbations = rule._compute_perturbation(model)
    # After perturbation, weight should have shifted
    assert not torch.equal(model.linear.weight.data, weight_before)

    rule._restore(model, perturbations)
    # After restore, weight should be back
    torch.testing.assert_close(model.linear.weight.data, weight_before, atol=1e-6, rtol=1e-5)
    # Frozen bias must be untouched throughout
    torch.testing.assert_close(model.linear.bias.data, bias_before)


# ===========================================================================
# Full step with frozen params — covers line 116 via the step() path
# ===========================================================================


def test_invariant_step_with_frozen_param_completes():
    """A full SAM step with a mixed trainable/frozen model completes without error.

    Exercises lines 106, 116 via the normal step() code path.
    """
    torch.manual_seed(0)
    model = _ModelWithFrozenParam()
    ctx, model, _ = _build_ctx(model=model)
    rule = SAMUpdateRule(rho=0.05, grad_clip=0.0)
    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics
    assert "grad_norm" in metrics


# ===========================================================================
# Non-ModelOutput return from model forward — lines 153-154
# ===========================================================================


def test_invariant_non_model_output_mapping_wrapped():
    """When model returns a Mapping (dict), it is wrapped into ModelOutput (line 154).

    Catches a refactor that assumes model.forward always returns ModelOutput.
    """
    torch.manual_seed(1)
    model = _TinyModelMapping()
    ctx, model, _ = _build_ctx(model=model)
    rule = SAMUpdateRule(grad_clip=0.0)
    # Should not raise even though model returns a dict
    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics


def test_invariant_non_model_output_tensor_wrapped():
    """When model returns a plain Tensor, it is wrapped into ModelOutput (line 153-154).

    The wrapping produces {"logits": tensor} so a loss_fn that reads
    model_output.outputs["logits"] still works.
    """
    torch.manual_seed(2)
    model = _TinyModelRawTensor()
    ctx, model, _ = _build_ctx(model=model)
    rule = SAMUpdateRule(grad_clip=0.0)
    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics


# ===========================================================================
# KeyError when loss_fn missing 'loss' key — line 159
# ===========================================================================


def test_invariant_loss_fn_missing_loss_key_raises():
    """_forward_loss raises KeyError with a clear message when loss_fn omits 'loss' (line 159)."""

    def _bad_loss(model_output, batch, ctx):
        return {"not_loss": torch.tensor(1.0)}

    ctx, model, _ = _build_ctx(loss_fn=_bad_loss)
    rule = SAMUpdateRule()
    with pytest.raises(KeyError, match="loss"):
        rule.step(model, _batch(), ctx)


# ===========================================================================
# Extra loss_dict keys beyond 'loss' copied to metrics — line 269
# ===========================================================================


def test_invariant_extra_loss_keys_surfaced_in_metrics():
    """Extra keys in loss_fn's return dict beyond 'loss' are merged into ctx.metrics (line 269).

    Construction: loss_fn returns {"loss": ..., "aux_loss": ..., "nll": ...}.
    Expected: all three extra keys appear in the returned metrics dict.
    """

    def _loss_with_extras(model_output, batch, ctx):
        pred = model_output.outputs["logits"]
        base = (pred - 1.0).pow(2).mean()
        return {
            "loss": base,
            "aux_loss": base * 0.1,
            "nll": base * 0.5,
        }

    torch.manual_seed(3)
    ctx, model, _ = _build_ctx(loss_fn=_loss_with_extras)
    rule = SAMUpdateRule(grad_clip=0.0)
    metrics = rule.step(model, _batch(), ctx)

    assert "aux_loss" in metrics, f"aux_loss missing; got {list(metrics.keys())}"
    assert "nll" in metrics, f"nll missing; got {list(metrics.keys())}"
    assert "loss" in metrics


def test_invariant_extra_loss_keys_converted_via_to_metric():
    """Extra loss values are run through _to_metric, so tensors become floats (line 269)."""

    def _loss_with_tensor_extra(model_output, batch, ctx):
        pred = model_output.outputs["logits"]
        base = (pred - 1.0).pow(2).mean()
        return {"loss": base, "token_acc": torch.tensor(0.75)}

    torch.manual_seed(4)
    ctx, model, _ = _build_ctx(loss_fn=_loss_with_tensor_extra)
    rule = SAMUpdateRule(grad_clip=0.0)
    metrics = rule.step(model, _batch(), ctx)

    assert "token_acc" in metrics
    assert isinstance(metrics["token_acc"], float)
    assert pytest.approx(metrics["token_acc"], rel=1e-4) == 0.75


# ===========================================================================
# Accelerator path for backward and grad clip — lines 201-203, 241-243
# ===========================================================================


class _FakeAccelerator:
    """Minimal accelerator stub that records calls and delegates backward."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def backward(self, loss):
        self.calls.append("backward")
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        self.calls.append("clip_grad_norm_")
        return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def test_invariant_accelerator_backward_used_when_present():
    """When accelerator has .backward, SAM uses it for both backward passes."""
    torch.manual_seed(5)
    acc = _FakeAccelerator()
    model = _TinyModel()
    ctx, model, _ = _build_ctx(model=model)
    ctx.accelerator = acc
    rule = SAMUpdateRule(grad_clip=1.0)
    rule.step(model, _batch(), ctx)
    # backward is called twice (pass 1 + pass 2)
    assert acc.calls.count("backward") == 2


def test_invariant_accelerator_clip_grad_norm_used_when_present():
    """When accelerator has .clip_grad_norm_, SAM uses it instead of torch.nn.utils."""
    torch.manual_seed(6)
    acc = _FakeAccelerator()
    model = _TinyModel()
    ctx, model, _ = _build_ctx(model=model)
    ctx.accelerator = acc
    rule = SAMUpdateRule(grad_clip=1.0)
    rule.step(model, _batch(), ctx)
    assert "clip_grad_norm_" in acc.calls


# ===========================================================================
# setup() called inside step via model attribute — contract variant
# ===========================================================================


def test_pin_current_behavior_setup_is_noop():
    """setup() is a no-op — calling it twice on the same rule is idempotent.

    Pinning current behavior: SAMUpdateRule.setup does not mutate self.
    """
    rule = SAMUpdateRule(rho=0.1)
    rule.setup(model=_TinyModel(), sample={"x": torch.zeros(2, 4)})
    rule.setup(model=_TinyModel(), sample={"x": torch.ones(2, 4)})
    # No attribute should have changed
    assert rule.rho == 0.1
    assert rule._micro_step == 0


# ===========================================================================
# _current_lr with wrapped optimizer (has .optimizer attribute) — line 46
# ===========================================================================


def test_invariant_current_lr_unwraps_inner_optimizer():
    """_current_lr reads param_groups from inner .optimizer when present (line 46)."""
    param = nn.Parameter(torch.zeros(1))
    inner = torch.optim.SGD([param], lr=0.05)

    class _Wrapped:
        optimizer = inner

    result = _current_lr(_Wrapped())
    assert pytest.approx(result, rel=1e-6) == 0.05
