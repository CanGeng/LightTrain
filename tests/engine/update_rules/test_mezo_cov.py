"""Edge-case / gap-coverage tests for MeZOUpdateRule and its private helpers.

Uncovered lines targeted (source: lighttrain/builtin_plugins/engine/update_rules/mezo.py):

  36-37  _to_metric: isinstance(value, torch.Tensor) → float via .detach().item()
  38-39  _to_metric: successful float() coercion of a plain Python number
  40-41  _to_metric: TypeError/ValueError branch → float("nan")
  48     _current_lr: "no param_groups" guard → return 0.0
  76     setup(): exists and returns None (no-op; still needs coverage)
  143    _forward inner fn: model returns a plain Mapping (dict) — wraps into
         ModelOutput(outputs=dict(_out)) rather than the non-Mapping tensor path
  147    _forward inner fn: loss_fn returns a dict without "loss" key → KeyError
  179    extra keys in loss_dict routed through _to_metric(v) for each k != "loss"

General additional edge cases also added for robustness.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.mezo import (
    MeZOUpdateRule,
    _current_lr,
    _to_metric,
)
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _ConstModel(nn.Module):
    """Model that returns a fixed scalar logit, independent of weights."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x, **_):
        # Returns ModelOutput (the standard path)
        return ModelOutput(outputs={"logits": self.w * x.mean()})


class _DictOutputModel(nn.Module):
    """Model whose forward returns a plain dict (a Mapping, not ModelOutput).

    Triggers line 143: ``isinstance(_out, Mapping)`` branch in _forward.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x, **_):
        # returns a dict (Mapping) — NOT a ModelOutput
        return {"logits": self.w * x.mean()}


class _TensorOutputModel(nn.Module):
    """Model whose forward returns a raw Tensor (not a Mapping and not a ModelOutput).

    Triggers the non-Mapping tensor-wrapping branch of _forward (line 142).
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x, **_):
        return self.w * x.mean()  # plain tensor


def _simple_loss(out, batch, ctx):
    """Loss that returns only 'loss', constant w.r.t. params."""
    return {"loss": torch.tensor(1.0)}


def _loss_with_extras(out, batch, ctx):
    """Loss that returns extra metrics alongside 'loss' (covers line 178-179)."""
    return {
        "loss": torch.tensor(0.5),
        "aux_tensor": torch.tensor(3.14),   # _to_metric: tensor branch (36-37)
        "aux_float": 2.71,                  # _to_metric: float() branch (38-39)
        "aux_bad": object(),               # _to_metric: TypeError/ValueError → nan (40-41)
    }


def _loss_missing_key(out, batch, ctx):
    """Loss that omits 'loss' — triggers KeyError at line 147."""
    return {"not_loss": torch.tensor(1.0)}


def _build_ctx(model, loss_fn=_simple_loss, lr: float = 0.01):
    """Build a minimal StepContext for the given model."""
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=EventBus([]),
        loss_fn=loss_fn,
    )
    return ctx, optim


def _batch():
    torch.manual_seed(0)
    return {"x": torch.randn(2, 4)}


# ===========================================================================
# _to_metric — lines 36-41
# ===========================================================================


def test_to_metric_tensor_returns_float():
    """_to_metric: a scalar Tensor goes through the isinstance branch (line 36-37)."""
    t = torch.tensor(3.14)
    result = _to_metric(t)
    assert isinstance(result, float)
    assert result == pytest.approx(3.14, rel=1e-5)


def test_to_metric_tensor_detach_not_grad():
    """_to_metric: a Tensor requiring grad is detached cleanly without raising."""
    t = torch.tensor(2.0, requires_grad=True)
    result = _to_metric(t)
    assert isinstance(result, float)
    assert result == pytest.approx(2.0)


def test_to_metric_plain_float_branch():
    """_to_metric: a plain float goes through float() in the try block (line 38-39)."""
    result = _to_metric(1.23)
    assert result == pytest.approx(1.23)


def test_to_metric_plain_int_branch():
    """_to_metric: an int is coercible to float (try branch, line 38-39)."""
    result = _to_metric(7)
    assert result == pytest.approx(7.0)


def test_to_metric_unconvertible_returns_nan():
    """_to_metric: object() triggers ValueError/TypeError → float('nan') (lines 40-41)."""
    result = _to_metric(object())
    assert math.isnan(result)


def test_to_metric_none_returns_nan():
    """_to_metric: None is not a Tensor and float(None) raises TypeError → nan."""
    result = _to_metric(None)
    assert math.isnan(result)


def test_to_metric_string_returns_nan():
    """_to_metric: a non-numeric string triggers ValueError → nan (lines 40-41)."""
    result = _to_metric("hello")
    assert math.isnan(result)


def test_to_metric_numeric_string_branch():
    """_to_metric: a numeric string like '3.14' succeeds via float() (line 39)."""
    result = _to_metric("3.14")
    assert result == pytest.approx(3.14)


# ===========================================================================
# _current_lr — line 48
# ===========================================================================


def test_current_lr_no_param_groups_returns_zero():
    """_current_lr: an optimizer-like with empty param_groups returns 0.0 (line 48)."""

    class _NoGroups:
        optimizer = None
        param_groups = []  # falsy → branch at line 47-48

    result = _current_lr(_NoGroups())
    assert result == 0.0


def test_current_lr_none_param_groups_returns_zero():
    """_current_lr: optimizer whose param_groups attr is None returns 0.0 (line 48)."""

    class _NoneGroups:
        param_groups = None

    result = _current_lr(_NoneGroups())
    assert result == 0.0


def test_current_lr_wraps_inner_optimizer():
    """_current_lr: reads from optimizer.optimizer.param_groups (the .inner chain)."""
    inner = MagicMock()
    inner.param_groups = [{"lr": 0.05}]
    outer = MagicMock()
    outer.optimizer = inner
    # Make sure outer itself does NOT have param_groups so the inner path is taken
    del outer.param_groups
    result = _current_lr(outer)
    assert result == pytest.approx(0.05)


def test_current_lr_reads_group_zero_lr():
    """_current_lr: reads lr from first param_group of a real SGD optimizer."""
    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.007)
    result = _current_lr(optim)
    assert result == pytest.approx(0.007)


# ===========================================================================
# setup() — line 76
# ===========================================================================


def test_setup_returns_none():
    """setup() is a no-op that returns None (line 76)."""
    rule = MeZOUpdateRule()
    result = rule.setup(model=MagicMock(), sample=MagicMock())  # type: ignore[func-returns-value]
    assert result is None


def test_setup_accepts_arbitrary_args():
    """setup() ignores both args (ARG002 noqa); calling with complex objects is safe."""
    rule = MeZOUpdateRule()
    # Neither model nor sample should cause any error
    result = rule.setup(model=None, sample={"input_ids": torch.zeros(1)})  # type: ignore[func-returns-value]
    assert result is None


# ===========================================================================
# _forward inner fn: Mapping output model — line 143
# ===========================================================================


def test_step_dict_output_model_is_wrapped_into_model_output():
    """When model returns a dict (Mapping), _forward wraps it into ModelOutput
    via the ``isinstance(_out, Mapping)`` branch (line 143).

    Checks that MeZO can complete a step end-to-end with such a model.
    """
    model = _DictOutputModel()
    ctx, _ = _build_ctx(model, loss_fn=_simple_loss)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)

    # Must not raise; loss must be reported in metrics
    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics


def test_step_tensor_output_model_is_wrapped_with_logits_key():
    """When model returns a raw Tensor (not a Mapping), _forward wraps it with
    outputs={'logits': tensor} (the non-Mapping branch on line 142).
    """
    model = _TensorOutputModel()
    ctx, _ = _build_ctx(model, loss_fn=_simple_loss)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)

    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics


# ===========================================================================
# _forward inner fn: missing 'loss' key — line 147
# ===========================================================================


def test_step_loss_fn_without_loss_key_raises_key_error():
    """If loss_fn returns a dict without 'loss', step() raises KeyError (line 147).

    Pins current behavior: the user is told immediately that their LossFn is broken.
    """
    model = _ConstModel()
    ctx, _ = _build_ctx(model, loss_fn=_loss_missing_key)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)

    with pytest.raises(KeyError, match="loss"):
        rule.step(model, _batch(), ctx)


# ===========================================================================
# Extra loss_dict values routed through _to_metric — lines 177-179
# ===========================================================================


def test_step_extra_loss_dict_keys_appear_in_metrics():
    """Loss dicts with extra keys ('aux_*') are routed through _to_metric and
    stored in ctx.metrics (lines 177-179). The returned dict reflects those keys.
    """
    model = _ConstModel()
    ctx, _ = _build_ctx(model, loss_fn=_loss_with_extras)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)
    torch.manual_seed(1)
    metrics = rule.step(model, _batch(), ctx)

    # aux_tensor is a Tensor → _to_metric converts to float (line 36-37)
    assert "aux_tensor" in metrics
    assert isinstance(metrics["aux_tensor"], float)
    assert metrics["aux_tensor"] == pytest.approx(3.14, rel=1e-4)

    # aux_float is a plain float → _to_metric returns it (line 38-39)
    assert "aux_float" in metrics
    assert metrics["aux_float"] == pytest.approx(2.71, rel=1e-4)

    # aux_bad is an object() → _to_metric returns nan (lines 40-41)
    assert "aux_bad" in metrics
    assert math.isnan(metrics["aux_bad"])


def test_step_extra_loss_dict_loss_key_not_duplicated():
    """The 'loss' key from loss_dict is NOT passed through _to_metric again;
    only keys != 'loss' are forwarded (line 178 `if k != 'loss'`).
    """
    model = _ConstModel()
    ctx, _ = _build_ctx(model, loss_fn=_loss_with_extras)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)
    torch.manual_seed(2)
    metrics = rule.step(model, _batch(), ctx)
    # 'loss' should be the averaged L+/L- value, NOT the raw tensor value from
    # loss_dict["loss"] re-applied through _to_metric
    assert "loss" in metrics
    # Average of the two forward passes' loss values (both will be near 0.5
    # since _loss_with_extras returns a constant 0.5 tensor)
    assert metrics["loss"] == pytest.approx(0.5, abs=1e-5)


# ===========================================================================
# load_state_dict edge cases
# ===========================================================================


def test_load_state_dict_missing_keys_use_defaults():
    """load_state_dict with partially empty dict keeps existing values as defaults."""
    rule = MeZOUpdateRule(eps=5e-3, seed_per_step=False)
    rule._step_count = 10
    # Empty mapping: every .get(..., self.xxx) should return the existing value
    rule.load_state_dict({})
    assert rule.eps == pytest.approx(5e-3)
    assert rule.seed_per_step is False
    assert rule._step_count == 0  # step_count falls back to sd.get("step_count", 0)


def test_load_state_dict_overrides_all_fields():
    """load_state_dict with all keys set correctly overrides every field."""
    rule = MeZOUpdateRule()
    rule.load_state_dict({"eps": 1e-4, "seed_per_step": False, "step_count": 99})
    assert rule.eps == pytest.approx(1e-4)
    assert rule.seed_per_step is False
    assert rule._step_count == 99


# ===========================================================================
# bus=None path (no EventBus)
# ===========================================================================


def test_step_with_no_bus_does_not_crash():
    """step() tolerates ctx.bus=None (branches at lines 129-130, 165-166, 181-182)."""
    model = _ConstModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=None,  # explicit None
        loss_fn=_simple_loss,
    )
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)
    metrics = rule.step(model, _batch(), ctx)
    assert "loss" in metrics


# ===========================================================================
# step_count increments
# ===========================================================================


def test_step_count_increments_each_call():
    """_step_count grows monotonically: 1 after first step, 2 after second."""
    model = _ConstModel()
    ctx, _ = _build_ctx(model)
    rule = MeZOUpdateRule(eps=1e-3)
    assert rule._step_count == 0

    rule.step(model, _batch(), ctx)
    assert rule._step_count == 1

    rule.step(model, _batch(), ctx)
    assert rule._step_count == 2


# ===========================================================================
# metrics keys presence
# ===========================================================================


def test_step_metrics_include_grad_norm_and_skipped():
    """step() returns 'grad_norm' and 'skipped' in addition to the standard keys."""
    model = _ConstModel()
    ctx, _ = _build_ctx(model)
    rule = MeZOUpdateRule(eps=1e-3, seed_per_step=False)
    torch.manual_seed(3)
    metrics = rule.step(model, _batch(), ctx)
    assert "grad_norm" in metrics
    assert "skipped" in metrics
    assert metrics["skipped"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0
