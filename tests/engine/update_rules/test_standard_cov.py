"""Coverage-completion tests for StandardUpdateRule.

Pins the remaining uncovered branches in:
  lighttrain/builtin_plugins/engine/update_rules/standard.py

Targeted lines:
  - 81   : setup() return None
  - 127-132 : rng_state() exception -> warning + rng_snap = None
  - 153  : model output is a plain Mapping -> wrapped in ModelOutput
  - 203-204 : frozen_step_writer.restore_snapshot() raises -> warning + continue
  - 211-212 : restore_rng_state() raises during RETRY -> warning + continue
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.standard import StandardUpdateRule
from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Minimal shared helpers (kept independent of existing test_standard.py)
# ---------------------------------------------------------------------------


class _LinearModel(nn.Module):
    """A minimal nn.Module returning plain dicts or ModelOutput on demand."""

    def __init__(self, return_type: str = "model_output") -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)
        self._return_type = return_type

    def forward(self, x):
        out = self.linear(x)
        if self._return_type == "model_output":
            return ModelOutput(outputs={"logits": out})
        elif self._return_type == "mapping":
            # Returns a plain dict, which is a Mapping but NOT a ModelOutput.
            # This exercises line 153-156 (the Mapping branch in _run_forward_and_loss).
            return {"logits": out}
        else:
            # Returns a bare tensor (non-Mapping, non-ModelOutput).
            # This exercises line 156 (the ``else {"logits": _outputs}`` branch).
            return out


def _simple_loss(model_output, batch, ctx):
    """Basic MSE loss — produces real gradients."""
    pred = model_output.outputs["logits"]
    loss = (pred - 1.0).pow(2).mean()
    return {"loss": loss}


def _build_ctx(
    *,
    model=None,
    callbacks=None,
    scheduler=None,
    accelerator=None,
) -> tuple[StepContext, nn.Module, torch.optim.Optimizer]:
    if model is None:
        model = _LinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bus = EventBus(callbacks or [])
    ctx = StepContext(
        model=model,
        optimizer=optimizer,
        bus=bus,
        loss_fn=_simple_loss,
        scheduler=scheduler,
        accelerator=accelerator,
    )
    return ctx, model, optimizer


def _batch():
    torch.manual_seed(0)
    return {"x": torch.randn(2, 4)}


# ===========================================================================
# Line 81 — setup() returns None
# ===========================================================================


def test_invariant_setup_returns_none():
    """setup() is a no-op that returns None; callers may check the return value."""
    rule = StandardUpdateRule()
    result = rule.setup(model=MagicMock(), sample={"x": torch.zeros(1)})  # type: ignore[func-returns-value]
    assert result is None


def test_invariant_setup_accepts_arbitrary_args():
    """setup() accepts any model and sample without raising."""
    rule = StandardUpdateRule()
    # Should not raise for any combination of types.
    assert rule.setup(model=None, sample=None) is None  # type: ignore[func-returns-value]
    assert rule.setup(model=MagicMock(), sample=42) is None  # type: ignore[func-returns-value]


# ===========================================================================
# Lines 127-132 — rng_state() raises => warning logged + rng_snap = None
# ===========================================================================


def test_pin_current_behavior_rng_snapshot_failure_logs_warning_and_continues():
    """When rng_state() raises, StandardUpdateRule logs a warning and sets
    rng_snap = None, then the step completes normally (no propagation of the
    exception).

    NOTE: This pins current behavior: the exception is intentionally swallowed
    (BLE001 suppressed) so a snapshot failure is non-fatal. If a future commit
    makes this fatal, this test should be updated to reflect that.
    """
    module_path = "lighttrain.builtin_plugins.engine.update_rules.standard.rng_state"
    ctx, model, _ = _build_ctx()

    with patch(module_path, side_effect=RuntimeError("simulated rng_state failure")):
        # step should complete; the warning goes through logging, not warnings.warn
        metrics = StandardUpdateRule().step(model, _batch(), ctx)

    # The step continued to produce loss despite the RNG snapshot failure.
    assert "loss" in metrics
    assert metrics.get("skipped", 0.0) == 0.0


def test_pin_current_behavior_rng_snapshot_failure_emits_log_warning(caplog):
    """When rng_state() raises, a WARNING is emitted via the module logger.

    Pins the exact log-level and message fragment so refactors that delete
    the warning (or change level to DEBUG) are caught.
    """
    import logging

    module_path = "lighttrain.builtin_plugins.engine.update_rules.standard.rng_state"
    ctx, model, _ = _build_ctx()

    with caplog.at_level(logging.WARNING, logger="lighttrain.builtin_plugins.engine.update_rules.standard"):
        with patch(module_path, side_effect=OSError("snap failed")):
            StandardUpdateRule().step(model, _batch(), ctx)

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("RNG snapshot failed" in m for m in warning_msgs), warning_msgs


# ===========================================================================
# Line 153 — model returns a plain Mapping (dict) => wrapped in ModelOutput
# ===========================================================================


def test_invariant_model_returning_mapping_is_wrapped_in_model_output():
    """When the model returns a plain dict (a Mapping but not ModelOutput),
    the update rule must wrap it into ModelOutput before calling the loss fn.

    Concretely: the loss_fn receives a ModelOutput whose .outputs has the
    same keys as the dict the model returned.
    """
    received_outputs: list = []

    def _recording_loss(model_output, batch, ctx):
        received_outputs.append(model_output)
        # Must be a ModelOutput by now.
        pred = model_output.outputs["logits"]
        loss = (pred - 1.0).pow(2).mean()
        return {"loss": loss}

    model: nn.Module = _LinearModel(return_type="mapping")
    ctx, model, _ = _build_ctx(model=model)
    ctx.loss_fn = _recording_loss

    metrics = StandardUpdateRule().step(model, _batch(), ctx)

    assert len(received_outputs) == 1, "loss_fn should be called exactly once"
    mo = received_outputs[0]
    assert isinstance(mo, ModelOutput), f"Expected ModelOutput, got {type(mo)}"
    assert "logits" in mo.outputs
    assert metrics.get("skipped", 0.0) == 0.0


def test_invariant_model_returning_bare_tensor_is_wrapped_in_model_output():
    """When the model returns a non-Mapping, non-ModelOutput (e.g. a Tensor),
    it is wrapped as ModelOutput(outputs={'logits': tensor}).

    This exercises the ``else {'logits': _outputs}`` branch on line 156.
    """
    received_outputs: list = []

    def _recording_loss(model_output, batch, ctx):
        received_outputs.append(model_output)
        pred = model_output.outputs["logits"]
        loss = (pred - 1.0).pow(2).mean()
        return {"loss": loss}

    model: nn.Module = _LinearModel(return_type="bare_tensor")
    ctx, model, _ = _build_ctx(model=model)
    ctx.loss_fn = _recording_loss

    StandardUpdateRule().step(model, _batch(), ctx)

    assert len(received_outputs) == 1
    mo = received_outputs[0]
    assert isinstance(mo, ModelOutput)
    assert "logits" in mo.outputs


# ===========================================================================
# Lines 203-204 — restore_snapshot() raises => warning + step continues
# ===========================================================================


def test_pin_current_behavior_restore_snapshot_failure_logs_warning_and_continues(caplog):
    """When frozen_step_writer.restore_snapshot() raises during RETRY_STEP,
    a WARNING is logged and the retry continues on unrestored params — not fatal.

    NOTE: pins current "swallow and warn" behavior. If this branch is hardened
    to re-raise, update accordingly.
    """
    import logging

    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            return None

    writer = MagicMock()
    writer.restore_snapshot = MagicMock(side_effect=RuntimeError("snap restore exploded"))

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])
    ctx.frozen_step_writer = writer

    with caplog.at_level(logging.WARNING, logger="lighttrain.builtin_plugins.engine.update_rules.standard"):
        metrics = StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    # The step still completed (no exception propagated).
    assert "loss" in metrics

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("snapshot restore failed" in m for m in warning_msgs), warning_msgs


def test_pin_current_behavior_restore_snapshot_failure_still_runs_retry():
    """Even when restore_snapshot() raises, the retry forward+loss still runs.

    Concretely: on_forward_post fires during the retry (not short-circuited),
    which means the retry loop continued after the restore_snapshot exception.
    """
    forward_post_count = [0]
    retries_left = [1]

    class _Retrier:
        def on_forward_post(self, **_):
            forward_post_count[0] += 1

        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            return None

    writer = MagicMock()
    writer.restore_snapshot = MagicMock(side_effect=ValueError("broken checkpoint"))

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])
    ctx.frozen_step_writer = writer

    StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    # Initial forward + 1 retry = 2 on_forward_post calls.
    assert forward_post_count[0] >= 2, forward_post_count[0]


# ===========================================================================
# Lines 211-212 — restore_rng_state() raises during RETRY => warning + continue
# ===========================================================================


def test_pin_current_behavior_rng_restore_failure_logs_warning_and_continues(caplog):
    """When restore_rng_state() raises during a RETRY_STEP, a WARNING is logged
    and the retry proceeds with the current (unrestored) RNG state — not fatal.

    NOTE: pins the "swallow and warn" contract. A future hardening that re-raises
    should update this test.
    """
    import logging

    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            return None

    rng_restore_path = (
        "lighttrain.builtin_plugins.engine.update_rules.standard.restore_rng_state"
    )

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])

    with caplog.at_level(logging.WARNING, logger="lighttrain.builtin_plugins.engine.update_rules.standard"):
        with patch(rng_restore_path, side_effect=RuntimeError("rng restore boom")):
            metrics = StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    # Step completed despite the restore failure.
    assert "loss" in metrics

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("RNG restore failed" in m for m in warning_msgs), warning_msgs


def test_pin_current_behavior_rng_restore_failure_still_runs_retry_forward():
    """After restore_rng_state() raises, the retry's forward+loss still executes.

    Concretely: on_forward_post fires during the retry iteration even though
    RNG restore raised — the exception is swallowed and execution continues.
    """
    forward_post_count = [0]
    retries_left = [1]

    class _Retrier:
        def on_forward_post(self, **_):
            forward_post_count[0] += 1

        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            return None

    rng_restore_path = (
        "lighttrain.builtin_plugins.engine.update_rules.standard.restore_rng_state"
    )

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])

    with patch(rng_restore_path, side_effect=OSError("rng broken")):
        StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    # Initial forward + 1 retry = 2 on_forward_post calls.
    assert forward_post_count[0] >= 2, forward_post_count[0]


# ===========================================================================
# Combined edge-cases: rng_snap = None path during RETRY
# (rng_state() failed at step entry, so rng_snap is None; RETRY should not
#  attempt restore_rng_state — the ``if rng_snap is not None`` guard on line 208)
# ===========================================================================


def test_invariant_retry_does_not_call_restore_rng_when_snap_is_none():
    """When rng_state() fails at step entry (rng_snap = None), a subsequent
    RETRY_STEP must NOT attempt restore_rng_state — the guard ``if rng_snap
    is not None`` (line 208) protects against that.

    We verify by patching restore_rng_state and asserting it is never called
    even though a retry fires.
    """
    rng_state_path = (
        "lighttrain.builtin_plugins.engine.update_rules.standard.rng_state"
    )
    rng_restore_path = (
        "lighttrain.builtin_plugins.engine.update_rules.standard.restore_rng_state"
    )
    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            return None

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])
    restore_mock = MagicMock()

    with patch(rng_state_path, side_effect=RuntimeError("snap failed")):
        with patch(rng_restore_path, restore_mock):
            StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    restore_mock.assert_not_called()
