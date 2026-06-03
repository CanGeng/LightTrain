"""Adversarial tests for StandardEngine — the thin orchestrator at
``lighttrain/builtin_plugins/engine/standard.py``.

The engine has a tiny surface: it injects ``loss_fn`` and ``accelerator``
into ``ctx`` if missing, then directly delegates to
``update_rule.step(ctx.model, batch, ctx)``. Existing tests
(``tests/test_engine_standard.py``) verify weights update, SKIP_STEP
aborts backward, and accumulation holds off optimizer. This file pins the
delegation invariants those tests don't check:

  - The "if None" guards must not overwrite trainer-set fields.
  - ``ctx.model`` (not ``self.model``) is the source of truth for the model.
  - The return value is the update_rule's metrics dict, unmodified.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from lighttrain.engine._context import StepContext
from lighttrain.builtin_plugins.engine.standard import StandardEngine


def _stub_update_rule(return_metrics: dict[str, Any] | None = None):
    """Build a recording stub UpdateRule.

    Captures the (model, batch, ctx) it was called with so tests can assert
    on identity/value of each.
    """
    rule = SimpleNamespace()
    rule.calls = []

    def _step(model, batch, ctx):
        rule.calls.append({"model": model, "batch": batch, "ctx": ctx})
        return dict(return_metrics or {"loss": 0.42, "lr": 1e-3, "skipped": 0.0})

    rule.step = _step
    return rule


def test_engine_injects_loss_fn_when_ctx_loss_fn_is_none():
    """Goal: engine fills ``ctx.loss_fn`` from ``self.loss_fn`` when ctx has None.

    Input: StepContext with loss_fn=None; engine constructed with a sentinel loss_fn.
    Expected: after ``engine.step``, ``ctx.loss_fn is sentinel``.

    Catches a refactor that drops the ``if ctx.loss_fn is None`` guard
    (line 31-32 in engine/standard.py) and always overwrites — or never
    writes — the slot.
    """
    sentinel = object()
    rule = _stub_update_rule()
    engine = StandardEngine(update_rule=rule, loss_fn=sentinel)
    ctx = StepContext(model=MagicMock(), optimizer=MagicMock())
    assert ctx.loss_fn is None

    engine.step({}, ctx)

    assert ctx.loss_fn is sentinel


def test_engine_does_not_overwrite_existing_ctx_loss_fn():
    """Goal: if ctx already carries a loss_fn (set by trainer), the engine MUST
    NOT replace it with its own.

    Input: ctx.loss_fn = trainer_set; engine.loss_fn = engine_default.
    Expected: after step, ctx.loss_fn is still trainer_set.

    Catches a refactor that inverts the guard (``if ctx.loss_fn is not None``)
    or drops the guard entirely — both would silently switch a trainer's
    custom RL loss to a stale engine default.
    """
    trainer_set = object()
    engine_default = object()
    rule = _stub_update_rule()
    engine = StandardEngine(update_rule=rule, loss_fn=engine_default)
    ctx = StepContext(model=MagicMock(), optimizer=MagicMock(), loss_fn=trainer_set)

    engine.step({}, ctx)

    assert ctx.loss_fn is trainer_set


def test_engine_injects_accelerator_when_ctx_accelerator_is_none():
    """Symmetric to loss_fn: accelerator slot must be filled if empty."""
    accel_sentinel = object()
    rule = _stub_update_rule()
    engine = StandardEngine(update_rule=rule, accelerator=accel_sentinel)
    ctx = StepContext(model=MagicMock(), optimizer=MagicMock())
    assert ctx.accelerator is None

    engine.step({}, ctx)

    assert ctx.accelerator is accel_sentinel


def test_engine_does_not_overwrite_existing_ctx_accelerator():
    """Symmetric to loss_fn: trainer-set accelerator wins."""
    trainer_accel = object()
    engine_accel = object()
    rule = _stub_update_rule()
    engine = StandardEngine(update_rule=rule, accelerator=engine_accel)
    ctx = StepContext(
        model=MagicMock(), optimizer=MagicMock(), accelerator=trainer_accel
    )

    engine.step({}, ctx)

    assert ctx.accelerator is trainer_accel


def test_engine_passes_ctx_model_not_engine_model_to_update_rule():
    """Goal: pin the contract that the engine reads ``ctx.model``, never
    ``self.model`` (the engine doesn't even own a ``self.model``).

    Input: ctx.model is a specific object; we record what the update_rule
    receives as its first positional arg.

    Catches a refactor that adds ``self.model`` to the engine and passes
    that instead — would silently route the wrong model in trainers that
    swap models mid-run (eg. EMA / ref policy).
    """
    rule = _stub_update_rule()
    engine = StandardEngine(update_rule=rule, loss_fn=object())
    model_obj = object()
    ctx = StepContext(model=model_obj, optimizer=MagicMock(), loss_fn=object())

    engine.step({"k": 1}, ctx)

    assert len(rule.calls) == 1
    assert rule.calls[0]["model"] is model_obj
    assert rule.calls[0]["batch"] == {"k": 1}
    assert rule.calls[0]["ctx"] is ctx


def test_engine_step_returns_metrics_dict_unchanged():
    """Goal: engine must return whatever the update_rule returned, byte-for-byte.

    Input: stub update_rule returns a dict with custom keys.
    Expected: engine.step returns the same dict (same keys, same values).

    Catches a refactor that wraps the return (adds a "engine_version" key,
    converts to StepOutput, etc.) — breaks ``trainer.train_step`` which
    expects a plain dict.
    """
    payload = {"loss": 1.23, "grad_norm": 4.56, "custom_key": 99.9}
    rule = _stub_update_rule(return_metrics=payload)
    engine = StandardEngine(update_rule=rule, loss_fn=object())
    ctx = StepContext(model=MagicMock(), optimizer=MagicMock(), loss_fn=object())

    out = engine.step({}, ctx)

    assert out == payload
    # ensure no spurious extra keys
    assert set(out.keys()) == set(payload.keys())


def test_engine_step_propagates_update_rule_exception():
    """Goal: errors from update_rule must propagate (no silent swallow).

    Input: a stub rule whose ``step`` raises RuntimeError.
    Expected: the same RuntimeError bubbles out of ``engine.step``.

    Catches a refactor that wraps update_rule.step in try/except and turns
    a crash into a metric flag — would mask broken training.
    """
    rule = SimpleNamespace()

    def _boom(model, batch, ctx):
        raise RuntimeError("simulated update_rule failure")

    rule.step = _boom
    engine = StandardEngine(update_rule=rule, loss_fn=object())
    ctx = StepContext(model=MagicMock(), optimizer=MagicMock(), loss_fn=object())

    with pytest.raises(RuntimeError, match="simulated update_rule failure"):
        engine.step({}, ctx)


def test_engine_constructor_stores_components_by_reference():
    """Goal: engine stores update_rule/loss_fn/accelerator by reference, not by copy.

    Catches a refactor that deepcopies any of these — would break weight-tying
    or shared state across components.
    """
    rule = _stub_update_rule()
    loss = object()
    accel = object()
    engine = StandardEngine(update_rule=rule, loss_fn=loss, accelerator=accel)

    assert engine.update_rule is rule
    assert engine.loss_fn is loss
    assert engine.accelerator is accel
