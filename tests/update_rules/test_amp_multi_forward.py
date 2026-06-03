"""Regression: AMP autocast must be rebuilt per forward, not cached.

``accelerator.autocast()`` returns a single-use ``_GeneratorContextManager``;
re-entering the same object raises
``AttributeError: '_GeneratorContextManager' object has no attribute 'args'``.

Update rules that run multiple forwards per step crash if they cache the
context object:
  - StandardUpdateRule on a RETRY_STEP replay (2nd forward),
  - SAMUpdateRule on its second pass (always 2 forwards),
  - MeZOUpdateRule on its L- forward (always 2 forwards).

The conftest ``_FakeAccelerator.autocast()`` returns a *re-entrant*
``nullcontext()`` and so cannot reproduce this — these tests use a fake whose
``autocast()`` returns a genuine single-use CM, matching real Accelerate.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput
from lighttrain.builtin_plugins.update_rules.standard import StandardUpdateRule
from lighttrain.builtin_plugins.update_rules.sam import SAMUpdateRule
from lighttrain.builtin_plugins.update_rules.mezo import MeZOUpdateRule


class _SingleUseAutocastAccelerator:
    """Accelerator stub whose ``autocast()`` returns a *single-use* CM.

    Mirrors real ``Accelerator.autocast`` (a ``@contextmanager`` generator):
    entering the returned object twice raises ``AttributeError``. Counts how
    many times ``autocast()`` was called so tests can assert a fresh CM per
    forward.
    """

    def __init__(self) -> None:
        self.autocast_calls = 0

    def autocast(self):
        self.autocast_calls += 1

        @contextmanager
        def _cm():
            yield

        return _cm()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return ModelOutput(outputs={"logits": self.linear(x)})


def _simple_loss(model_output, batch, ctx):
    pred = model_output.outputs["logits"]
    return {"loss": (pred - 1.0).pow(2).mean()}


def _build_ctx(*, callbacks=None, accelerator=None):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optimizer,
        bus=EventBus(callbacks or []),
        loss_fn=_simple_loss,
        accelerator=accelerator,
    )
    return ctx, model


def _batch():
    return {"x": torch.randn(2, 4)}


def test_fake_autocast_is_genuinely_single_use():
    """Sanity: the stub reproduces the single-use crash (unlike nullcontext)."""
    acc = _SingleUseAutocastAccelerator()
    cm = acc.autocast()
    with cm:
        pass
    with pytest.raises(AttributeError, match="args"):
        with cm:
            pass


def test_standard_retry_replay_does_not_crash_under_amp():
    """RETRY_STEP forces a second forward; a fresh autocast CM must be built."""
    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP

    acc = _SingleUseAutocastAccelerator()
    ctx, model = _build_ctx(callbacks=[_Retrier()], accelerator=acc)

    StandardUpdateRule(max_retries=3).step(model, _batch(), ctx)

    # initial forward + one retry forward → two fresh autocast contexts
    assert acc.autocast_calls == 2


def test_sam_two_pass_does_not_crash_under_amp():
    """SAM always runs two forwards per step — each needs a fresh autocast CM."""
    acc = _SingleUseAutocastAccelerator()
    ctx, model = _build_ctx(accelerator=acc)

    SAMUpdateRule(grad_clip=1.0).step(model, _batch(), ctx)

    assert acc.autocast_calls == 2


def test_mezo_two_pass_does_not_crash_under_amp():
    """MeZO runs L+ and L- forwards per step — each needs a fresh autocast CM."""
    acc = _SingleUseAutocastAccelerator()
    ctx, model = _build_ctx(accelerator=acc)

    MeZOUpdateRule(eps=1e-3).step(model, _batch(), ctx)

    assert acc.autocast_calls == 2
