"""LayerOffloadEngine — DESIGN §14 (M5).

CPU-side correctness checks:

* LayerOffloadEngine is a drop-in replacement for StandardEngine in terms
  of return contract and EventBus events.
* Numerically equivalent to StandardEngine when seeded identically — loss
  values match within fp32 tolerance after one full step.
* All swap_in / swap_out hooks fire without raising on tiny_lm.
"""

from __future__ import annotations

import pytest
import torch

# Force eager registration.
import lighttrain.plugins.layer_offload  # noqa: F401

from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.engine.standard import StandardEngine
from lighttrain.losses.core import CrossEntropyLoss
from lighttrain.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.registry import get as _registry_get
from lighttrain.update_rules.standard import StandardUpdateRule


def _seed_everything(seed: int = 0):
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def _make_components(seed: int = 0):
    _seed_everything(seed)
    model = TinyCausalLM(
        vocab_size=64, d_model=16, n_layers=3, n_heads=4, max_seq_len=16
    )
    loss_fn = CrossEntropyLoss()
    ids = torch.randint(0, 64, (2, 8))
    batch = {"input_ids": ids, "labels": ids.clone()}
    return model, loss_fn, batch


def _take_one_step(engine, model, loss_fn, batch):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    update_rule = StandardUpdateRule(grad_clip=0.0)
    ctx = StepContext(
        step=0, model=model, optimizer=optimizer, loss_fn=loss_fn, bus=EventBus()
    )
    return engine.step(batch, ctx), ctx


def test_layer_offload_engine_registered():
    cls = _registry_get("engine", "layer_offload")
    assert cls is not None


def test_layer_offload_step_returns_metrics_with_loss():
    model, loss_fn, batch = _make_components(seed=0)
    update_rule = StandardUpdateRule()
    LO = _registry_get("engine", "layer_offload")
    engine = LO(update_rule=update_rule, loss_fn=loss_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ctx = StepContext(
        step=0, model=model, optimizer=optimizer, loss_fn=loss_fn, bus=EventBus()
    )
    metrics = engine.step(batch, ctx)
    assert "loss" in metrics
    assert metrics["loss"] > 0


def test_layer_offload_loss_matches_standard_engine_at_lr_zero():
    """With lr=0 (no weight update), loss after one step must match. We use
    SGD lr=0 + grad_clip=0 so the *only* operation is forward + loss; any
    drift comes from the offload path itself."""
    # Build two independent copies with identical seeds.
    m1, lf, b1 = _make_components(seed=42)
    m2, _, b2 = _make_components(seed=42)
    # Standard.
    update_rule1 = StandardUpdateRule(grad_clip=0.0)
    e1 = StandardEngine(update_rule=update_rule1, loss_fn=lf)
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.0)
    ctx1 = StepContext(step=0, model=m1, optimizer=opt1, loss_fn=lf, bus=EventBus())
    out1 = e1.step(b1, ctx1)
    # LayerOffload.
    update_rule2 = StandardUpdateRule(grad_clip=0.0)
    LO = _registry_get("engine", "layer_offload")
    e2 = LO(update_rule=update_rule2, loss_fn=lf, resident_layers=1, prefetch=0)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.0)
    ctx2 = StepContext(step=0, model=m2, optimizer=opt2, loss_fn=lf, bus=EventBus())
    out2 = e2.step(b2, ctx2)

    assert out2["loss"] == pytest.approx(out1["loss"], rel=1e-4, abs=1e-4)


def test_layer_offload_engine_paginates_layers_via_hooks(tmp_path):
    """After step, all layers should be on host (not device).

    On CPU-only runs the device is already 'cpu', so we just check that
    swap_out was called — verified indirectly through the storage stash
    being populated with every layer name.
    """
    model, loss_fn, batch = _make_components(seed=1)
    update_rule = StandardUpdateRule()
    LO = _registry_get("engine", "layer_offload")
    engine = LO(
        update_rule=update_rule,
        loss_fn=loss_fn,
        resident_layers=1,
        prefetch=1,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    ctx = StepContext(
        step=0, model=model, optimizer=opt, loss_fn=loss_fn, bus=EventBus()
    )
    engine.step(batch, ctx)
    # All three layers must have a host-side stash entry.
    stash = engine._weights_storage.stash
    assert set(stash.keys()) == {"block.0", "block.1", "block.2"}


def test_layer_offload_engine_close_removes_hooks():
    model, loss_fn, batch = _make_components(seed=2)
    update_rule = StandardUpdateRule()
    LO = _registry_get("engine", "layer_offload")
    engine = LO(update_rule=update_rule, loss_fn=loss_fn)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    ctx = StepContext(
        step=0, model=model, optimizer=opt, loss_fn=loss_fn, bus=EventBus()
    )
    engine.step(batch, ctx)
    n_hooks = len(engine._installed)
    assert n_hooks > 0
    engine.close()
    assert engine._installed == []
