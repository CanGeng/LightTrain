"""Lazy hidden-state projection lifecycle for HiddenStatesMSELoss(project=True).

Mirror of the flat ``tests/test_hidden_projection.py`` suite (DESIGN §8.3 / §9.1).

These exercise the *integration* concerns of the projection path — not the
loss math (covered in ``test_distill.py``) — so they build a real
``TinyCausalLM`` student rather than the conftest dummy fixtures:

* dim-mismatch with ``project=False`` still raises (the M3 contract).
* ``project=True`` lazily creates an ``nn.Linear`` under
  ``model._distill_projections.*`` so it follows ``state_dict`` + ``to(device)``.
* the fresh ``Linear`` params get auto-registered for the optimizer via
  ``ctx.extras['_new_trainable_params']`` and actually update on ``step()``.
* overfit smoke: loss drops over 20 steps.
* ``state_dict`` round-trips the projection (so a resume picks it up).
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.losses.distill import HiddenStatesMSELoss
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.protocols import LossContext


def _make_student(d=16, output_hidden_states=True):
    return TinyCausalLM(
        vocab_size=64, d_model=d, n_layers=2, n_heads=4, max_seq_len=8,
        output_hidden_states=output_hidden_states,
    )


def _teacher_tensor(L=3, B=2, T=4, H_t=32, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, B, T, H_t, generator=g)


def test_projection_dim_mismatch_without_project_raises_runtime_error():
    """Goal: project=False on mismatched hidden dims raises the M3 contract error.

    Input: student d=16 vs teacher H_t=32, mapping {1:1}, project=False.
    Contract: must raise RuntimeError matching "hidden_mse hidden dim mismatch".
    """
    student = _make_student(d=16)
    teacher = _teacher_tensor(H_t=32)
    loss_fn = HiddenStatesMSELoss(mapping={1: 1}, project=False)
    ids = torch.randint(0, 64, (2, 4))
    out = student(input_ids=ids, output_hidden_states=True)
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "aux.teacher.hidden_states_layers": teacher,
    }
    ctx = LossContext(extras={"model": student})
    with pytest.raises(RuntimeError, match="hidden_mse hidden dim mismatch"):
        loss_fn(out, batch, ctx)


def test_projection_lazily_attached_under_model_and_exposed_for_optimizer():
    """Goal: project=True lazy-creates a Linear under ``model._distill_projections``,
            pushes its params to ``ctx.extras['_new_trainable_params']``, and
            includes it in ``state_dict``.
    """
    student = _make_student(d=16)
    teacher = _teacher_tensor(H_t=32)
    loss_fn = HiddenStatesMSELoss(mapping={1: 1}, project=True)
    ids = torch.randint(0, 64, (2, 4))
    out = student(input_ids=ids, output_hidden_states=True)
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "aux.teacher.hidden_states_layers": teacher,
    }
    ctx = LossContext(extras={"model": student})
    result = loss_fn(out, batch, ctx)
    assert "loss" in result
    # Projection lives on the model.
    sub_names = dict(student.named_modules())
    proj_keys = [k for k in sub_names if "_distill_projections" in k]
    assert proj_keys, f"projection wasn't attached: {list(sub_names.keys())[:5]}"
    # The projection params got pushed to ctx.extras for optimizer registration.
    new_params = ctx.extras.get("_new_trainable_params", [])
    assert len(new_params) >= 1
    # state_dict includes it.
    sd = student.state_dict()
    assert any("_distill_projections" in k for k in sd)


def test_projection_overfit_loss_decreases_over_twenty_steps():
    """Goal: optimizing a tiny student to a fixed teacher via the projection
            drives the loss down (final < 0.9 * initial) over 20 steps.
    """
    torch.manual_seed(0)
    student = _make_student(d=16)
    teacher = _teacher_tensor(H_t=32, seed=42)
    loss_fn = HiddenStatesMSELoss(mapping={1: 1}, project=True)
    ids = torch.randint(0, 64, (2, 4))
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "aux.teacher.hidden_states_layers": teacher,
    }
    # Build optimizer on the model BEFORE projection exists (mimics real flow).
    opt = torch.optim.AdamW(student.parameters(), lr=5e-3)
    ctx = LossContext(extras={"model": student})

    losses = []
    for _step in range(20):
        opt.zero_grad()
        out = student(input_ids=ids, output_hidden_states=True)
        loss = loss_fn(out, batch, ctx)["loss"]
        loss.backward()
        # Drain newly-created params on the first iteration (mimics what
        # StandardUpdateRule does).
        new_params = ctx.extras.pop("_new_trainable_params", None)
        if new_params:
            opt.add_param_group({"params": list(new_params), "lr": 5e-3})
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.9, (
        f"loss didn't decrease meaningfully: start={losses[0]:.4f}, "
        f"end={losses[-1]:.4f}, traj={losses}"
    )


def test_projection_round_trips_through_state_dict():
    """Goal: a projection created on one student round-trips through ``state_dict``
            into a freshly-projected student (so a resume restores it exactly).
    """
    student = _make_student(d=16)
    teacher = _teacher_tensor(H_t=32)
    loss_fn = HiddenStatesMSELoss(mapping={1: 1}, project=True)
    ids = torch.randint(0, 64, (2, 4))
    out = student(input_ids=ids, output_hidden_states=True)
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "aux.teacher.hidden_states_layers": teacher,
    }
    ctx = LossContext(extras={"model": student})
    loss_fn(out, batch, ctx)

    sd = {k: v.clone() for k, v in student.state_dict().items()}
    proj_keys = [k for k in sd if "_distill_projections" in k]
    assert proj_keys

    # Make a fresh student, attach a projection by re-running loss, then load.
    student2 = _make_student(d=16)
    loss_fn2 = HiddenStatesMSELoss(mapping={1: 1}, project=True)
    out2 = student2(input_ids=ids, output_hidden_states=True)
    ctx2 = LossContext(extras={"model": student2})
    loss_fn2(out2, batch, ctx2)
    student2.load_state_dict(sd, strict=True)

    sd2 = student2.state_dict()
    for k in proj_keys:
        assert torch.allclose(sd[k], sd2[k], atol=1e-6), f"{k} did not round-trip"


def test_projection_registered_with_optimizer_via_standard_update_rule():
    """Goal: full integration — the StandardUpdateRule drain step adds the
            projection params as a new optimizer param group and ``step()``
            actually updates them.
    """
    from lighttrain.builtin_plugins.engine.update_rules.standard import (
        StandardUpdateRule,
    )
    from lighttrain.callbacks.base import EventBus
    from lighttrain.engine._context import StepContext

    torch.manual_seed(0)
    student = _make_student(d=16)
    teacher = _teacher_tensor(H_t=32, seed=42)
    loss_fn = HiddenStatesMSELoss(mapping={1: 1}, project=True)
    ids = torch.randint(0, 64, (2, 4))
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "aux.teacher.hidden_states_layers": teacher,
    }
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
    initial_n_groups = len(optimizer.param_groups)
    update_rule = StandardUpdateRule(grad_clip=1.0)
    ctx = StepContext(
        step=0,
        model=student,
        optimizer=optimizer,
        loss_fn=loss_fn,
        bus=EventBus(),
    )
    metrics = update_rule.step(student, batch, ctx)
    # Loss reported.
    assert "loss" in metrics
    # Optimizer grew a param group with the projection.
    assert len(optimizer.param_groups) == initial_n_groups + 1
    new_group_params = optimizer.param_groups[-1]["params"]
    proj_weight = student._distill_projections.layer_1_s16_t32.weight
    assert any(p is proj_weight for p in new_group_params)
    # Take one more step to confirm the projection actually trains.
    before = proj_weight.detach().clone()
    ctx.step = 1
    update_rule.step(student, batch, ctx)
    assert not torch.allclose(before, proj_weight)
