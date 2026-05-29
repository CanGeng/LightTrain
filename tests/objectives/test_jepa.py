"""Adversarial tests for lighttrain.objectives.jepa."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.objectives.jepa import JEPAObjective
from lighttrain.protocols import LossContext, ModelOutput


def test_jepa_perfect_pred_zero_loss():
    """Goal: pred_embeddings == target → loss = 0."""
    torch.manual_seed(81)
    obj = JEPAObjective(num_context_patches=4, num_target_patches=2)
    batch = {"patches": torch.randn(2, 8, 6)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    target = batch["target_patches"]
    mo = ModelOutput(outputs={"pred_embeddings": target.clone()})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_jepa_known_mse_closed_form():
    """Goal: pred = target + 1 → loss = mean(1²) = 1.0."""
    torch.manual_seed(82)
    obj = JEPAObjective(num_context_patches=4, num_target_patches=2)
    batch = {"patches": torch.randn(2, 8, 6)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    target = batch["target_patches"]
    mo = ModelOutput(outputs={"pred_embeddings": target.clone() + 1.0})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_jepa_target_path_has_no_gradient():
    """Goal: gradient through target_embeddings must be blocked (detach()).

    Input: pred requires_grad=True; target embeddings provided as a
           requires_grad=True tensor that should be detached internally.
    Analytical: after backward, the target tensor's grad must remain None.
    """
    obj = JEPAObjective(num_context_patches=4, num_target_patches=2)
    base_batch = {"patches": torch.randn(2, 8, 6)}
    base_batch = obj.prepare_batch(base_batch, step=0, device="cpu")
    pred = base_batch["target_patches"].detach().clone().requires_grad_(True)
    target_with_grad = base_batch["target_patches"].clone().requires_grad_(True)
    mo = ModelOutput(
        outputs={"pred_embeddings": pred},
        extras={"target_embeddings": target_with_grad},
    )
    out = obj(mo, base_batch, LossContext())
    out["loss"].backward()
    assert pred.grad is not None  # pred path must receive gradient
    assert target_with_grad.grad is None, (
        "target_embeddings path must be detached so no gradient flows back."
    )


def test_jepa_ema_step_applies_known_decay():
    """Goal: θ_t ← m·θ_t + (1-m)·θ_s for each parameter (analytical).

    Input: trivial 1-param student/teacher; m = 0.9.
    Analytical: new = 0.9·old_teacher + 0.1·student.
    """
    student = nn.Linear(2, 2, bias=False)
    teacher = nn.Linear(2, 2, bias=False)
    # Force known values.
    with torch.no_grad():
        student.weight.copy_(torch.ones(2, 2) * 1.0)
        teacher.weight.copy_(torch.zeros(2, 2))
    obj = JEPAObjective(ema_momentum=0.9)
    obj.set_target_encoder(teacher)
    obj.ema_step(student)
    expected = 0.9 * 0.0 + 0.1 * 1.0  # = 0.1 everywhere
    torch.testing.assert_close(
        teacher.weight, torch.full((2, 2), expected), atol=1e-6, rtol=1e-5
    )


def test_regression_jepa_target_must_not_have_grad():
    """Regression pin for ``jepa_target_grad``.

    Bug: dropping ``.detach()`` on target_emb lets the predictor optimize
    toward a target that also moves, creating an unstable training signal
    (representation collapse). Repeats test_jepa_target_path_has_no_gradient
    standalone so the bug name maps 1-1 to a single test.
    """
    obj = JEPAObjective(num_context_patches=4, num_target_patches=2)
    batch = {"patches": torch.randn(2, 8, 6)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    pred = batch["target_patches"].clone().requires_grad_(True)
    target_with_grad = batch["target_patches"].clone().requires_grad_(True)
    mo = ModelOutput(
        outputs={"pred_embeddings": pred},
        extras={"target_embeddings": target_with_grad},
    )
    out = obj(mo, batch, LossContext())
    out["loss"].backward()
    assert target_with_grad.grad is None
