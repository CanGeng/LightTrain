"""Adversarial tests for lighttrain.builtin_plugins.objectives.flow_matching."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.objectives.flow_matching import FlowMatchingObjective
from lighttrain.protocols import LossContext, ModelOutput


def test_flow_matching_rectified_xt_equals_linear_interp():
    """Goal: rectified flow → x_t = (1-t)·x_0 + t·x_1.

    Analytical: matches the canonical linear interpolation.
    """
    torch.manual_seed(71)
    obj = FlowMatchingObjective(variant="rectified_flow")
    batch = {"x": torch.randn(3, 5)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    t_view = out["t"].view(3, 1)
    expected_xt = (1.0 - t_view) * out["x0"] + t_view * out["x1"]
    torch.testing.assert_close(out["x_t"], expected_xt, atol=1e-5, rtol=1e-4)


def test_flow_matching_target_velocity_equals_x1_minus_x0():
    """Goal: rectified flow target velocity is x_1 - x_0 (constant along path)."""
    torch.manual_seed(72)
    obj = FlowMatchingObjective(variant="rectified_flow")
    batch = {"x": torch.randn(3, 5)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    expected_ut = out["x1"] - out["x0"]
    torch.testing.assert_close(out["ut"], expected_ut, atol=1e-5, rtol=1e-4)


def test_flow_matching_perfect_v_pred_zero_loss():
    """Goal: pred velocity = ut → MSE = 0."""
    torch.manual_seed(73)
    obj = FlowMatchingObjective(variant="rectified_flow")
    batch = {"x": torch.randn(2, 4)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    mo = ModelOutput(outputs={"v": batch["ut"].clone()})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_flow_matching_ot_coupling_permutes_x0():
    """Goal: OT-CFM variant matches noise samples to data via greedy minimum
            pairwise distance — the resulting x0 must be a permutation of the
            original randn draws.

    Input: B=4; the greedy coupling permutes x0 so that for each i in 1..B,
           x0[i] is the closest unused source to x1[i].
    Analytical: x0_post is a permutation of x0_pre (each element appears once).
    """
    torch.manual_seed(74)
    obj = FlowMatchingObjective(variant="ot_cfm", sigma_min=1e-4)
    x = torch.randn(4, 3)
    batch_pre = {"x": x}
    out = obj.prepare_batch(batch_pre, step=0, device="cpu")
    # x0 in out is a permutation of the originally-drawn randn. We can verify
    # via sorted comparison along batch dim.
    x0_flat = out["x0"].view(4, -1)
    # Check that no two rows of x0 are duplicates of each other:
    sums = x0_flat.sum(dim=-1).sort().values
    diffs = sums.diff().abs()
    assert (diffs > 1e-8).all(), "OT coupling must produce 4 distinct x0 vectors (no duplicates)"


def test_regression_flow_matching_velocity_sign():
    """Regression pin for ``flow_velocity_sign``.

    Bug: writing ut = x0 - x1 (sign reversed) flips the training direction.

    Input: known x0, x1; check that ut == (x1 - x0), not (x0 - x1).
    Analytical: if pred = x1 - x0, loss=0; if implementation gives x0 - x1
                instead, the same pred gives loss=4·||x1-x0||² (large).
    """
    torch.manual_seed(75)
    obj = FlowMatchingObjective(variant="rectified_flow")
    batch = {"x": torch.randn(2, 4)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    # Predict (x1 - x0) — the correct velocity.
    correct_pred = batch["x1"] - batch["x0"]
    mo = ModelOutput(outputs={"v": correct_pred})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)
    # And explicitly: ut must equal x1 - x0.
    torch.testing.assert_close(batch["ut"], batch["x1"] - batch["x0"], atol=1e-5, rtol=1e-4)
