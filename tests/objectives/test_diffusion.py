"""Adversarial tests for lighttrain.builtin_plugins.objectives.diffusion."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.objectives.diffusion import DiffusionObjective
from lighttrain.protocols import LossContext, ModelOutput


def _seed_obj_batch(target="eps", schedule="linear", timesteps=10, seed=61):
    """Builds a prepared batch ready for forward."""
    torch.manual_seed(seed)
    obj = DiffusionObjective(target=target, noise_schedule=schedule, timesteps=timesteps)
    batch = {"x": torch.randn(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    return obj, batch


def test_diffusion_eps_perfect_pred_zero_loss():
    """Goal: with target='eps' and pred == noise → loss = 0.

    Analytical: MSE(noise, noise) = 0.
    """
    obj, batch = _seed_obj_batch(target="eps")
    mo = ModelOutput(outputs={"pred": batch["noise"].clone()})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_diffusion_x0_perfect_pred_zero_loss():
    """Goal: with target='x0' and pred == clean x → loss = 0."""
    obj, batch = _seed_obj_batch(target="x0")
    mo = ModelOutput(outputs={"pred": batch["x"].clone()})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_diffusion_v_target_closed_form():
    """Goal: with target='v' and perfect pred = α·noise - σ·x → loss = 0.

    Analytical: implementation computes v = sqrt_acp·noise - sqrt_omacp·x0.
    """
    torch.manual_seed(62)
    obj = DiffusionObjective(target="v", timesteps=10)
    batch = {"x": torch.randn(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    obj._ensure_schedule("cpu")  # ensure schedule arrays exist
    B = batch["x"].shape[0]
    t = batch["t"]
    sqrt_acp = obj._sqrt_acp[t].view(B, 1)
    sqrt_omacp = obj._sqrt_one_minus_acp[t].view(B, 1)
    v_target = sqrt_acp * batch["noise"] - sqrt_omacp * batch["x"]
    mo = ModelOutput(outputs={"pred": v_target.clone()})
    out = obj(mo, batch, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_diffusion_linear_vs_cosine_schedule_alphas_differ_at_midstep():
    """Goal: linear and cosine schedules give different α at t = T/2.

    Analytical: numerical evaluation of the two schedules — values differ.
    """
    obj_lin = DiffusionObjective(noise_schedule="linear", timesteps=10)
    obj_cos = DiffusionObjective(noise_schedule="cosine", timesteps=10)
    obj_lin._ensure_schedule("cpu")
    obj_cos._ensure_schedule("cpu")
    diff = (obj_lin._sqrt_acp[5] - obj_cos._sqrt_acp[5]).abs().item()
    assert diff > 1e-3, f"linear and cosine schedules should differ at t=5; got {diff}"


def test_diffusion_noisy_x_equals_alpha_x0_plus_sigma_noise():
    """Goal: forward q(x_t|x_0) = α·x_0 + σ·noise (the canonical DDPM formula).

    Analytical: reconstruct manually from the schedule arrays.
    """
    torch.manual_seed(63)
    obj = DiffusionObjective(target="eps", timesteps=10)
    batch = {"x": torch.randn(3, 6)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    obj._ensure_schedule("cpu")
    t = batch["t"]
    sqrt_acp = obj._sqrt_acp[t].view(3, 1)
    sqrt_omacp = obj._sqrt_one_minus_acp[t].view(3, 1)
    expected = sqrt_acp * batch["x"] + sqrt_omacp * batch["noise"]
    torch.testing.assert_close(batch["noisy_x"], expected, atol=1e-5, rtol=1e-4)


def test_regression_diffusion_target_dispatch():
    """Regression pin for ``diffusion_target_dispatch``.

    Bug: a refactor wiring the wrong tensor into ``gt`` (e.g. always using
    ``noise`` regardless of self.target) would silently corrupt x0 and v
    training.

    Input: identical batch, three objectives differing only in target.
           Predict the eps tensor for all three.
    Analytical:
        - eps target → MSE(noise, noise) = 0
        - x0 target  → MSE(noise, x) > 0 (noise != x)
        - v target   → MSE(noise, v) > 0 generally
    A dispatch bug that always uses ``noise`` would give 0 for all three.
    """
    torch.manual_seed(64)
    base_batch = {"x": torch.randn(2, 8)}
    obj_eps = DiffusionObjective(target="eps", timesteps=10)
    obj_x0 = DiffusionObjective(target="x0", timesteps=10)
    obj_v = DiffusionObjective(target="v", timesteps=10)

    batch_eps = obj_eps.prepare_batch({**base_batch}, step=0, device="cpu")
    # Reuse the same noise/t so the three batches share x_t.
    pred = batch_eps["noise"].clone()

    out_eps = obj_eps(ModelOutput(outputs={"pred": pred}), batch_eps, LossContext())
    out_x0 = obj_x0(
        ModelOutput(outputs={"pred": pred}),
        {**batch_eps},  # reuse t/noise/x
        LossContext(),
    )
    out_v = obj_v(
        ModelOutput(outputs={"pred": pred}),
        {**batch_eps},
        LossContext(),
    )
    torch.testing.assert_close(out_eps["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)
    assert float(out_x0["loss"]) > 0.1, "x0 target should yield nonzero loss when pred = noise"
    assert float(out_v["loss"]) > 1e-3, "v target should yield nonzero loss when pred = noise"
