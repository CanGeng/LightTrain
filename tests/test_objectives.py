"""Tests for M7 Objective implementations."""
import pytest
import torch

from lighttrain.builtin_plugins.objectives.diffusion import DiffusionObjective
from lighttrain.builtin_plugins.objectives.flow_matching import FlowMatchingObjective
from lighttrain.builtin_plugins.objectives.jepa import JEPAObjective
from lighttrain.builtin_plugins.objectives.masked_denoising import MaskedDenoisingObjective
from lighttrain.builtin_plugins.objectives.next_token import NextTokenObjective
from lighttrain.protocols import LossContext, ModelOutput


# ---------------------------------------------------------------------------
# NextTokenObjective
# ---------------------------------------------------------------------------

def test_next_token_loss_family():
    obj = NextTokenObjective()
    assert obj.loss_family == "next_token"


def test_next_token_prepare_batch_noop():
    obj = NextTokenObjective()
    batch = {"input_ids": torch.randint(0, 100, (2, 8)), "labels": torch.randint(0, 100, (2, 8))}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert out is batch  # no-op


# ---------------------------------------------------------------------------
# DiffusionObjective
# ---------------------------------------------------------------------------

def test_diffusion_prepare_batch_keys():
    obj = DiffusionObjective(timesteps=10)
    batch = {"x": torch.randn(4, 32)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert "noisy_x" in out and "noise" in out and "t" in out


def test_diffusion_noisy_x_shape():
    obj = DiffusionObjective(timesteps=10)
    x = torch.randn(3, 16)
    batch = {"x": x}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert out["noisy_x"].shape == x.shape


def test_diffusion_eps_loss_finite():
    obj = DiffusionObjective(target="eps", timesteps=10)
    batch = {"x": torch.randn(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    pred = torch.randn_like(batch["noise"])
    mo = ModelOutput(outputs={"pred": pred})
    ctx = LossContext()
    ld = obj(mo, batch, ctx)
    assert torch.isfinite(ld["loss"])
    assert ctx.loss_family == "diffusion"


def test_diffusion_x0_target():
    obj = DiffusionObjective(target="x0", timesteps=10)
    batch = {"x": torch.randn(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    pred = torch.randn_like(batch["x"])
    mo = ModelOutput(outputs={"pred": pred})
    ld = obj(mo, batch, LossContext())
    assert torch.isfinite(ld["loss"])


def test_diffusion_v_target():
    obj = DiffusionObjective(target="v", timesteps=10)
    batch = {"x": torch.randn(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    pred = torch.randn_like(batch["x"])
    mo = ModelOutput(outputs={"pred": pred})
    ld = obj(mo, batch, LossContext())
    assert torch.isfinite(ld["loss"])


def test_diffusion_cosine_schedule():
    obj = DiffusionObjective(noise_schedule="cosine", timesteps=10)
    batch = {"x": torch.randn(2, 8)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert "noisy_x" in out


# ---------------------------------------------------------------------------
# FlowMatchingObjective
# ---------------------------------------------------------------------------

def test_flow_matching_prepare_batch():
    obj = FlowMatchingObjective()
    batch = {"x": torch.randn(4, 16)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert "x_t" in out and "ut" in out and "t" in out


def test_flow_matching_loss_finite():
    obj = FlowMatchingObjective()
    batch = {"x": torch.randn(3, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    v_pred = torch.randn_like(batch["ut"])
    mo = ModelOutput(outputs={"v": v_pred})
    ld = obj(mo, batch, LossContext())
    assert torch.isfinite(ld["loss"])
    assert ld["flow_mse"].shape == ()


def test_ot_cfm_variant():
    obj = FlowMatchingObjective(variant="ot_cfm")
    batch = {"x": torch.randn(4, 8)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert "x0" in out and "x1" in out


# ---------------------------------------------------------------------------
# JEPAObjective
# ---------------------------------------------------------------------------

def test_jepa_prepare_batch():
    obj = JEPAObjective(num_context_patches=6, num_target_patches=4)
    batch = {"patches": torch.randn(2, 16, 32)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert out["context_patches"].shape == (2, 6, 32)
    assert out["target_patches"].shape == (2, 4, 32)


def test_jepa_loss_finite():
    obj = JEPAObjective(num_context_patches=6, num_target_patches=4)
    batch = {"patches": torch.randn(2, 16, 32)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    pred = torch.randn(2, 4, 32)
    mo = ModelOutput(outputs={"pred_embeddings": pred}, extras={"target_embeddings": batch["target_patches"]})
    ld = obj(mo, batch, LossContext())
    assert torch.isfinite(ld["loss"])
    assert LossContext().loss_family is None  # default


def test_jepa_loss_family_set():
    obj = JEPAObjective()
    batch = {"patches": torch.randn(2, 16, 32)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    ctx = LossContext()
    mo = ModelOutput(outputs={"pred_embeddings": batch["target_patches"]},
                     extras={"target_embeddings": batch["target_patches"]})
    obj(mo, batch, ctx)
    assert ctx.loss_family == "jepa"


# ---------------------------------------------------------------------------
# MaskedDenoisingObjective
# ---------------------------------------------------------------------------

def test_masked_denoising_prepare_batch():
    obj = MaskedDenoisingObjective(mask_prob=0.5, mask_token_id=4, vocab_size=20)
    ids = torch.randint(0, 20, (2, 16))
    batch = {"input_ids": ids.clone(), "attention_mask": torch.ones(2, 16)}
    out = obj.prepare_batch(batch, step=0, device="cpu")
    assert "mlm_labels" in out
    # Labels should be -100 for non-masked positions
    assert (out["mlm_labels"] == -100).any()


def test_masked_denoising_loss_finite():
    obj = MaskedDenoisingObjective(mask_prob=0.5, vocab_size=50)
    ids = torch.randint(0, 50, (2, 8))
    batch = {"input_ids": ids, "attention_mask": torch.ones(2, 8)}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    logits = torch.randn(2, 8, 50)
    mo = ModelOutput(outputs={"logits": logits})
    ld = obj(mo, batch, LossContext())
    assert torch.isfinite(ld["loss"])
    assert ld["mlm_loss"].shape == ()
