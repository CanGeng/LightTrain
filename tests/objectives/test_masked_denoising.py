"""Adversarial tests for lighttrain.plugins.objectives.masked_denoising."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lighttrain.plugins.objectives.masked_denoising import MaskedDenoisingObjective
from lighttrain.protocols import LossContext, ModelOutput


def test_mlm_expected_mask_ratio_in_large_sample():
    """Goal: with attention_mask all-ones, fraction of masked positions ≈ mask_prob.

    Input: B=64, T=64, mask_prob=0.15.
    Analytical: with N=4096 positions, expected #masked = 614.4, std ≈ 22.85.
                Use a wide tolerance band (±5σ) to be reproducible across seeds.
    """
    torch.manual_seed(91)
    obj = MaskedDenoisingObjective(mask_prob=0.15, mask_token_id=103, vocab_size=100)
    input_ids = torch.randint(0, 100, (64, 64))
    attention_mask = torch.ones(64, 64)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    masked_count = (batch["mlm_labels"] != -100).sum().item()
    expected = 0.15 * 64 * 64
    std = math.sqrt(0.15 * 0.85 * 64 * 64)
    assert abs(masked_count - expected) < 5 * std, (
        f"masked count {masked_count} too far from expected {expected:.1f} (5σ={5*std:.1f})"
    )


def test_mlm_labels_ignore_outside_masked_positions():
    """Goal: at positions where no masking happened, label = -100 (ignored).

    Construction: build deterministic mask via fixed seed and verify exactly
    which positions have non-(-100) labels.
    """
    torch.manual_seed(92)
    obj = MaskedDenoisingObjective(mask_prob=0.3, mask_token_id=99, vocab_size=50)
    input_ids = torch.randint(0, 50, (2, 8))
    batch = {"input_ids": input_ids.clone()}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    # Verify: at every position where labels is NOT -100, the value equals the
    # original input_ids[position] (the label encodes the *original* token).
    non_ignored = batch["mlm_labels"] != -100
    if non_ignored.any():
        orig_at_masked = input_ids[non_ignored]
        lbl_at_masked = batch["mlm_labels"][non_ignored]
        torch.testing.assert_close(lbl_at_masked, orig_at_masked, atol=0, rtol=0)
    # And every -100 position retains its original input id (no replacement happened).
    # (Strong claim! With 30% replace_with_mask of *non-masked subset* — wait,
    # the BERT logic only replaces at masked positions, so unmasked positions are
    # unchanged.)
    ignored = ~non_ignored
    if ignored.any():
        torch.testing.assert_close(
            batch["input_ids"][ignored], input_ids[ignored], atol=0, rtol=0
        )


def test_mlm_loss_matches_cross_entropy_on_mlm_labels():
    """Goal: forward loss equals F.cross_entropy(logits, mlm_labels, ignore_index=-100).

    Input: known logits, prepared batch.
    Analytical: hand-reconstruct CE and compare.
    """
    torch.manual_seed(93)
    B, T, V = 2, 6, 30
    obj = MaskedDenoisingObjective(mask_prob=0.3, mask_token_id=29, vocab_size=V)
    batch = {"input_ids": torch.randint(0, V, (B, T))}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    logits = torch.randn(B, T, V)
    mo = ModelOutput(outputs={"logits": logits})
    out = obj(mo, batch, LossContext())
    expected = F.cross_entropy(
        logits.view(B * T, V),
        batch["mlm_labels"].view(B * T),
        ignore_index=-100,
    )
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_mlm_attention_mask_prevents_padding_from_being_masked():
    """Goal: positions with attention_mask=0 are never selected as MLM targets.

    Input: B=8, T=16. Mark last 8 columns as padding (attention_mask=0).
    Analytical: mlm_labels in padding region must be -100 for every batch row.
    """
    torch.manual_seed(94)
    obj = MaskedDenoisingObjective(mask_prob=0.5, mask_token_id=99, vocab_size=50)
    input_ids = torch.randint(0, 50, (8, 16))
    attention_mask = torch.cat([torch.ones(8, 8), torch.zeros(8, 8)], dim=1)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    padding_labels = batch["mlm_labels"][:, 8:]
    assert (padding_labels == -100).all(), (
        "padding positions must never be selected as MLM targets."
    )


def test_mlm_sets_loss_family_on_context():
    """Goal: ctx.loss_family is stamped to 'masked_denoising'.

    Combined with a numerical loss check so this isn't pure metadata.
    """
    torch.manual_seed(95)
    obj = MaskedDenoisingObjective(mask_prob=0.5, vocab_size=20)
    batch = {"input_ids": torch.randint(0, 20, (2, 4))}
    batch = obj.prepare_batch(batch, step=0, device="cpu")
    logits = torch.zeros(2, 4, 20)  # uniform → CE = log(V) on masked positions
    mo = ModelOutput(outputs={"logits": logits})
    ctx = LossContext()
    out = obj(mo, batch, ctx)
    assert ctx.loss_family == "masked_denoising"
    # Verify the loss equals log(V) since logits are zero and CE on uniform.
    # (Only counts masked positions; if any are masked the result is log(V).)
    if (batch["mlm_labels"] != -100).any():
        torch.testing.assert_close(
            out["loss"], torch.tensor(math.log(20)), atol=1e-5, rtol=1e-4
        )
