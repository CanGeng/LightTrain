"""Adversarial tests for lighttrain.objectives.next_token."""

from __future__ import annotations

import math

import pytest
import torch

from lighttrain.losses.core import CrossEntropyLoss
from lighttrain.objectives.next_token import NextTokenObjective
from lighttrain.protocols import LossContext, ModelOutput


def test_next_token_matches_cross_entropy_loss_value():
    """Goal: NextTokenObjective output equals direct CrossEntropyLoss output
            on the same inputs.

    Analytical: NextTokenObjective is a thin wrapper around CrossEntropyLoss.
                Values must agree to floating-point precision.
    """
    torch.manual_seed(51)
    B, T, V = 2, 4, 5
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    ctx_a = LossContext()
    ctx_b = LossContext()
    obj_loss = NextTokenObjective()(mo, {"labels": labels}, ctx_a)["loss"]
    ce_loss = CrossEntropyLoss()(mo, {"labels": labels}, ctx_b)["loss"]
    torch.testing.assert_close(obj_loss, ce_loss, atol=1e-6, rtol=1e-5)


def test_next_token_uniform_logits_log_V():
    """Goal: zero logits → loss = log V (analytical CE on uniform softmax)."""
    B, T, V = 2, 3, 7
    logits = torch.zeros(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    out = NextTokenObjective()(mo, {"labels": labels}, LossContext())
    torch.testing.assert_close(out["loss"], torch.tensor(math.log(V)), atol=1e-5, rtol=1e-4)


def test_next_token_prepare_batch_returns_input_dict_unchanged():
    """Goal: prepare_batch is a no-op — returns the same dict values.

    Analytical: input identity check on all tensor entries.
    """
    obj = NextTokenObjective()
    batch_in = {
        "input_ids": torch.arange(6).view(2, 3),
        "labels": torch.zeros(2, 3, dtype=torch.long),
    }
    batch_out = obj.prepare_batch(batch_in, step=0, device="cpu")
    # Tensors must be value-identical (we don't require same Python identity).
    assert set(batch_in.keys()) == set(batch_out.keys())
    torch.testing.assert_close(batch_in["input_ids"], batch_out["input_ids"], atol=0, rtol=0)
    torch.testing.assert_close(batch_in["labels"], batch_out["labels"], atol=0, rtol=0)


def test_next_token_sets_loss_family_on_context_and_returns_loss():
    """Goal: loss_family stamping AND loss value present.

    Combined check so the test isn't pure shape — verifies both side effect
    (ctx mutation) and result.
    """
    torch.manual_seed(52)
    B, T, V = 1, 2, 3
    logits = torch.randn(B, T, V)
    labels = torch.zeros(B, T, dtype=torch.long)
    mo = ModelOutput(outputs={"logits": logits})
    ctx = LossContext()
    out = NextTokenObjective()(mo, {"labels": labels}, ctx)
    assert ctx.loss_family == "next_token"
    # Loss must equal direct CE.
    ce = CrossEntropyLoss()(mo, {"labels": labels}, LossContext())["loss"]
    torch.testing.assert_close(out["loss"], ce, atol=1e-6, rtol=1e-5)
