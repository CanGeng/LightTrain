"""Adversarial tests for lighttrain.builtin_plugins.losses.core.

Each test verifies a closed-form mathematical property (not just shape/finite),
or pins a known/possible bug via a regression test.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.core import (
    CompositeLoss,
    CrossEntropyLoss,
    MaskedLMLoss,
    ZLoss,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# CrossEntropyLoss
# ---------------------------------------------------------------------------


def test_ce_matches_torch_F_cross_entropy_after_shift(dummy_ctx):
    """Goal: verify causal-LM shift (logits[:,:-1] vs labels[:,1:]).

    Input: random logits (B=2, T=4, V=5) and labels (B=2, T=4).
    Analytical: should equal F.cross_entropy of the shifted views (T-1 positions).
    Bug it catches: if shift is reversed (logits[:,1:] vs labels[:,:-1]) or removed,
                    numbers diverge.
    """
    B, T, V = 2, 4, 5
    torch.manual_seed(1)
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V),
        labels[:, 1:].reshape(-1).long(),
        ignore_index=-100,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_ce_ignore_index_all_masked_loss_is_zero_or_nan_handled(dummy_ctx):
    """Goal: all labels = -100 → no valid targets.

    PyTorch's F.cross_entropy returns NaN when all targets are ignore_index.
    Input: labels = -100 everywhere (after shift).
    Analytical: F.cross_entropy contract — we pin behavior so refactors don't
                silently change it. The loss is NaN (well-defined undefined),
                NOT zero (which would mask the bug of forgetting masking).
    Bug it catches: if someone "fixes" NaN by quietly substituting 0, masked
                    batches would silently emit zero loss.
    """
    B, T, V = 2, 3, 4
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    mo = ModelOutput(outputs={"logits": logits})
    out = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)
    assert torch.isnan(out["loss"]), "PyTorch CE on all-ignored should be NaN; pinning behavior."


def test_ce_partial_mask_equals_unmasked_subset(dummy_ctx):
    """Goal: half labels masked → loss equals CE on the unmasked subset only.

    Input: 4 labels post-shift; mask 2 of them → CE on remaining 2.
    Analytical: CE is averaged over non-ignored tokens by F.cross_entropy default.
    """
    torch.manual_seed(2)
    V = 5
    # B=1, T=3 post-shift gives 2 positions. We need T=3 input (shift drops 1).
    logits = torch.randn(1, 3, V)
    # labels[:,1:] becomes (B, 2). Mask first, keep second.
    labels = torch.tensor([[0, -100, 2]], dtype=torch.long)
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    # After shift: shifted_logits = logits[:,:-1,:] = (1, 2, V), shifted_labels = labels[:,1:] = [[-100, 2]]
    # Only position 1 (label=2) contributes. CE on that single position:
    expected = F.cross_entropy(
        logits[:, 1:2, :].reshape(-1, V),
        torch.tensor([2], dtype=torch.long),
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_ce_uniform_logits_equals_log_V(dummy_ctx):
    """Goal: logits=0 → softmax uniform → CE = log(V) for any label.

    Input: zero logits (B=2, T=3, V=7); arbitrary labels.
    Analytical: -log(1/V) = log V = log 7.
    Bug it catches: any rescaling of softmax base (e.g. natural log → log2)
                    breaks this identity.
    """
    B, T, V = 2, 3, 7
    logits = torch.zeros(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    expected = torch.tensor(math.log(V))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_ce_one_hot_extreme_logits_zero_loss(dummy_ctx):
    """Goal: confident correct predictions → loss → 0.

    Input: V=4, labels (after shift) are deterministic; logits put +1e3 on the
           correct class, -1e3 elsewhere.
    Analytical: softmax → ≈ one-hot, CE → 0.
    """
    B, T, V = 1, 3, 4  # shift → 2 positions
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    # logits at positions 0,1 are predictions for labels at positions 1,2 (post shift).
    # We set logits[0,0,:] target = labels[0,1] = 1; logits[0,1,:] target = labels[0,2] = 2.
    logits = torch.full((B, T, V), -1e3)
    logits[0, 0, 1] = 1e3
    logits[0, 1, 2] = 1e3
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    torch.testing.assert_close(actual, torch.tensor(0.0), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_ce_label_smoothing_increases_loss(dummy_ctx, smoothing):
    """Goal: label_smoothing > 0 increases loss vs unsmoothed for confident preds.

    Input: confident correct prediction (one-hot logits); compare smoothing=0
           to smoothing=s.
    Analytical: with one-hot correct logits, unsmoothed CE → 0; smoothed CE
                → smoothing * log(V) (per the cross-entropy soft-target formula).
    """
    B, T, V = 1, 3, 4
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = torch.full((B, T, V), -1e3)
    logits[0, 0, 1] = 1e3
    logits[0, 1, 2] = 1e3
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss(label_smoothing=smoothing)(mo, {"labels": labels}, dummy_ctx)["loss"]
    if smoothing == 0.0:
        torch.testing.assert_close(actual, torch.tensor(0.0), atol=1e-5, rtol=1e-4)
    else:
        # With extreme one-hot correct logits, contribution from non-target ≈ 0
        # because log_softmax for non-target ≈ -2000 (negligible after smoothing weight).
        # CE_smoothed = (1-s)*0 + s/V * sum(-log_softmax). The (V-1) very-negative
        # log_softmax terms blow up; the formula collapses. Skip exact match
        # for non-zero smoothing in extreme regime — assert > 0 with explicit bound.
        # Instead use moderate logits.
        logits_moderate = torch.zeros(B, T, V)
        logits_moderate[0, 0, 1] = 5.0
        logits_moderate[0, 1, 2] = 5.0
        mo2 = ModelOutput(outputs={"logits": logits_moderate})
        smoothed = CrossEntropyLoss(label_smoothing=smoothing)(
            mo2, {"labels": labels}, dummy_ctx
        )["loss"]
        unsmoothed = CrossEntropyLoss(label_smoothing=0.0)(
            mo2, {"labels": labels}, dummy_ctx
        )["loss"]
        assert float(smoothed) > float(unsmoothed), (
            "Label smoothing must increase loss when predictions are not perfectly uniform."
        )


def test_ce_gradient_pulls_correct_class_logit_up(dummy_ctx):
    """Goal: backward sends negative grad to the correct-class logit (SGD raises it).

    Input: one position, V=3, label=0; logits start at zero.
    Analytical: ∂CE/∂logit_correct = softmax_correct − 1 < 0  → SGD update raises it.
                ∂CE/∂logit_other   = softmax_other         > 0  → SGD update lowers it.
    """
    V = 3
    logits = torch.zeros(1, 2, V, requires_grad=True)
    labels = torch.tensor([[0, 0]], dtype=torch.long)
    mo = ModelOutput(outputs={"logits": logits})
    out = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)
    out["loss"].backward()
    grad = logits.grad
    assert grad is not None
    # The correct class for post-shift position 0 is labels[:,1] = 0.
    # logits[0,0,:] is the predictor for position 0 post-shift.
    assert grad[0, 0, 0].item() < 0.0, "grad on correct class must be negative."
    assert grad[0, 0, 1].item() > 0.0, "grad on incorrect class must be positive."


def test_regression_ce_shift_off_by_one(dummy_ctx):
    """Regression pin for ``ce_shift_off_by_one``.

    Bug: forgetting the shift means logits[t] is asked to predict labels[t],
         not labels[t+1]. If we craft logits that perfectly match labels
         **without shifting**, a buggy implementation gives loss≈0; the correct
         (shifted) implementation gives loss > 0 since predictions misalign.

    Input: logits constructed so that argmax(logits[:,t,:]) == labels[:,t]
           for every t. Without shift → perfect → loss≈0.
           With shift → predictions misaligned → loss>0.
    """
    B, T, V = 1, 4, 5
    labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits = torch.full((B, T, V), -1e3)
    for t in range(T):
        logits[0, t, labels[0, t]] = 1e3
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    # With correct shift, logits[:,:-1,:] (positions 0..2) tries to predict
    # labels[:,1:] = [2,3,4]. But logits[0,0,:] is one-hot on 1 (not 2),
    # logits[0,1,:] is one-hot on 2 (not 3), etc. → loss is large (≈ extreme).
    assert float(actual) > 100.0, (
        "Correct CE must perform shift; without shift this would be 0."
    )


def test_regression_ce_2d_logits_not_shifted(dummy_ctx):
    """Regression: ``(B, V)`` classification logits must NOT be next-token shifted.

    Bug: the shift guard ``logits.dim() >= 2 and ... and size(-2) == size(-1)``
    fired for ``(B, V)`` logits + ``(B,)`` labels because ``size(-2) == size(-1)``
    coincidentally holds (B == B). That silently dropped the first logit row and
    last label, misaligning a classification head. The guard is now ``dim >= 3``.

    Input: a plain classification head — ``(B, V)`` logits, ``(B,)`` labels.
    Expected: equals ``F.cross_entropy(logits, labels)`` with NO shift.
    """
    B, V = 4, 10
    torch.manual_seed(0)
    logits = torch.randn(B, V)
    labels = torch.tensor([1, 2, 3, 4])
    mo = ModelOutput(outputs={"logits": logits})
    actual = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    expected = F.cross_entropy(logits, labels)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# MaskedLMLoss
# ---------------------------------------------------------------------------


def test_mlm_no_shift_distinct_from_ce(dummy_ctx):
    """Goal: MLM has no shift — same logits/labels yield different loss than CE.

    Input: random logits and labels of equal time-length.
    Analytical: MLM uses positions 0..T-1; CE uses 0..T-2 vs labels 1..T-1.
                Numerical values will differ unless inputs are uniform.
    """
    torch.manual_seed(3)
    B, T, V = 2, 4, 5
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    mlm_loss = MaskedLMLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    ce_loss = CrossEntropyLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    assert abs(float(mlm_loss) - float(ce_loss)) > 1e-3, (
        "MLM (no shift) must differ from CE (with shift) for random inputs."
    )


def test_mlm_uniform_equals_log_V(dummy_ctx):
    """Goal: zero logits → MLM loss = log(V).

    Input: zero logits and arbitrary labels.
    Analytical: same as CE uniform — softmax uniform → -log(1/V) = log V.
    """
    B, T, V = 2, 3, 8
    logits = torch.zeros(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mo = ModelOutput(outputs={"logits": logits})
    actual = MaskedLMLoss()(mo, {"labels": labels}, dummy_ctx)["loss"]
    expected = torch.tensor(math.log(V))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# ZLoss
# ---------------------------------------------------------------------------


def test_zloss_logsumexp_squared_closed_form(dummy_ctx):
    """Goal: ZLoss = weight * mean((logsumexp(logits))²) per-token.

    Input: logits = [[1.0, 2.0, 3.0]] (B=1, T=1, V=3).
    Analytical: lse = log(e+e²+e³) = log(30.1928...) ≈ 3.40760596...
                loss = weight * lse² ≈ 1e-4 * 11.61178... ≈ 1.161178e-3.
    """
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    mo = ModelOutput(outputs={"logits": logits})
    lse = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    expected = torch.tensor(1e-4 * lse * lse)
    actual = ZLoss(weight=1e-4)(mo, {}, dummy_ctx)["loss"]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_zloss_zero_logits_equals_weight_log_V_squared(dummy_ctx):
    """Goal: zero logits → log Z = log V → loss = w · (log V)².

    Input: zero logits with V=4.
    Analytical: lse = log(4·e⁰) = log 4. loss = w * (log 4)².
    """
    B, T, V = 2, 3, 4
    logits = torch.zeros(B, T, V)
    mo = ModelOutput(outputs={"logits": logits})
    w = 0.5
    actual = ZLoss(weight=w)(mo, {}, dummy_ctx)["loss"]
    expected = torch.tensor(w * (math.log(V) ** 2))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_zloss_weight_linear_scaling(dummy_ctx):
    """Goal: doubling ``weight`` exactly doubles the loss.

    Input: identical logits, two weights w and 2w.
    Analytical: w·E[lse²] vs 2w·E[lse²].
    """
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 5)
    mo = ModelOutput(outputs={"logits": logits})
    l1 = ZLoss(weight=1e-3)(mo, {}, dummy_ctx)["loss"]
    l2 = ZLoss(weight=2e-3)(mo, {}, dummy_ctx)["loss"]
    torch.testing.assert_close(l2, 2 * l1, atol=1e-6, rtol=1e-5)


def test_regression_zloss_missing_square(dummy_ctx):
    """Regression pin for ``zloss_missing_square``.

    Bug: dropping the squaring (using |lse| or lse instead of lse²) changes
    the magnitude in a detectable way for non-trivial logits.

    Input: zero logits → lse = log V; correct loss = w·log²V; buggy linear
           loss would be w·log V. For V=4, w=1.0: correct ≈ 1.922, buggy ≈ 1.386.
    """
    V = 4
    logits = torch.zeros(1, 1, V)
    mo = ModelOutput(outputs={"logits": logits})
    actual = ZLoss(weight=1.0)(mo, {}, dummy_ctx)["loss"]
    correct_squared = math.log(V) ** 2
    incorrect_linear = math.log(V)
    torch.testing.assert_close(actual, torch.tensor(correct_squared), atol=1e-5, rtol=1e-4)
    # Defensive: ensure the squared form is materially different from linear
    # so a "forgot the square" regression couldn't pass both.
    assert abs(correct_squared - incorrect_linear) > 0.1


# ---------------------------------------------------------------------------
# CompositeLoss
# ---------------------------------------------------------------------------


def test_composite_weighted_sum_matches_components(dummy_ctx, clean_registry):
    """Goal: composite total == Σ w_i · child_loss_i exactly.

    Input: two ZLoss children with different weights, applied to known logits.
    Analytical: total = w_a · zloss(weight=1.0) + w_b · zloss(weight=1.0)
                Each child runs with its own weight, then composite applies w_a / w_b.
    """
    V = 4
    logits = torch.zeros(1, 1, V)
    mo = ModelOutput(outputs={"logits": logits})
    composite = CompositeLoss(
        children=[
            {"name": "z_loss", "weight": 0.3, "params": {"weight": 1.0}},
            {"name": "z_loss", "weight": 0.7, "params": {"weight": 1.0}},
        ]
    )
    out = composite(mo, {}, dummy_ctx)
    each = math.log(V) ** 2  # weight=1.0 inside child
    expected_total = 0.3 * each + 0.7 * each
    torch.testing.assert_close(out["loss"], torch.tensor(expected_total), atol=1e-5, rtol=1e-4)


def test_composite_zero_weight_excludes_child_numerically(dummy_ctx, clean_registry):
    """Goal: a child with weight=0 contributes 0 to the total but still records.

    Input: composite of two ZLoss children, weight 1.0 and 0.0.
    Analytical: total equals only the first child's contribution.
    """
    V = 4
    logits = torch.zeros(1, 1, V)
    mo = ModelOutput(outputs={"logits": logits})
    composite = CompositeLoss(
        children=[
            {"name": "z_loss", "weight": 1.0, "params": {"weight": 1.0}},
            {"name": "z_loss", "weight": 0.0, "params": {"weight": 1.0}},
        ]
    )
    out = composite(mo, {}, dummy_ctx)
    expected = math.log(V) ** 2
    torch.testing.assert_close(out["loss"], torch.tensor(expected), atol=1e-5, rtol=1e-4)
