"""Adversarial tests for lighttrain.builtin_plugins.losses.aux (InfoNCE / MoEBalance)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.aux import InfoNCELoss, MoEBalanceLoss
from lighttrain.protocols import LossContext, ModelOutput


# ---------------------------------------------------------------------------
# InfoNCELoss
# ---------------------------------------------------------------------------


def test_infonce_perfectly_aligned_pairs_low_loss(dummy_ctx, dummy_model_output):
    """Goal: anchor==positive AND batch B=1 → only diag, loss → 0.

    Input: identical anchor and positive vectors, B=1, D=4.
    Analytical: logits is a (1,1) matrix; CE on a single-class softmax = 0.
    """
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    batch = {"embeddings_anchor": z, "embeddings_positive": z}
    out = InfoNCELoss(temperature=1.0, normalize=True)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_infonce_random_orthonormal_lower_bound(dummy_ctx, dummy_model_output):
    """Goal: orthonormal pairs (perfect alignment, zero off-diag) → loss → 0.

    Input: anchor = positive = identity rows (B=4, D=4); τ=1, normalize=True.
    Analytical: with normalized identity rows the off-diagonal cosine is 0,
                logits diag=1, off-diag=0; softmax(diag) = e/(e+3·1) per row.
                CE = -log(e/(e+3)) = log(1 + 3/e) ≈ log(2.1036) ≈ 0.7437.
    """
    z = torch.eye(4)
    batch = {"embeddings_anchor": z, "embeddings_positive": z}
    out = InfoNCELoss(temperature=1.0, normalize=True)(dummy_model_output, batch, dummy_ctx)
    expected = math.log(1.0 + 3.0 / math.e)
    torch.testing.assert_close(out["loss"], torch.tensor(expected), atol=1e-5, rtol=1e-4)


def test_infonce_symmetric_loss_value_closed_form(dummy_ctx, dummy_model_output):
    """Goal: verify (CE + CE^T)/2 averaging, not just CE on one direction.

    Input: B=2, D=2 unit vectors that produce an asymmetric similarity matrix.
              z1 = I (identity); z2 = [[1, 0], [0.6, 0.8]] (both unit-norm).
              logits = z1 @ z2.T = [[1, 0.6], [0, 0.8]] — non-symmetric.
    Analytical: hand-compute CE(logits, [0,1]) and CE(logits.T, [0,1]) and
                average; result must match the loss exactly.
    """
    z1 = torch.eye(2)  # already unit-norm
    z2 = torch.tensor([[1.0, 0.0], [0.6, 0.8]])  # both rows unit-norm
    batch = {"embeddings_anchor": z1, "embeddings_positive": z2}
    out = InfoNCELoss(temperature=1.0, normalize=True)(dummy_model_output, batch, dummy_ctx)
    logits = z1 @ z2.T
    labels = torch.arange(2)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_infonce_temperature_monotonicity(dummy_ctx, dummy_model_output):
    """Goal: smaller τ → sharper softmax → lower loss when diag is dominant.

    Input: identity z1/z2 (B=3); compare τ=0.5 vs τ=2.0.
    Analytical: with diagonal cosines = 1 and off-diag = 0, smaller τ
                amplifies the gap, lowering CE.  Loss(0.5) < Loss(2.0).
    """
    z = torch.eye(3)
    batch = {"embeddings_anchor": z, "embeddings_positive": z}
    out_low_tau = InfoNCELoss(temperature=0.5, normalize=True)(dummy_model_output, batch, dummy_ctx)
    out_high_tau = InfoNCELoss(temperature=2.0, normalize=True)(dummy_model_output, batch, dummy_ctx)
    assert float(out_low_tau["loss"]) < float(out_high_tau["loss"]), (
        "Lower τ should sharpen softmax and reduce InfoNCE on diagonal-dominant inputs."
    )


def test_infonce_normalize_makes_scale_invariant(dummy_ctx, dummy_model_output):
    """Goal: with normalize=True, scaling the inputs leaves the loss unchanged.

    Input: same vectors, with one batch scaled by 100x.
    Analytical: L2 normalization maps both to unit-norm vectors → identical
                similarity matrices → identical loss.
    """
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch_1x = {"embeddings_anchor": z, "embeddings_positive": z}
    batch_100x = {"embeddings_anchor": 100 * z, "embeddings_positive": 100 * z}
    loss_a = InfoNCELoss(normalize=True)(dummy_model_output, batch_1x, dummy_ctx)["loss"]
    loss_b = InfoNCELoss(normalize=True)(dummy_model_output, batch_100x, dummy_ctx)["loss"]
    torch.testing.assert_close(loss_a, loss_b, atol=1e-5, rtol=1e-4)


def test_regression_infonce_symmetric_not_one_sided(dummy_ctx, dummy_model_output):
    """Regression pin for ``infonce_one_sided``.

    Bug: dropping the second CE (anchor→positive only) yields a different
    value on asymmetric inputs.

    Input: asymmetric similarity matrix logits = [[1, 0.6], [0, 0.8]] from
           the same setup as the closed-form test.
    Analytical: CE(logits, [0,1]) ≈ 0.4420, CE(logits.T, [0,1]) ≈ 0.4557.
                Symmetric mean ≈ 0.4488. One-sided alone misses this by ~0.007.
    """
    z1 = torch.eye(2)
    z2 = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    batch = {"embeddings_anchor": z1, "embeddings_positive": z2}
    actual = InfoNCELoss(temperature=1.0, normalize=True)(
        dummy_model_output, batch, dummy_ctx
    )["loss"]
    logits = z1 @ z2.T
    labels = torch.arange(2)
    one_sided = F.cross_entropy(logits, labels)
    symmetric = (one_sided + F.cross_entropy(logits.T, labels)) / 2.0
    torch.testing.assert_close(actual, symmetric, atol=1e-5, rtol=1e-4)
    # The two forms must differ meaningfully so a one-sided regression is caught.
    assert abs(float(symmetric) - float(one_sided)) > 1e-3, (
        "Test inputs must be asymmetric enough to distinguish symmetric vs one-sided."
    )


# ---------------------------------------------------------------------------
# MoEBalanceLoss
# ---------------------------------------------------------------------------


def test_moe_balance_uniform_routing_value(dummy_model_output):
    """Goal: uniform router_probs and uniform expert_mask → known minimum.

    Input: router_probs = 1/E everywhere (B=2, T=3, E=4) and expert_mask uniform.
    Analytical: fraction_e = 1/E, prob_e = 1/E → sum = E · (1/E)·(1/E) = 1/E.
                loss = weight · E · 1/E = weight · 1 = weight.
                (Note: actually loss = weight * E * sum(fraction*prob) = w * E * 1/E = w)
                Wait — sum is over E experts so sum = E * (1/E^2) = 1/E. Then
                weight * E * (1/E) = weight.
    """
    B, T, E = 2, 3, 4
    router_probs = torch.full((B, T, E), 1.0 / E)
    expert_mask = torch.full((B, T, E), 1.0 / E)
    ctx = LossContext(extras={"router_probs": router_probs, "expert_mask": expert_mask})
    w = 0.05
    out = MoEBalanceLoss(weight=w)(dummy_model_output, {}, ctx)
    expected = torch.tensor(w * 1.0)  # E * (1/E) = 1
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_moe_balance_collapsed_routing_upper_bound(dummy_model_output):
    """Goal: all tokens routed to expert 0 → maximum imbalance.

    Input: router_probs all on expert 0, expert_mask all on expert 0.
    Analytical: fraction[0] = 1, others = 0; prob[0] = 1, others = 0.
                sum(fraction * prob) = 1·1 + 0 + ... = 1.
                loss = weight · E · 1 = weight · E.
    """
    B, T, E = 2, 3, 4
    router_probs = torch.zeros(B, T, E)
    router_probs[..., 0] = 1.0
    expert_mask = torch.zeros(B, T, E)
    expert_mask[..., 0] = 1.0
    ctx = LossContext(extras={"router_probs": router_probs, "expert_mask": expert_mask})
    w = 0.05
    out = MoEBalanceLoss(weight=w)(dummy_model_output, {}, ctx)
    expected = torch.tensor(w * E * 1.0)
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_moe_balance_falls_back_to_router_probs_when_mask_missing(dummy_model_output):
    """Goal: missing expert_mask uses router_probs for fraction.

    Input: skewed router_probs but NO expert_mask in ctx.
    Analytical: fraction = router_probs.mean over (B,T) → same as prob_mean.
                loss = weight · E · sum(prob² over experts).
                With router_probs = [[0.7, 0.3]] for all (B,T,E=2):
                prob_mean = [0.7, 0.3]; sum(prob²) = 0.49 + 0.09 = 0.58.
                loss = weight · 2 · 0.58 = 0.116 · weight / 0.1 ...
                Concrete: weight=1.0 → loss = 2 · 0.58 = 1.16.
    """
    B, T, E = 3, 4, 2
    router_probs = torch.zeros(B, T, E)
    router_probs[..., 0] = 0.7
    router_probs[..., 1] = 0.3
    ctx = LossContext(extras={"router_probs": router_probs})
    out = MoEBalanceLoss(weight=1.0)(dummy_model_output, {}, ctx)
    expected = torch.tensor(2.0 * (0.7**2 + 0.3**2))
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_moe_balance_weight_linear_scaling(dummy_model_output):
    """Goal: doubling ``weight`` doubles the loss."""
    B, T, E = 2, 2, 3
    router_probs = torch.softmax(torch.randn(B, T, E), dim=-1)
    ctx_a = LossContext(extras={"router_probs": router_probs})
    ctx_b = LossContext(extras={"router_probs": router_probs})
    a = MoEBalanceLoss(weight=1e-2)(dummy_model_output, {}, ctx_a)["loss"]
    b = MoEBalanceLoss(weight=2e-2)(dummy_model_output, {}, ctx_b)["loss"]
    torch.testing.assert_close(b, 2 * a, atol=1e-6, rtol=1e-5)


def test_moe_balance_mask_vs_no_mask_distinguishable(dummy_model_output):
    """Goal: providing a hard expert_mask yields a different loss than soft routing.

    Input: skewed router_probs with a hard mask that picks the dominant expert.
    Analytical: with mask, fraction is hard-1-hot; without, it's soft. The two
                numerical results differ ⇒ implementation must read expert_mask
                when present (not silently use router_probs in both paths).
    """
    B, T, E = 3, 4, 3
    router_probs = torch.tensor([0.6, 0.3, 0.1]).expand(B, T, E).contiguous()
    expert_mask = torch.zeros(B, T, E)
    expert_mask[..., 0] = 1.0  # all tokens hard-routed to expert 0
    ctx_with_mask = LossContext(
        extras={"router_probs": router_probs, "expert_mask": expert_mask}
    )
    ctx_without_mask = LossContext(extras={"router_probs": router_probs})
    with_mask = MoEBalanceLoss(weight=1.0)(dummy_model_output, {}, ctx_with_mask)["loss"]
    without_mask = MoEBalanceLoss(weight=1.0)(dummy_model_output, {}, ctx_without_mask)["loss"]
    # With mask: fraction = [1, 0, 0], prob = [0.6, 0.3, 0.1].
    # sum = 1*0.6 = 0.6; loss = 1.0 · 3 · 0.6 = 1.8.
    torch.testing.assert_close(with_mask, torch.tensor(1.8), atol=1e-5, rtol=1e-4)
    # Without mask: prob = [0.6, 0.3, 0.1]; sum(p²) = 0.36 + 0.09 + 0.01 = 0.46.
    # loss = 1.0 · 3 · 0.46 = 1.38.
    torch.testing.assert_close(without_mask, torch.tensor(1.38), atol=1e-5, rtol=1e-4)
    # And the two must be materially different.
    assert abs(float(with_mask) - float(without_mask)) > 1e-3
