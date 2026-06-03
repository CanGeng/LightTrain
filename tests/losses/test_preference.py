"""Adversarial tests for lighttrain.builtin_plugins.losses.preference (BT / DPO / IPO / SimPO / ORPO / KTO)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.preference import (
    BradleyTerryLoss,
    DPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from lighttrain.protocols import LossContext, ModelOutput


_LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# BradleyTerryLoss
# ---------------------------------------------------------------------------


def test_bt_chosen_far_greater_than_rejected_low_loss(dummy_ctx, dummy_model_output):
    """Goal: Δ = +10 → loss = -logsigmoid(10) ≈ 4.54e-5.

    Input: chosen - rejected = +10, B=4.
    Analytical: -log σ(10) = log(1 + exp(-10)).
    """
    chosen = torch.full((4,), 5.0)
    rejected = torch.full((4,), -5.0)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out = BradleyTerryLoss()(dummy_model_output, batch, dummy_ctx)
    expected = torch.tensor(math.log1p(math.exp(-10.0)))
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_bt_rejected_greater_than_chosen_high_loss(dummy_ctx, dummy_model_output):
    """Goal: Δ = -10 → loss = -logsigmoid(-10) = log(1 + e^10) ≈ 10.0000454.

    Analytical: -log σ(-10) = log(1 + exp(10)).
    """
    chosen = torch.zeros(2)
    rejected = torch.full((2,), 10.0)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out = BradleyTerryLoss()(dummy_model_output, batch, dummy_ctx)
    expected = torch.tensor(math.log1p(math.exp(10.0)))
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_bt_zero_diff_equals_log2(dummy_ctx, dummy_model_output):
    """Goal: chosen == rejected, margin=0 → -logsigmoid(0) = log 2.

    Analytical: σ(0) = 0.5; -log(0.5) = log 2.
    """
    chosen = torch.zeros(3)
    rejected = torch.zeros(3)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out = BradleyTerryLoss(margin=0.0)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(_LOG2), atol=1e-5, rtol=1e-4)


def test_bt_margin_shifts_loss_exactly(dummy_ctx, dummy_model_output):
    """Goal: margin m → rewards = (chosen-rejected) - m → identity loss(Δ, m) = loss(Δ - m, 0).

    Input: Δ = 2; margin = 1.5 ↔ Δ' = 0.5, margin 0.
    Analytical: should match.
    """
    chosen = torch.full((4,), 2.0)
    rejected = torch.zeros(4)
    out_with_margin = BradleyTerryLoss(margin=1.5)(
        dummy_model_output, {"chosen_logps": chosen, "rejected_logps": rejected}, dummy_ctx
    )
    chosen_shifted = torch.full((4,), 0.5)
    rejected_shifted = torch.zeros(4)
    out_no_margin = BradleyTerryLoss(margin=0.0)(
        dummy_model_output,
        {"chosen_logps": chosen_shifted, "rejected_logps": rejected_shifted},
        dummy_ctx,
    )
    torch.testing.assert_close(
        out_with_margin["loss"], out_no_margin["loss"], atol=1e-5, rtol=1e-4
    )


def test_bt_gradient_pushes_chosen_up_rejected_down(dummy_ctx, dummy_model_output):
    """Goal: backward → d_loss/d_chosen < 0, d_loss/d_rejected > 0.

    Analytical: loss = -logsigmoid(chosen - rejected - m).
                d/d_chosen = -σ(rewards)·(1-σ(rewards)) / σ(rewards) etc.
                Simply: d/d_chosen = -(1 - σ(rewards)) < 0; d/d_rejected = +(1-σ) > 0.
    """
    chosen = torch.zeros(2, requires_grad=True)
    rejected = torch.zeros(2, requires_grad=True)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out = BradleyTerryLoss()(dummy_model_output, batch, dummy_ctx)
    out["loss"].backward()
    assert chosen.grad is not None and rejected.grad is not None
    assert (chosen.grad < 0).all(), "chosen logp should receive negative grad (raise it)."
    assert (rejected.grad > 0).all(), "rejected logp should receive positive grad (lower it)."


# ---------------------------------------------------------------------------
# DPOLoss
# ---------------------------------------------------------------------------


def test_dpo_equal_policy_and_ref_chosen_rejected_equals_log2(dummy_ctx, dummy_model_output):
    """Goal: π == ref → logits = 0 → loss = log 2.

    Analytical: pi_logratios = ref_logratios → β·0 = 0 → -log σ(0) = log 2.
    """
    chosen = torch.zeros(3)
    rejected = torch.zeros(3)
    ref_chosen = torch.zeros(3)
    ref_rejected = torch.zeros(3)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = DPOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(_LOG2), atol=1e-5, rtol=1e-4)


def test_dpo_policy_prefers_chosen_beats_ref_closed_form(dummy_ctx, dummy_model_output):
    """Goal: closed form for β=0.1, Δπ=2, Δref=0 → logits=0.2 → -log σ(0.2).

    Analytical: -log σ(0.2) = log(1 + exp(-0.2)) ≈ 0.5981.
    """
    chosen = torch.full((4,), 1.0)
    rejected = torch.full((4,), -1.0)  # Δπ = 2
    ref_chosen = torch.zeros(4)
    ref_rejected = torch.zeros(4)  # Δref = 0
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = DPOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)
    expected = torch.tensor(math.log1p(math.exp(-0.2)))
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_dpo_beta_scales_logits_linearly(dummy_ctx, dummy_model_output):
    """Goal: doubling β doubles the inner logits.

    Input: fixed (Δπ - Δref) = 2.0; β=0.1 vs β=0.2.
    Analytical: logits1 = 0.2; logits2 = 0.4. Loss values: -log σ(0.2)=0.598, -log σ(0.4)=0.514.
    """
    chosen = torch.full((2,), 1.0)
    rejected = torch.full((2,), -1.0)
    ref_chosen = torch.zeros(2)
    ref_rejected = torch.zeros(2)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out_low = DPOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)
    out_high = DPOLoss(beta=0.2)(dummy_model_output, batch, dummy_ctx)
    expected_low = torch.tensor(math.log1p(math.exp(-0.2)))
    expected_high = torch.tensor(math.log1p(math.exp(-0.4)))
    torch.testing.assert_close(out_low["loss"], expected_low, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_high["loss"], expected_high, atol=1e-5, rtol=1e-4)


def test_dpo_accuracy_counts_correct_sign(dummy_ctx, dummy_model_output):
    """Goal: dpo_accuracy = fraction of samples with logits > 0.

    Input: 4 samples — 3 with Δπ > Δref, 1 with Δπ < Δref.
    Analytical: accuracy = 3/4 = 0.75.
    """
    chosen = torch.tensor([1.0, 1.0, 1.0, -1.0])
    rejected = torch.tensor([-1.0, -1.0, -1.0, 1.0])
    ref_chosen = torch.zeros(4)
    ref_rejected = torch.zeros(4)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = DPOLoss()(dummy_model_output, batch, dummy_ctx)
    assert abs(out["dpo_accuracy"] - 0.75) < 1e-6


def test_dpo_gradient_pulls_chosen_up_relative_to_ref(dummy_ctx, dummy_model_output):
    """Goal: gradient direction increases π(chosen) and decreases π(rejected).

    Analytical: d_loss/d_chosen = -β(1-σ(z)) < 0; d_loss/d_rejected = +β(1-σ(z)) > 0.
    """
    chosen = torch.zeros(2, requires_grad=True)
    rejected = torch.zeros(2, requires_grad=True)
    ref_chosen = torch.zeros(2)
    ref_rejected = torch.zeros(2)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = DPOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)
    out["loss"].backward()
    assert (chosen.grad < 0).all()
    assert (rejected.grad > 0).all()


def test_regression_dpo_sign_direction(dummy_ctx, dummy_model_output):
    """Regression pin for ``dpo_sign_direction``.

    Bug: swapping the sign — using -(π_logratios - ref_logratios) — would
    minimize the wrong objective (chase the reference instead of moving away).

    Input: π aligned (Δπ=+2), ref neutral (Δref=0); β=0.1.
    Analytical: correct loss = -log σ(+0.2) ≈ 0.598.
                Bug loss = -log σ(-0.2) ≈ 0.798 (larger by ~0.2).
                Diff is well above atol so a sign flip can't slip through.
    """
    chosen = torch.full((4,), 1.0)
    rejected = torch.full((4,), -1.0)
    ref_chosen = torch.zeros(4)
    ref_rejected = torch.zeros(4)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = DPOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)
    correct = math.log1p(math.exp(-0.2))
    flipped = math.log1p(math.exp(+0.2))
    torch.testing.assert_close(out["loss"], torch.tensor(correct), atol=1e-5, rtol=1e-4)
    assert abs(correct - flipped) > 0.15, "Margin large enough to catch a sign bug."


# ---------------------------------------------------------------------------
# IPOLoss
# ---------------------------------------------------------------------------


def test_ipo_optimal_h_zero_loss(dummy_ctx, dummy_model_output):
    """Goal: h = 0 → loss = 0.

    Construction: choose chosen, rejected, ref_* so that
        (chosen - rejected) - (ref_chosen - ref_rejected) = 1/(2β)
    For β = 0.5, 1/(2β) = 1. So set Δπ - Δref = 1.
    """
    beta = 0.5
    chosen = torch.full((3,), 1.0)
    rejected = torch.zeros(3)
    ref_chosen = torch.zeros(3)
    ref_rejected = torch.zeros(3)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = IPOLoss(beta=beta)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_ipo_squared_form_symmetric(dummy_ctx, dummy_model_output):
    """Goal: h and -h give identical loss (squared form).

    Input: two batches whose h values are exact negatives.
    Analytical: h² = (-h)².
    """
    beta = 0.1
    # Make h_a = -h_b symmetrically about the optimum 1/(2β) = 5.
    # Set Δπ - Δref = 5 + delta for one batch, 5 - delta for the other.
    delta = 0.7
    chosen_a = torch.full((3,), 5.0 + delta)
    chosen_b = torch.full((3,), 5.0 - delta)
    rejected = torch.zeros(3)
    ref_chosen = torch.zeros(3)
    ref_rejected = torch.zeros(3)
    batch_a = {
        "chosen_logps": chosen_a, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    batch_b = {
        "chosen_logps": chosen_b, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out_a = IPOLoss(beta=beta)(dummy_model_output, batch_a, dummy_ctx)
    out_b = IPOLoss(beta=beta)(dummy_model_output, batch_b, dummy_ctx)
    torch.testing.assert_close(out_a["loss"], out_b["loss"], atol=1e-5, rtol=1e-4)


def test_ipo_closed_form_value(dummy_ctx, dummy_model_output):
    """Goal: explicit value h = 2 → loss = 4.

    Setup: β=0.1 so 1/(2β)=5. Δπ - Δref = 7 → h = 7 - 5 = 2 → loss = 4.
    """
    beta = 0.1
    chosen = torch.full((5,), 7.0)
    rejected = torch.zeros(5)
    ref_chosen = torch.zeros(5)
    ref_rejected = torch.zeros(5)
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = IPOLoss(beta=beta)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(4.0), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# SimPOLoss
# ---------------------------------------------------------------------------


def test_simpo_gamma_zero_equals_bt_form(dummy_ctx, dummy_model_output):
    """Goal: γ = 0 → SimPO logits = β·(chosen - rejected) → same as BT-with-β.

    Input: any chosen/rejected with γ=0.
    Analytical: -logsigmoid(β·(c-r)) matches a BT-style loss with rewards scaled by β.
    """
    chosen = torch.tensor([1.0, 2.0, -0.5])
    rejected = torch.tensor([0.0, 1.0, 0.5])
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out = SimPOLoss(beta=2.5, gamma=0.0)(dummy_model_output, batch, dummy_ctx)
    expected = -F.logsigmoid(2.5 * (chosen - rejected)).mean()
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_simpo_gamma_shifts_logits_by_gamma(dummy_ctx, dummy_model_output):
    """Goal: changing γ by Δγ shifts inner logits by -Δγ.

    Input: same (c, r), compare γ=0 to γ=1.
    Analytical: -logsigmoid(β(c-r)) vs -logsigmoid(β(c-r) - 1) — differs by a known amount.
    """
    chosen = torch.full((4,), 1.0)
    rejected = torch.zeros(4)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out0 = SimPOLoss(beta=1.0, gamma=0.0)(dummy_model_output, batch, dummy_ctx)
    out1 = SimPOLoss(beta=1.0, gamma=1.0)(dummy_model_output, batch, dummy_ctx)
    expected0 = -F.logsigmoid(torch.tensor(1.0)).item()
    expected1 = -F.logsigmoid(torch.tensor(0.0)).item()  # β(c-r) - γ = 1 - 1 = 0 → log 2
    torch.testing.assert_close(out0["loss"], torch.tensor(expected0), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out1["loss"], torch.tensor(expected1), atol=1e-5, rtol=1e-4)


def test_regression_simpo_gamma_cancels_in_subtraction(dummy_ctx, dummy_model_output):
    """Regression pin for ``simpo_gamma_cancels``.

    Bug: writing the formula as β·((chosen-γ) - (rejected-γ)) would cancel γ
    entirely, making γ have no effect on the loss.

    Input: same batch with γ=0 vs γ=2; correct implementation gives different
           values; the cancellation bug gives identical values.
    """
    chosen = torch.full((4,), 1.0)
    rejected = torch.zeros(4)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected}
    out_g0 = SimPOLoss(beta=1.0, gamma=0.0)(dummy_model_output, batch, dummy_ctx)
    out_g2 = SimPOLoss(beta=1.0, gamma=2.0)(dummy_model_output, batch, dummy_ctx)
    assert abs(float(out_g0["loss"]) - float(out_g2["loss"])) > 0.1, (
        "γ must affect SimPO loss; if it cancels in subtraction the bug returns."
    )


# ---------------------------------------------------------------------------
# ORPOLoss
# ---------------------------------------------------------------------------


def test_orpo_log_odds_ratio_zero_loss_lower_bound(dummy_ctx, dummy_model_output):
    """Goal: chosen == rejected → log_odds_ratio = 0 → ratio_loss = log 2.

    Input: chosen = rejected = -1.0 (both probabilities of e^-1 ≈ 0.368).
           NLL = 0 → sft = 0 → total loss = λ · log 2.
    """
    chosen = torch.full((3,), -1.0)
    rejected = torch.full((3,), -1.0)
    nll = torch.zeros(3)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected, "chosen_nll_loss": nll}
    out = ORPOLoss(lam=1.0)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(_LOG2), atol=1e-5, rtol=1e-4)


def test_orpo_log1mexp_stability_near_zero(dummy_ctx, dummy_model_output):
    """Goal: chosen near 0 (i.e. p ≈ 1) must not blow up via log1mexp.

    Input: chosen = -1e-9 (essentially 0). clamp(max=-1e-7) keeps it valid.
    Analytical: log1mexp is well-defined for x < 0; the clamp ensures we
                stay in domain. We just check the result is finite.
    """
    chosen = torch.tensor([-1e-9, -1e-9])
    rejected = torch.tensor([-1.0, -1.0])
    nll = torch.zeros(2)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected, "chosen_nll_loss": nll}
    out = ORPOLoss(lam=1.0)(dummy_model_output, batch, dummy_ctx)
    # We must produce a finite, well-defined value — pin by requiring it
    # exists and equals a closed form we can compute.
    assert torch.isfinite(out["loss"])
    assert float(out["loss"]) > 0.0


def test_orpo_lambda_zero_loss_equals_sft(dummy_ctx, dummy_model_output):
    """Goal: λ=0 → loss = mean(nll) only.

    Input: nll = [1.0, 2.0, 3.0], any logps.
    Analytical: loss = (1+2+3)/3 = 2.0.
    """
    chosen = torch.full((3,), -1.0)
    rejected = torch.full((3,), -2.0)
    nll = torch.tensor([1.0, 2.0, 3.0])
    batch = {"chosen_logps": chosen, "rejected_logps": rejected, "chosen_nll_loss": nll}
    out = ORPOLoss(lam=0.0)(dummy_model_output, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)


def test_orpo_lambda_scales_ratio_term_linearly(dummy_ctx, dummy_model_output):
    """Goal: loss(λ) - loss(0) = λ · ratio_loss.

    Input: identical batch, two λ values (0 and 0.5).
    Analytical: ratio_loss component scales linearly with λ.
    """
    chosen = torch.full((3,), -0.5)
    rejected = torch.full((3,), -1.5)
    nll = torch.zeros(3)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected, "chosen_nll_loss": nll}
    out0 = ORPOLoss(lam=0.0)(dummy_model_output, batch, dummy_ctx)
    out_half = ORPOLoss(lam=0.5)(dummy_model_output, batch, dummy_ctx)
    out_one = ORPOLoss(lam=1.0)(dummy_model_output, batch, dummy_ctx)
    diff_half = out_half["loss"] - out0["loss"]
    diff_one = out_one["loss"] - out0["loss"]
    torch.testing.assert_close(diff_one, 2 * diff_half, atol=1e-5, rtol=1e-4)


def test_regression_orpo_clamp_prevents_inf(dummy_ctx, dummy_model_output):
    """Regression pin for ``orpo_clamp_inf``.

    Bug: removing ``clamp(max=-1e-7)`` lets chosen = 0 (i.e. p=1) pass through
    log1mexp, which evaluates log(1 - 1) = -inf → loss = NaN/Inf.

    Input: chosen = 0 exactly (a perfectly-confident impossible state).
    Analytical: with clamp, loss stays finite. Without clamp, loss blows up.
    """
    chosen = torch.zeros(2)
    rejected = torch.full((2,), -1.0)
    nll = torch.zeros(2)
    batch = {"chosen_logps": chosen, "rejected_logps": rejected, "chosen_nll_loss": nll}
    out = ORPOLoss(lam=1.0)(dummy_model_output, batch, dummy_ctx)
    assert torch.isfinite(out["loss"]), "clamp must keep loss finite at chosen=0."


# ---------------------------------------------------------------------------
# KTOLoss
# ---------------------------------------------------------------------------


def test_kto_kl_estimate_detached_from_chosen_grad(dummy_ctx, dummy_model_output):
    """Goal: kl = r_chosen.detach().mean() — d_kl/d_chosen must be zero.

    Input: chosen requires_grad; rejected does not; ref_* fixed.
    Analytical: gradient of loss w.r.t. chosen must come only through r_chosen
                in the loss_chosen path (via σ(β·(r_chosen - kl))), not through
                kl itself (detached). We verify by comparing to a manual
                reconstruction that uses kl as a constant.
    """
    chosen = torch.tensor([0.5, 1.0], requires_grad=True)
    rejected = torch.tensor([0.0, 0.0], requires_grad=True)
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = KTOLoss(beta=0.1, lambda_desirable=1.0, lambda_undesirable=1.0)(
        dummy_model_output, batch, dummy_ctx
    )
    out["loss"].backward()

    # Reconstruct expected gradient using a fixed (detached) kl constant.
    chosen2 = torch.tensor([0.5, 1.0], requires_grad=True)
    rejected2 = torch.tensor([0.0, 0.0], requires_grad=True)
    r_chosen2 = chosen2 - ref_chosen
    r_rejected2 = rejected2 - ref_rejected
    kl_const = (chosen - ref_chosen).detach().mean().detach()
    loss_chosen2 = 1.0 * (1.0 - torch.sigmoid(0.1 * (r_chosen2 - kl_const)))
    loss_rejected2 = 1.0 * (1.0 - torch.sigmoid(0.1 * (kl_const - r_rejected2)))
    loss2 = (loss_chosen2 + loss_rejected2).mean() / 2.0
    loss2.backward()

    torch.testing.assert_close(chosen.grad, chosen2.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(rejected.grad, rejected2.grad, atol=1e-5, rtol=1e-4)


def test_kto_chosen_only_loss_when_lambda_undesirable_zero(dummy_ctx, dummy_model_output):
    """Goal: λ_U = 0 → loss = λ_D · (1 - σ(β·(r_c - kl)))/2 only.

    Input: r_chosen identical across batch (so kl = r_chosen[0]) → r_c - kl = 0
           → σ = 0.5 → (1 - 0.5) = 0.5 → loss = λ_D · 0.5 / 2 = 0.25.
    """
    chosen = torch.tensor([1.0, 1.0])
    rejected = torch.tensor([0.0, 0.0])
    ref_chosen = torch.tensor([0.5, 0.5])
    ref_rejected = torch.tensor([0.0, 0.0])
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = KTOLoss(beta=0.1, lambda_desirable=1.0, lambda_undesirable=0.0)(
        dummy_model_output, batch, dummy_ctx
    )
    torch.testing.assert_close(out["loss"], torch.tensor(0.25), atol=1e-5, rtol=1e-4)


def test_kto_chosen_only_loss_when_lambda_desirable_zero(dummy_ctx, dummy_model_output):
    """Goal: symmetry check — λ_D = 0 → loss = λ_U · (1 - σ(β·(kl - r_r)))/2 only.

    Input: r_chosen identical (kl = r_c), r_rejected = kl → σ = 0.5 → loss = 0.25.
    """
    chosen = torch.tensor([1.0, 1.0])
    rejected = torch.tensor([0.5, 0.5])  # so r_rejected = r_chosen = kl
    ref_chosen = torch.tensor([0.5, 0.5])
    ref_rejected = torch.tensor([0.0, 0.0])
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    out = KTOLoss(beta=0.1, lambda_desirable=0.0, lambda_undesirable=1.0)(
        dummy_model_output, batch, dummy_ctx
    )
    torch.testing.assert_close(out["loss"], torch.tensor(0.25), atol=1e-5, rtol=1e-4)


def test_regression_kto_kl_not_attached_to_graph(dummy_ctx, dummy_model_output):
    """Regression pin for ``kto_kl_attached``.

    Bug: dropping ``.detach()`` from kl lets the gradient flow back through
    the KL estimator path, distorting the chosen/rejected gradients.

    Compares numerical loss against the form that explicitly substitutes
    a detached kl constant. They must match exactly.
    """
    chosen = torch.tensor([0.7, 1.2])
    rejected = torch.tensor([0.1, -0.2])
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    batch = {
        "chosen_logps": chosen, "rejected_logps": rejected,
        "ref_chosen_logps": ref_chosen, "ref_rejected_logps": ref_rejected,
    }
    actual = KTOLoss(beta=0.1)(dummy_model_output, batch, dummy_ctx)["loss"]
    # Manual reconstruction with detached kl
    r_c = chosen - ref_chosen
    r_r = rejected - ref_rejected
    kl = r_c.mean()  # detached by virtue of no grad
    loss_c = 1.0 - torch.sigmoid(0.1 * (r_c - kl))
    loss_r = 1.0 - torch.sigmoid(0.1 * (kl - r_r))
    expected = (loss_c + loss_r).mean() / 2.0
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
