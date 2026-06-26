"""Adversarial tests for lighttrain.builtin_plugins.losses.rl (PPOSurrogateLoss / GRPOLoss).

Attack focus
------------
PPO/GRPO clipping is the highest-risk implementation area because the bug
classes (``min`` vs ``max``, ``min`` vs clipped-only, ratio vs log_ratio,
wrong KL direction, biased vs unbiased std) all silently pass shape/finite
checks. These tests force each branch via constructed inputs whose closed-form
answer differs across the buggy and correct implementations.
"""

from __future__ import annotations

import math

import pytest
import torch

from lighttrain.builtin_plugins.losses.rl import GRPOLoss, PPOSurrogateLoss
from lighttrain.protocols import LossContext, ModelOutput

_DUMMY = ModelOutput(outputs={})


# ---------------------------------------------------------------------------
# PPOSurrogateLoss
# ---------------------------------------------------------------------------


def _ppo_ctx_simple(B=2, T=4, lp_new=0.0, lp_old=0.0, adv=1.0):
    return LossContext(extras={
        "log_probs_new": torch.full((B, T), float(lp_new)),
        "log_probs_old": torch.full((B, T), float(lp_old)),
        "advantages": torch.full((B, T), float(adv)),
    })


def test_ppo_ratio_one_surrogate_equals_neg_advantages():
    """Goal: ratio=1 → min(ratio·A, clip·A) = A → policy_loss = -mean(A).

    Input: log_new = log_old = 0 → ratio = 1; A = 2.5 everywhere.
    Analytical: policy_loss = -2.5. With vf_coef=ent_coef=0 → total = -2.5.
    """
    ctx = _ppo_ctx_simple(adv=2.5)
    loss = PPOSurrogateLoss(vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(-2.5), atol=1e-5, rtol=1e-4)


def test_ppo_positive_advantage_ratio_above_clip_uses_clipped():
    """Goal: A > 0, ratio > 1+ε → min picks clipped (= (1+ε)·A).

    Input: log_new = log(1.5), log_old = 0 → ratio = 1.5; A = 1; clip_eps=0.2.
    Analytical: surr1 = 1.5; surr2 = 1.2; min = 1.2; policy_loss = -1.2.
    """
    B, T = 2, 3
    lp_new = torch.full((B, T), math.log(1.5))
    lp_old = torch.zeros(B, T)
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    loss = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(-1.2), atol=1e-5, rtol=1e-4)


def test_ppo_negative_advantage_ratio_above_clip_uses_unclipped():
    """Goal: A < 0, ratio > 1+ε → min picks UNCLIPPED (more punitive).

    Input: log_new = log(2.0), log_old = 0 → ratio = 2.0; A = -1; clip_eps=0.2.
    Analytical: surr1 = 2.0 · -1 = -2.0; surr2 = 1.2 · -1 = -1.2.
                min(-2.0, -1.2) = -2.0. policy_loss = -mean(-2.0) = +2.0.
    Bug it catches: replacing torch.min with surr2 (clipped-only) gives 1.2.
    """
    B, T = 2, 3
    lp_new = torch.full((B, T), math.log(2.0))
    lp_old = torch.zeros(B, T)
    adv = torch.full((B, T), -1.0)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    loss = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)


def test_ppo_negative_advantage_ratio_below_clip_uses_unclipped():
    """Goal: A < 0, ratio < 1-ε → min picks UNCLIPPED (still more negative).

    Input: log_new = log(0.5), log_old = 0 → ratio = 0.5; A = -1; clip_eps=0.2.
    Analytical: surr1 = 0.5 · -1 = -0.5; surr2 = 0.8 · -1 = -0.8.
                min(-0.5, -0.8) = -0.8 → policy_loss = +0.8.
    """
    B, T = 2, 3
    lp_new = torch.full((B, T), math.log(0.5))
    lp_old = torch.zeros(B, T)
    adv = torch.full((B, T), -1.0)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    loss = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(0.8), atol=1e-5, rtol=1e-4)


def test_ppo_clip_frac_counts_outliers_exactly():
    """Goal: clip_frac is the exact fraction of |ratio-1| > clip_eps.

    Input: B=4, T=1 — half samples with ratio=1 (inside), half with ratio=2 (outside).
    Analytical: clip_frac = 0.5.
    """
    lp_old = torch.zeros(4, 1)
    # Two samples with ratio=1, two with ratio=2.
    lp_new = torch.tensor([[0.0], [0.0], [math.log(2.0)], [math.log(2.0)]])
    adv = torch.ones(4, 1)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(clip_eps=0.2)(_DUMMY, {}, ctx)
    assert abs(out["clip_frac"] - 0.5) < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_ppo_approx_kl_nonneg_invariant(seed):
    """Goal: approx_kl is reported as mean(log_old - log_new); on random inputs
            with log_old > log_new on average, it's >= 0. We don't assert it
            is always >= 0 in general (approximate KL can dip negative) — instead
            we verify it equals the exact masked mean we expect.
    """
    torch.manual_seed(seed)
    B, T = 3, 5
    lp_new = torch.randn(B, T) * 0.1
    lp_old = torch.randn(B, T) * 0.1
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss()(_DUMMY, {}, ctx)
    expected_kl = (lp_old - lp_new).mean().item()
    assert abs(out["approx_kl"] - expected_kl) < 1e-5


def test_ppo_value_loss_mse_when_no_clip_range():
    """Goal: vf_clip_range=None → value_loss = mean((values - returns)²).

    Input: values = [1, 2], returns = [0, 0] → squared diffs = [1, 4] → mean 2.5.
    """
    B, T = 1, 2
    lp_old = torch.zeros(B, T)
    lp_new = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    values = torch.tensor([[1.0, 2.0]])
    returns = torch.tensor([[0.0, 0.0]])
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
        "values": values, "returns": returns,
    })
    out = PPOSurrogateLoss(clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, vf_clip_range=None)(
        _DUMMY, {}, ctx
    )
    assert abs(out["value_loss"] - 2.5) < 1e-5


def test_ppo_value_loss_clipped_takes_max_branch():
    """Goal: vf_clip_range active → value_loss = mean(max(unclipped, clipped)).

    Input: values_old = 0, values = 10, returns = 0, vf_clip_range = 1.
                vf_unclipped = (10-0)² = 100.
                values_clipped = 0 + clip(10-0, -1, 1) = 1; vf_clipped = (1-0)² = 1.
                max = 100. value_loss = 100.
    """
    B, T = 1, 2
    lp_old = torch.zeros(B, T)
    lp_new = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    values = torch.full((B, T), 10.0)
    returns = torch.zeros(B, T)
    values_old = torch.zeros(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
        "values": values, "returns": returns, "values_old": values_old,
    })
    out = PPOSurrogateLoss(vf_coef=1.0, ent_coef=0.0, vf_clip_range=1.0)(
        _DUMMY, {}, ctx
    )
    assert abs(out["value_loss"] - 100.0) < 1e-4


def test_ppo_entropy_signed_correctly_in_total():
    """Goal: total = policy + vf·value - ent·entropy. Verify the MINUS on entropy.

    Input: policy_loss = 0 (ratio=1, A=0), value_loss = 0, entropy = -lp_new.mean() = 0.5 (lp_new=-0.5).
    Analytical: with ent_coef = 0.2, entropy_loss = 0.5 → total = 0 + 0 - 0.2·0.5 = -0.1.
    """
    B, T = 2, 4
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), -0.5)
    adv = torch.zeros(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(vf_coef=0.0, ent_coef=0.2)(_DUMMY, {}, ctx)
    # policy_loss = 0; entropy_loss = mean(-lp_new) = 0.5. Total = -0.2·0.5 = -0.1.
    torch.testing.assert_close(out["loss"], torch.tensor(-0.1), atol=1e-5, rtol=1e-4)


def test_ppo_entropy_gradient_flows_through_log_probs():
    """Goal: ent_coef > 0 path keeps log_probs_new in the graph so that
            backward propagates through it (no .detach()).

    Analytical: d_loss/d_lp_new should include the -ent_coef · d(-lp_new)/d_lp_new
                contribution = +ent_coef / N.
    """
    B, T = 2, 4
    lp_new = torch.full((B, T), -0.5, requires_grad=True)
    lp_old = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(vf_coef=0.0, ent_coef=0.1)(_DUMMY, {}, ctx)
    out["loss"].backward()
    # With adv=0 the policy term contributes zero gradient. Entropy term:
    # entropy_loss = mean(-lp_new); d/d_lp_new = -1/(B*T) per element; with the
    # -ent_coef sign in total → grad = +ent_coef / (B*T) per element.
    expected_grad = torch.full((B, T), 0.1 / (B * T))
    torch.testing.assert_close(lp_new.grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_ppo_mask_all_zero_returns_zero_policy_loss_not_nan():
    """Goal: labels all -100 → no positions contribute → masked mean uses
            denom.clamp_min(1) → policy_loss = 0 (no NaN).
    """
    B, T = 2, 3
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), -0.5)
    adv = torch.ones(B, T)
    labels = torch.full((B, T), -100, dtype=torch.long)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(vf_coef=0.0, ent_coef=0.0)(_DUMMY, {"labels": labels}, ctx)
    # masked mean: x * 0 / 1 = 0. policy_loss = -0 = 0.
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-7, rtol=1e-6)


def test_ppo_labels_mask_changes_loss_quantitatively():
    """Goal: with labels-mask vs no mask, loss differs by an exactly computable amount.

    Input: prompt half has lp_new=lp_old=0 (surr=A), response half has lp_new=-0.5 (ratio < 1, A>0 → use unclipped surr1 = e^-0.5 · A).
    Analytical: with labels=-100 on prompt, only response counts.
                no-labels: mean over all → mean(A, exp(-0.5)·A) per row.
                labels: mean only over response → exp(-0.5) · A.
    """
    B = 2
    A = 1.0
    lp_prompt = torch.zeros(B, 3)
    lp_resp = torch.full((B, 3), -0.5)
    lp_new = torch.cat([lp_prompt, lp_resp], dim=1)
    lp_old = torch.zeros(B, 6)
    adv = torch.full((B, 6), A)
    labels = torch.cat([torch.full((B, 3), -100), torch.ones(B, 3, dtype=torch.long)], dim=1)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out_labeled = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(
        _DUMMY, {"labels": labels}, ctx
    )
    out_no_labels = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(
        _DUMMY, {}, ctx
    )
    # response-only: ratio=exp(-0.5) ≈ 0.607, surr = ratio·A, policy_loss = -0.607.
    # all positions: half ratio=1 (surr=1), half ratio=0.607 (surr=0.607) → mean ≈ 0.803.
    torch.testing.assert_close(
        out_labeled["loss"], torch.tensor(-math.exp(-0.5)), atol=1e-5, rtol=1e-4
    )
    expected_all = -(0.5 * 1.0 + 0.5 * math.exp(-0.5))
    torch.testing.assert_close(
        out_no_labels["loss"], torch.tensor(expected_all), atol=1e-5, rtol=1e-4
    )


def test_regression_ppo_min_not_clipped_only():
    """Regression pin for ``ppo_min_replaced_by_clipped``.

    Bug: rewriting policy_loss = -mean(surr2) (drop the min) silently corrupts
    PPO for negative advantages with ratio > 1+ε. Same construction as
    test_ppo_negative_advantage_ratio_above_clip_uses_unclipped, kept as a
    standalone regression so the bug name maps 1-1 to a single test.
    """
    B, T = 2, 4
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), math.log(2.0))
    adv = torch.full((B, T), -1.0)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    # Correct: 2.0; buggy (drop min, use surr2 only): 1.2. Diff = 0.8 ≫ atol.
    torch.testing.assert_close(out["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)


def test_ppo_reports_loss_and_policy_loss_keys():
    """Goal: PPO always exposes both ``loss`` and ``policy_loss`` metric keys."""
    ctx = _ppo_ctx_simple(adv=1.0)
    out = PPOSurrogateLoss()(_DUMMY, {}, ctx)
    assert "loss" in out and "policy_loss" in out


def test_ppo_kl_zero_when_new_equals_ref():
    """Goal (A1): log_probs_new == log_probs_ref → k3 KL = exp(0) - 0 - 1 = 0."""
    ctx = _ppo_ctx_simple(B=2, T=3, lp_new=-0.5)
    ctx.extras["log_probs_ref"] = ctx.extras["log_probs_new"].clone()
    out = PPOSurrogateLoss(beta_kl=1.0, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    assert abs(out["kl"]) < 1e-6


@pytest.mark.parametrize("delta", [-1.0, -0.3, 0.3, 1.0])
def test_ppo_kl_nonneg_invariant_random_deltas(delta):
    """Goal (A1): the k3 estimator is non-negative for any ref≠new offset."""
    ctx = _ppo_ctx_simple(B=2, T=3, lp_new=0.0)
    ctx.extras["log_probs_ref"] = ctx.extras["log_probs_new"] + delta
    out = PPOSurrogateLoss(beta_kl=1.0, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    assert out["kl"] >= -1e-6


def test_ppo_kl_fail_loud_when_beta_kl_but_no_ref():
    """Goal (A1): beta_kl>0 with no log_probs_ref in ctx → raise, not silent drop."""
    ctx = _ppo_ctx_simple(adv=1.0)
    with pytest.raises(RuntimeError, match="log_probs_ref"):
        PPOSurrogateLoss(beta_kl=1.0)(_DUMMY, {}, ctx)


def test_ppo_beta_kl_zero_no_kl_term_added():
    """Behavior-neutral baseline: default beta_kl=0 → kl metric 0, ref not required."""
    ctx = _ppo_ctx_simple(adv=1.0)
    ctx.extras["log_probs_ref"] = ctx.extras["log_probs_new"] + 1.0  # present but ignored
    out = PPOSurrogateLoss(beta_kl=0.0)(_DUMMY, {}, ctx)
    assert out["kl"] == 0.0


# ---------------------------------------------------------------------------
# GRPOLoss
# ---------------------------------------------------------------------------


def test_grpo_ratio_one_in_group_equal_advantages_zero_loss():
    """Goal: within a single group, equal advantages → normalize → all zeros.

    Input: B=4, T=1, advantages = [5, 5, 5, 5], group_ids = [0,0,0,0].
    Analytical: mean=5, std=0 → clamped to ε → normalized = 0.
                ratio=1, surr = 0, policy_loss = 0.
    """
    B, T = 4, 1
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([5.0, 5.0, 5.0, 5.0])
    group_ids = torch.tensor([0, 0, 0, 0])
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "group_ids": group_ids,
    })
    out = GRPOLoss()(_DUMMY, {}, ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_grpo_group_norm_two_groups_independent_population_std():
    """Goal: each group normalized using its own population mean/std.

    Input: B=4, T=1. Group 0 advantages = [1, 3], group 1 = [10, 30].
    Analytical: Group 0: mean=2, std_pop=1 → normalized = [-1, +1].
                Group 1: mean=20, std_pop=10 → normalized = [-1, +1].
                ratio=1 → policy_loss = -mean(normalized) = -mean([-1, 1, -1, 1]) = 0.
                The fact that policy_loss == 0 confirms BOTH groups normalize
                independently (across-group normalization would yield a non-zero
                centered mean here).
    """
    B, T = 4, 1
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([1.0, 3.0, 10.0, 30.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "group_ids": group_ids,
    })
    out = GRPOLoss()(_DUMMY, {}, ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_grpo_kl_zero_when_new_equals_ref():
    """Goal: log_probs_new == log_probs_ref → k3 KL = exp(0) - 0 - 1 = 0.

    Input: log_ref == log_new.
    Analytical: kl_per_token = 0 → kl = 0 → kl_loss = β·0 = 0 → loss = policy_loss only.
    """
    B, T = 2, 3
    lp_new = torch.full((B, T), -0.5)
    lp_old = torch.zeros(B, T)
    lp_ref = lp_new.clone()
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "log_probs_ref": lp_ref,
    })
    out = GRPOLoss(beta_kl=1.0)(_DUMMY, {}, ctx)
    assert abs(out["kl"]) < 1e-6


@pytest.mark.parametrize("delta", [-1.0, -0.3, 0.3, 1.0])
def test_grpo_kl_nonneg_invariant_random_deltas(delta):
    """Goal: k3 estimator is always >= 0 (exp(x) - x - 1 >= 0 for any x).

    Input: log_ref - log_new = δ (a constant offset).
    Analytical: kl_per_token = exp(δ) - δ - 1 (always >= 0).
                After β=1: kl_loss = exp(δ) - δ - 1.
    """
    B, T = 2, 3
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    lp_ref = torch.full((B, T), float(delta))
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "log_probs_ref": lp_ref,
    })
    out = GRPOLoss(beta_kl=1.0)(_DUMMY, {}, ctx)
    expected_kl = math.exp(delta) - delta - 1.0
    torch.testing.assert_close(
        torch.tensor(out["kl"]), torch.tensor(expected_kl), atol=1e-5, rtol=1e-4
    )
    assert out["kl"] >= -1e-7  # nonneg invariant


def test_grpo_kl_k3_formula_closed_form_distinguishes_from_gaussian_approx():
    """Goal: kl = mean(exp(Δ) - Δ - 1) with Δ = log_ref - log_new.

    Construction: Δ = 0.5 → kl = e^0.5 - 0.5 - 1 ≈ 0.1487.
                  Gaussian approx 0.5·Δ² = 0.125 — different by ~20%.
    Bug it catches: replacing k3 with the Gaussian approximation.
    """
    B, T = 1, 4
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    lp_ref = torch.full((B, T), 0.5)
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "log_probs_ref": lp_ref,
    })
    out = GRPOLoss(beta_kl=1.0)(_DUMMY, {}, ctx)
    correct = math.exp(0.5) - 0.5 - 1.0  # ≈ 0.1487
    gaussian = 0.5 * 0.5 * 0.5  # ≈ 0.125
    torch.testing.assert_close(
        torch.tensor(out["kl"]), torch.tensor(correct), atol=1e-5, rtol=1e-4
    )
    assert abs(correct - gaussian) > 0.02  # well above tolerance


def test_grpo_clipping_negative_advantage_uses_min():
    """Goal: GRPO clipping must use min(unclipped, clipped) — same bug as PPO.

    Input: log_new = log(2.0), log_old = 0 → ratio = 2.0; A = -1; B=2, T=4
           (uniform group → norm is 0 by default; bypass by making one-element groups).
    Analytical: with single-sample groups, normalize_by_group leaves adv unchanged
                (std=0 → clamped to ε → normalized = (adv - mean)/eps = 0).
                That suppresses the test! → DON'T use group_ids; advantages broadcast
                only. (Note: GRPO skips normalization if group_ids.numel() <= 1.)
                With 2 distinct adv values in same group, norm picks them up.
                Use group_ids=[0,0]; adv per-sample = [-1, -1] → adv (B,T) full -1.
                But same-value group normalizes to 0 too. So we OMIT group_ids,
                so the code branch ``if group_ids is not None and numel > 1`` is
                skipped → adv stays as -1.
    """
    B, T = 2, 4
    lp_new = torch.full((B, T), math.log(2.0))
    lp_old = torch.zeros(B, T)
    adv = torch.full((B, T), -1.0)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = GRPOLoss(clip_eps=0.2)(_DUMMY, {}, ctx)
    # surr1 = 2·-1 = -2; surr2 = 1.2·-1 = -1.2; min = -2 → policy_loss = 2.0
    torch.testing.assert_close(out["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)


def test_grpo_beta_kl_zero_no_kl_term_added():
    """Goal: β_kl = 0 → loss == policy_loss exactly (no KL added).

    Input: arbitrary log_ref provided but β_kl=0.
    Analytical: kl_loss = torch.tensor(0.0) → total = policy_loss.
    """
    B, T = 2, 3
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    lp_ref = torch.full((B, T), 100.0)  # extreme value to amplify any bug
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "log_probs_ref": lp_ref,
    })
    out = GRPOLoss(beta_kl=0.0)(_DUMMY, {}, ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(-1.0), atol=1e-5, rtol=1e-4)


def test_grpo_ratio_mean_equals_one_when_logprobs_match():
    """Goal: log_probs_new == log_probs_old → ratio_mean = 1.0 exactly.

    Analytical: exp(0).mean() = 1.0.
    """
    B, T = 2, 3
    ctx = LossContext(extras={
        "log_probs_new": torch.full((B, T), 0.3),
        "log_probs_old": torch.full((B, T), 0.3),
        "advantages": torch.ones(B),
    })
    out = GRPOLoss()(_DUMMY, {}, ctx)
    assert abs(out["ratio_mean"] - 1.0) < 1e-6


def test_grpo_labels_mask_excludes_prompt_positions():
    """Goal: GRPO honors a labels-based mask — prompt positions (-100) are
            excluded so the masked loss differs from the unmasked loss.

    Input: prompt half uses lp_new == lp_old (ratio=1), response half uses
           lp_new < lp_old (ratio<1); per-group advantages differ across groups.
    Analytical: masking out the prompt half changes which positions contribute,
                so the labeled and unlabeled losses must differ.
    """
    B, T = 4, 6
    lp_new = torch.cat([torch.zeros(B, 3), torch.full((B, 3), -0.1)], dim=1)
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    group_ids = torch.tensor([0, 0, 1, 1])
    labels = torch.cat(
        [torch.full((B, 3), -100), torch.ones(B, 3, dtype=torch.long)], dim=1
    )
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "group_ids": group_ids,
    })
    out_labeled = GRPOLoss()(_DUMMY, {"labels": labels}, ctx)
    out_plain = GRPOLoss()(_DUMMY, {}, ctx)
    assert float(out_labeled["loss"]) != float(out_plain["loss"])


def test_regression_grpo_kl_sign():
    """Regression pin for ``grpo_kl_direction``.

    Bug: writing log_diff = log_new - log_ref (sign flipped) computes a
    different number for asymmetric Δ.

    Input: log_ref - log_new = 1.0.
    Analytical: correct k3 = e^1 - 1 - 1 ≈ 0.7183.
                Flipped: e^-1 - (-1) - 1 = e^-1 ≈ 0.3679.
                Diff > 0.3 — well above atol.
    """
    B, T = 1, 4
    lp_new = torch.zeros(B, T)
    lp_old = torch.zeros(B, T)
    lp_ref = torch.full((B, T), 1.0)
    adv = torch.ones(B, T)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "log_probs_ref": lp_ref,
    })
    out = GRPOLoss(beta_kl=1.0)(_DUMMY, {}, ctx)
    correct = math.exp(1.0) - 1.0 - 1.0
    flipped = math.exp(-1.0) - (-1.0) - 1.0
    torch.testing.assert_close(
        torch.tensor(out["kl"]), torch.tensor(correct), atol=1e-5, rtol=1e-4
    )
    assert abs(correct - flipped) > 0.3, "diff large enough to catch a sign bug."


def test_regression_grpo_population_vs_sample_std():
    """Regression pin for ``grpo_std_unbiased``.

    Bug: using std(unbiased=True) (denominator N-1) instead of unbiased=False
    (denominator N) gives a different normalization on small groups.

    Input: group of 2 advantages [0, 2].
    Analytical: pop std = 1.0 → normalized = [-1, +1].
                sample std = √2 ≈ 1.414 → normalized = [-1/√2, +1/√2] = [-0.707, +0.707].
                With ratio=1, policy_loss = -mean(normalized). Either way mean
                is 0 → policy_loss = 0; not distinguishable.
                Use log_new != log_old to make ratio dominate one element.
                Set lp_new[1] = log(2) (ratio=2 for second sample), lp_new[0]=0.
                Correct: surr1 = ratio·norm = [1·-1, 2·+1] = [-1, 2]; min vs clip:
                  sample 0: surr1=-1, surr2=0.8·-1=-0.8 → min=-1.
                  sample 1: A=+1, surr1=2·+1=2, surr2=1.2·1=1.2 → min=1.2.
                policy_loss = -mean(-1, 1.2) = -0.1.
                Buggy (unbiased=True): norm = [-0.707, +0.707].
                  sample 0: A=-0.707 < 0, ratio=1; surr1=surr2=-0.707 → min=-0.707.
                  sample 1: A=+0.707, ratio=2; surr1=1.414, surr2=1.2·0.707=0.849 → min=0.849.
                policy_loss = -(0.849 + (-0.707))/2 = -0.071.
                Diff ~ 0.03 — comfortably above 1e-5.
    """
    B, T = 2, 1
    lp_new = torch.tensor([[0.0], [math.log(2.0)]])
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([0.0, 2.0])  # group of 2; pop std = 1
    group_ids = torch.tensor([0, 0])
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "group_ids": group_ids,
    })
    out = GRPOLoss(clip_eps=0.2)(_DUMMY, {}, ctx)
    # Correct pop std = 1 → normalized = [-1, +1]
    # surr1 sample 0: 1·(-1) = -1; surr2: 0.8·(-1) = -0.8; min = -1
    # surr1 sample 1: 2·(+1) = 2;  surr2: 1.2·(+1) = 1.2; min = 1.2
    expected = -((-1.0) + 1.2) / 2.0  # = -0.1
    torch.testing.assert_close(out["loss"], torch.tensor(expected), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Malicious PR review — attack-path tests
#
# These are explicit anti-shortcuts that close common "make it pass cheaply"
# rewrites of the GRPO/PPO surrogate. See plan file for full discussion.
# ---------------------------------------------------------------------------


def test_attack_ppo_ratio_must_be_exp_not_linear_log_ratio():
    """Attack: someone replaces ratio = exp(log_new - log_old) with the linear
              approximation `1 + (log_new - log_old)` ("near-1 trick").

    Construction: log_new - log_old = log(2.0) ≈ 0.693 (very far from 0).
                  Correct ratio = 2.0 → unclipped surr = 2.0 · A.
                  Buggy ratio = 1.693 → unclipped surr = 1.693 · A.
                  With A=+1 and ratio above clip (1.2): correct min = 1.2
                  (both clamp identically). With A=-1: correct = -2.0, buggy = -1.693.
                  Use A=-1 setup so the difference is detectable.
    """
    B, T = 2, 4
    lp_old = torch.zeros(B, T)
    lp_new = torch.full((B, T), math.log(2.0))
    adv = torch.full((B, T), -1.0)
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old, "advantages": adv,
    })
    out = PPOSurrogateLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)(_DUMMY, {}, ctx)
    # Correct → 2.0; linear-approx ratio (1.693) → policy_loss = 1.693.
    correct = 2.0
    linear = 1.0 + math.log(2.0)  # = 1.693
    torch.testing.assert_close(out["loss"], torch.tensor(correct), atol=1e-5, rtol=1e-4)
    assert abs(correct - linear) > 0.2, "ratio extreme enough to catch linear approx."


def test_attack_grpo_advantage_norm_must_use_group_mean_not_global_mean():
    """Attack: someone replaces per-group normalization with a global
              normalize_advantages(advantages.flatten()) call.

    Construction: two groups with VERY different scales. Global normalization
                  centers across the whole batch and yields biased per-group means.

    Group 0 = [1, 3] (group mean 2, pop std 1) → in-group [-1, +1].
    Group 1 = [99, 101] (group mean 100, pop std 1) → in-group [-1, +1].
    Global pool: mean = 51, std ≈ 49.5 → normalized values cluster near
                 [-1.01, -0.97, +0.97, +1.01] — close to [-1, +1] only by accident,
                 but the per-sample assignments to each group flip badly when
                 lp_new differs across samples.
    Simpler check: with ratio=1 and equal weights, policy_loss = -mean(norm).
    Per-group: mean(-1, 1, -1, 1) = 0 → policy_loss = 0.
    Global  : mean ≈ 0 too — not separating.

    Use ratio that varies per-sample so the loss depends on assignment.
        lp_new = [0, 0, log(2), log(2)] → ratios [1,1,2,2].
        Per-group adv = [-1, +1, -1, +1] → surr1 = [-1, +1, -2, +2];
        surr2 = [-0.8, +0.8, -1.2, +1.2]. min = [-1, +0.8, -2, +1.2].
        policy_loss = -((-1 + 0.8 - 2 + 1.2)/4) = -(-1/4) = 0.25.

    Global adv approximation ≈ [-1.0202, -0.9798, +0.9798, +1.0202] (similar
    but not identical). policy_loss differs by ~0.005 — still > 1e-5.
    """
    B, T = 4, 1
    lp_new = torch.tensor([[0.0], [0.0], [math.log(2.0)], [math.log(2.0)]])
    lp_old = torch.zeros(B, T)
    adv = torch.tensor([1.0, 3.0, 99.0, 101.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    ctx = LossContext(extras={
        "log_probs_new": lp_new, "log_probs_old": lp_old,
        "advantages": adv, "group_ids": group_ids,
    })
    out = GRPOLoss(clip_eps=0.2)(_DUMMY, {}, ctx)
    # Per-group norm: each group → [-1, +1].
    # surr1 = ratio · adv = [1·-1, 1·+1, 2·-1, 2·+1] = [-1, +1, -2, +2]
    # surr2 = clamp(ratio)·adv with clip_eps=0.2:
    #   ratios [1,1,2,2] clamped to [1,1,1.2,1.2] → [-1, +1, -1.2, +1.2]
    # min = [-1, +1, -2, +1.2]
    # policy_loss = -mean = -(-1+1-2+1.2)/4 = -(-0.8/4) = +0.2
    torch.testing.assert_close(out["loss"], torch.tensor(0.2), atol=1e-5, rtol=1e-4)
