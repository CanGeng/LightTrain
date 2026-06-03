"""Adversarial tests for lighttrain.builtin_plugins.rl.gae (compute_gae / normalize_advantages)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.rl.gae import compute_gae, normalize_advantages

# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------


def test_gae_lambda_zero_equals_td_residual():
    """Goal: λ=0 → A_t = δ_t = r_t + γ·V_{t+1} - V_t (TD(0) residual).

    Input: rewards = [1, 1, 1], values = [0, 0, 0], γ=0.9, last_value=0.
    Analytical:
        t=2: A_2 = 1 + 0.9·0 - 0 = 1
        t=1: A_1 = 1 + 0.9·0 - 0 = 1
        t=0: A_0 = 1 + 0.9·0 - 0 = 1
    Same recursion regardless of λ when values are zero — use non-zero
    values to actually exercise λ=0:
        rewards = [1, 2, 3], values = [0.5, 1.5, 2.5], γ=0.9.
        t=2: A_2 = 3 + 0.9·0 - 2.5 = 0.5
        t=1: A_1 = 2 + 0.9·2.5 - 1.5 = 2 + 2.25 - 1.5 = 2.75
        t=0: A_0 = 1 + 0.9·1.5 - 0.5 = 1 + 1.35 - 0.5 = 1.85
    """
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 1.5, 2.5]])
    adv, ret = compute_gae(rewards, values, gamma=0.9, lam=0.0, last_value=0.0)
    expected_adv = torch.tensor([[1.85, 2.75, 0.5]])
    torch.testing.assert_close(adv, expected_adv, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ret, expected_adv + values, atol=1e-5, rtol=1e-4)


def test_gae_lambda_one_equals_full_monte_carlo():
    """Goal: λ=1 → A_t = Σ_{k=0..T-1-t} γ^k · δ_{t+k}.

    Input: rewards = [0, 0, 1], values = [0, 0, 0], γ=0.5, last_value=0.
    Analytical:
        δ_2 = 1 + 0.5·0 - 0 = 1
        δ_1 = 0 + 0.5·0 - 0 = 0
        δ_0 = 0 + 0.5·0 - 0 = 0
        A_2 = δ_2 = 1
        A_1 = δ_1 + 0.5·A_2 = 0 + 0.5·1 = 0.5
        A_0 = δ_0 + 0.5·A_1 = 0 + 0.25 = 0.25
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 0.0]])
    adv, ret = compute_gae(rewards, values, gamma=0.5, lam=1.0, last_value=0.0)
    expected = torch.tensor([[0.25, 0.5, 1.0]])
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=1e-4)


def test_gae_gamma_zero_advantage_equals_immediate_reward_minus_value():
    """Goal: γ=0 → A_t = r_t - V_t (no future bootstrap).

    Input: rewards = [2, 5, 7], values = [1, 4, 6], γ=0, any λ.
    Analytical: A = rewards - values = [1, 1, 1].
    """
    rewards = torch.tensor([[2.0, 5.0, 7.0]])
    values = torch.tensor([[1.0, 4.0, 6.0]])
    adv, _ = compute_gae(rewards, values, gamma=0.0, lam=0.95, last_value=99.0)
    expected = torch.tensor([[1.0, 1.0, 1.0]])
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=1e-4)


def test_gae_returns_equals_adv_plus_values_exact():
    """Goal: returns == advantages + values for any input.

    Input: random rewards/values.
    Analytical: by definition of returns in GAE (value targets).
    """
    torch.manual_seed(31)
    B, T = 2, 5
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    adv, ret = compute_gae(rewards, values, gamma=0.95, lam=0.9)
    torch.testing.assert_close(ret, adv + values, atol=1e-6, rtol=1e-5)


def test_gae_done_mask_zeros_bootstrap_at_episode_end():
    """Goal: done=True at t=K stops the bootstrap (next_val and last_gae cleared).

    Input: rewards = [1, 1, 1, 1], values = [0, 0, 0, 0], γ=0.9, λ=1.
           dones = [F, T, F, F] — episode ends at t=1.
    Analytical (reverse pass):
        t=3: A_3 = 1 + 0.9·0 - 0 = 1; last_gae=1
        t=2: A_2 = 1 + 0.9·0 - 0 + 0.9·1·1 = 1.9
        t=1 (done=True): next_val zeroed, last_gae zeroed BEFORE computing δ.
              δ_1 = 1 + 0.9·0 - 0 = 1; A_1 = 1 + 0.9·1·0 = 1; last_gae = 1.
        t=0: A_0 = 1 + 0.9·(values[1]=0) - 0 + 0.9·1·1 = 1.9
    """
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    dones = torch.tensor([[False, True, False, False]])
    adv, _ = compute_gae(rewards, values, gamma=0.9, lam=1.0, last_value=0.0, dones=dones)
    expected = torch.tensor([[1.9, 1.0, 1.9, 1.0]])
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=1e-4)


def test_gae_1d_input_consistent_with_2d_squeeze():
    """Goal: 1D input is equivalent to (1, T) 2D input + squeeze."""
    torch.manual_seed(32)
    rewards_1d = torch.randn(5)
    values_1d = torch.randn(5)
    adv_1d, ret_1d = compute_gae(rewards_1d, values_1d, gamma=0.9, lam=0.95)
    adv_2d, ret_2d = compute_gae(
        rewards_1d.unsqueeze(0), values_1d.unsqueeze(0), gamma=0.9, lam=0.95
    )
    torch.testing.assert_close(adv_1d, adv_2d.squeeze(0), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ret_1d, ret_2d.squeeze(0), atol=1e-6, rtol=1e-5)


def test_gae_last_value_bootstrap_used_at_terminal_step():
    """Goal: last_value parameter is used as V(s_T) for the last step.

    Input: rewards = [0], values = [0], γ=0.9, last_value=10.
    Analytical: δ_0 = 0 + 0.9·10 - 0 = 9. A_0 = 9.
    """
    rewards = torch.tensor([[0.0]])
    values = torch.tensor([[0.0]])
    adv, _ = compute_gae(rewards, values, gamma=0.9, lam=0.95, last_value=10.0)
    torch.testing.assert_close(adv, torch.tensor([[9.0]]), atol=1e-5, rtol=1e-4)


def test_regression_gae_backward_recursion_direction():
    """Regression pin for ``gae_recursion_direction``.

    Bug: a forward recursion (t = 0 → T-1) instead of backward (T-1 → 0)
    produces a different sequence.

    Input: rewards = [0, 0, 1], values = [0, 0, 0], γ=0.5, λ=1.
    Backward (correct): adv = [0.25, 0.5, 1.0] (from lambda_one test above).
    Forward (buggy): would propagate from t=0, giving zeros until t=2 → A=[0,0,1].
    Difference at A_0 is 0.25 vs 0 → catches the bug.
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 0.0]])
    adv, _ = compute_gae(rewards, values, gamma=0.5, lam=1.0, last_value=0.0)
    torch.testing.assert_close(adv, torch.tensor([[0.25, 0.5, 1.0]]), atol=1e-5, rtol=1e-4)
    assert adv[0, 0].item() > 0.2, "Backward recursion must propagate future reward back to A_0."


# ---------------------------------------------------------------------------
# normalize_advantages
# ---------------------------------------------------------------------------


def test_normalize_advantages_zero_mean_unit_std_pop():
    """Goal: output has approximate mean 0 and std (pop) 1.

    Input: random advantages.
    Analytical: by construction.
    """
    torch.manual_seed(33)
    adv = torch.randn(100) * 5.0 + 7.0
    normalized = normalize_advantages(adv)
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=1e-4
    )


def test_normalize_advantages_eps_prevents_div_by_zero():
    """Goal: all-equal advantages → std=0 → clamped to eps; no NaN/Inf produced.

    Input: constant tensor.
    Analytical: result = (adv - mean) / eps = 0 / eps = 0 everywhere.
    """
    adv = torch.full((10,), 5.0)
    out = normalize_advantages(adv, eps=1e-8)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(adv), atol=1e-5, rtol=1e-4)


def test_normalize_advantages_mask_uses_only_unmasked_for_stats():
    """Goal: with a mask, mean/std are computed over the unmasked subset only.

    Input: adv = [0, 100, 0, 100], mask = [True, False, True, False].
    Analytical: unmasked = [0, 0]; mean=0, std=0 → output unchanged-by-eps =
                (adv - 0) / eps = adv / eps for masked-out positions.
                More useful: adv = [0, 2, 4, 6], mask = [T, F, T, F].
                unmasked = [0, 4]; mean=2, std_pop=2.
                normalized = (adv - 2)/2 = [-1, 0, 1, 2].
    """
    adv = torch.tensor([0.0, 2.0, 4.0, 6.0])
    mask = torch.tensor([True, False, True, False])
    out = normalize_advantages(adv, mask=mask)
    expected = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)
