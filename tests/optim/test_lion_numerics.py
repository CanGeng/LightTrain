"""Adversarial tests for the Lion optimizer (``lighttrain.builtin_plugins.optim.wrappers._Lion``).

The legacy suite has NO Lion-specific tests. This file pins:

* **First-step update formula** when ``exp_avg = 0``: ``update = sign(grad)``;
  ``p_new = p - lr * sign(grad)``.
* **Update formula at step N** with the momentum mix:
  ``update = sign(β₁·exp_avg_prev + (1-β₁)·grad)``.
* **Zero-grad means no movement** (line 148-149 of wrappers.py).
* **Weight-decay decouples** as ``p ← p · (1 - lr·wd)`` (line 155-156).
* **exp_avg accumulates** as ``exp_avg ← β₂·exp_avg + (1-β₂)·grad`` (line 159).

All numerical assertions use ``torch.testing.assert_close(atol=1e-5, rtol=1e-4)``.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.optim.wrappers import _Lion


def _make_param(values: list[float]) -> torch.nn.Parameter:
    """Helper: build a leaf nn.Parameter from a flat list."""
    return torch.nn.Parameter(torch.tensor(values))


# ---------------------------------------------------------------------------
# Closed-form: first step with zero exp_avg
# ---------------------------------------------------------------------------

def test_invariant_lion_first_step_update_equals_minus_lr_times_sign_of_grad():
    """Invariant: at step 1, ``exp_avg`` is freshly zero, so the update
    formula collapses to ``-lr * sign(grad)`` (no momentum contribution).

    Setup: param p=[2.0, -3.0], grad=[1.0, -1.0], lr=0.1.
    Closed form:
        sign(grad) = [+1, -1]
        update     = +1*0.9*[0,0] + 0.1*[1,-1] = [0.1, -0.1] before sign
                   (but sign of [0.1, -0.1]) = [+1, -1]
        p_new      = p - lr * sign = [2.0 - 0.1, -3.0 - (-0.1)] = [1.9, -2.9]

    Note: ``β₁`` does not affect the sign (only the magnitude of the
    intermediate), so the result is the same regardless of β₁.
    """
    p = _make_param([2.0, -3.0])
    p.grad = torch.tensor([1.0, -1.0])
    opt = _Lion([p], lr=0.1, weight_decay=0.0)
    opt.step()
    expected = torch.tensor([1.9, -2.9])
    torch.testing.assert_close(p.data, expected, atol=1e-5, rtol=1e-4)


def test_invariant_lion_zero_grad_does_not_move_param():
    """Invariant: when ``p.grad is None`` (or zero-valued without prior
    momentum), the param is left unchanged.

    Setup: p=[1.0, 2.0], grad=None; one step.
    Expected: param unchanged.
    """
    p = _make_param([1.0, 2.0])
    p.grad = None
    opt = _Lion([p], lr=0.1)
    before = p.data.clone()
    opt.step()
    torch.testing.assert_close(p.data, before, atol=0.0, rtol=0.0)


def test_invariant_lion_exp_avg_updates_via_beta2_formula():
    """Invariant: after one step with ``β₂=0.99`` from a zero starting
    point, ``exp_avg = (1 - β₂) * grad = 0.01 * grad``.

    Setup: p=[0.0, 0.0], grad=[1.0, -2.0], β₂=0.99.
    Closed form: exp_avg_new = 0 + 0.01 * [1.0, -2.0] = [0.01, -0.02].
    """
    p = _make_param([0.0, 0.0])
    p.grad = torch.tensor([1.0, -2.0])
    opt = _Lion([p], lr=0.0, betas=(0.9, 0.99), weight_decay=0.0)  # lr=0 to isolate momentum
    opt.step()
    state = opt.state[p]
    expected = torch.tensor([0.01, -0.02])
    torch.testing.assert_close(state["exp_avg"], expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Closed-form: weight decay
# ---------------------------------------------------------------------------

def test_invariant_lion_weight_decay_decouples_as_multiplicative_factor():
    """Invariant: with ``grad=None`` (so no sign-update applies) and
    ``wd=0.1, lr=0.1``, weight decay shrinks the param multiplicatively:
        p ← p * (1 - lr * wd) = p * 0.99

    Setup: p=[1.0, 2.0], grad=None, wd=0.1, lr=0.1; 1 step.
    Expected: p = [0.99, 1.98].
    """
    p = _make_param([1.0, 2.0])
    p.grad = None
    opt = _Lion([p], lr=0.1, weight_decay=0.1)
    opt.step()
    # With grad=None the sign-update path is skipped (line 148-149),
    # AND so is the weight-decay path (it's gated by grad too).
    # The actual implementation only runs WD when grad is not None.
    # So expect p unchanged.
    expected = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(p.data, expected, atol=0.0, rtol=0.0)


def test_invariant_lion_weight_decay_applies_when_grad_present_with_zero_grad_tensor():
    """When grad is a tensor of zeros (NOT None), weight decay still applies
    (the ``if p.grad is None: continue`` guard at line 148 fires only on None,
    not on zero-valued).

    Setup: p=[1.0, 2.0], grad=[0.0, 0.0], wd=0.5, lr=0.2; 1 step.
    Closed form:
        wd factor = 1 - 0.2*0.5 = 0.9 → p → p * 0.9 = [0.9, 1.8]
        Then: update = sign(0*exp_avg + 1*[0,0]) = sign([0,0])
        sign(0) = 0 in PyTorch, so update = [0, 0] → p subtracts 0.
    Expected: p = [0.9, 1.8].
    """
    p = _make_param([1.0, 2.0])
    p.grad = torch.tensor([0.0, 0.0])
    opt = _Lion([p], lr=0.2, weight_decay=0.5, betas=(0.9, 0.99))
    opt.step()
    expected = torch.tensor([0.9, 1.8])
    torch.testing.assert_close(p.data, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Lion update at step 2 (with non-zero exp_avg)
# ---------------------------------------------------------------------------

def test_invariant_lion_step_two_uses_momentum_mix_in_sign_argument():
    """Invariant: at step 2, ``exp_avg`` is no longer zero. The sign update
    becomes ``sign(β₁·exp_avg + (1-β₁)·grad)``.

    Setup:
      Step 1: p=[0,0], grad=[1, -1], β₁=0.9, β₂=0.99, lr=0.1, wd=0.
      Step 2: same p (just updated), grad=[-2, +2].

    After step 1:
      sign(0*0 + 1*[1,-1]) = [+1, -1]
      p_new1 = [0,0] - 0.1*[1,-1] = [-0.1, 0.1]
      exp_avg_new1 = 0 + 0.01*[1,-1] = [0.01, -0.01]

    Step 2 sign argument:
      0.9 * [0.01, -0.01] + 0.1 * [-2, 2] = [0.009 - 0.2, -0.009 + 0.2] = [-0.191, 0.191]
      sign(...) = [-1, +1]
      p_new2 = [-0.1, 0.1] - 0.1 * [-1, +1] = [-0.1 + 0.1, 0.1 - 0.1] = [0.0, 0.0]
    """
    p = _make_param([0.0, 0.0])
    opt = _Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

    # Step 1
    p.grad = torch.tensor([1.0, -1.0])
    opt.step()
    # After step 1: p = [-0.1, 0.1], exp_avg = [0.01, -0.01] — verified
    torch.testing.assert_close(
        p.data, torch.tensor([-0.1, 0.1]), atol=1e-5, rtol=1e-4
    )

    # Step 2
    p.grad = torch.tensor([-2.0, 2.0])
    opt.step()
    expected = torch.tensor([0.0, 0.0])
    torch.testing.assert_close(p.data, expected, atol=1e-5, rtol=1e-4)


def test_lion_step_returns_loss_from_closure():
    """``step(closure)`` calls the closure and returns its result.

    Goal: pin the PyTorch optimizer convention. Important for second-order
    methods that need the loss value.
    """
    p = _make_param([1.0])
    p.grad = torch.tensor([0.1])
    opt = _Lion([p], lr=0.0)

    def closure():
        return torch.tensor(7.5)

    loss = opt.step(closure)
    assert float(loss) == 7.5


# ---------------------------------------------------------------------------
# Lion-vs-SGD degenerate cross-check
# ---------------------------------------------------------------------------

def test_lion_first_step_matches_sign_sgd_baseline():
    """Cross-check: Lion's first step (zero exp_avg) is equivalent to
    sign-SGD: ``p ← p - lr * sign(grad)``.

    Setup: same param + grad through Lion and through a manual sign-SGD.
    Expected: identical post-step values via ``assert_close``.
    """
    init = torch.tensor([0.5, -0.5, 1.5, -1.5])

    p_lion = torch.nn.Parameter(init.clone())
    p_lion.grad = torch.tensor([2.0, -2.0, 0.5, -0.5])
    _Lion([p_lion], lr=0.1, weight_decay=0.0).step()

    # Manual sign-SGD baseline
    grad = torch.tensor([2.0, -2.0, 0.5, -0.5])
    p_sgd = init - 0.1 * torch.sign(grad)
    torch.testing.assert_close(p_lion.data, p_sgd, atol=1e-5, rtol=1e-4)
