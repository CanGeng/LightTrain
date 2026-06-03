"""Adversarial tests for ``lighttrain.builtin_plugins.optim.schedulers``.

The legacy suite has no scheduler-specific tests. This file pins
closed-form values for:

* **Constant**: factor=1.0 at every step.
* **Linear**: warmup ramp, post-warmup linear decay, endpoint = end_factor,
  progress clamped to [0, 1] beyond ``total_steps``.
* **WarmupCosine**: warmup ramp, cosine endpoints at progress=0 and 1,
  midpoint value, ``min_lr_ratio`` floor at the end.
* **WSD**: three-phase shape — warmup → stable (factor=1.0) → linear decay
  to ``min_lr_ratio``.
* **base_lrs captured at attach time**: post-attach mutation of the
  optimizer's ``lr`` does not affect schedule (pin).
* **state_dict round-trip** preserves ``last_step`` and ``base_lrs``.
* **step_per_batch is always True** for all four scheduler classes.
"""

from __future__ import annotations

import math

import pytest
import torch

from lighttrain.builtin_plugins.optim.schedulers import (
    ConstantScheduler,
    LinearScheduler,
    WSDScheduler,
    WarmupCosineScheduler,
)


def _fresh_opt(lr: float = 1.0) -> torch.optim.Optimizer:
    """Tiny optimizer with one trainable scalar param at the given lr."""
    p = torch.nn.Parameter(torch.tensor([0.0]))
    return torch.optim.SGD([p], lr=lr)


# ---------------------------------------------------------------------------
# ConstantScheduler
# ---------------------------------------------------------------------------

def test_constant_scheduler_keeps_lr_unchanged_across_steps():
    """``ConstantScheduler`` never changes the lr — same value every step.

    Setup: optimizer at lr=0.5; tick 1, 5, 50 steps.
    Expected: lr stays at 0.5 each time.
    """
    opt = _fresh_opt(lr=0.5)
    sched = ConstantScheduler(opt)
    for _ in range(50):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LinearScheduler — endpoint, midpoint, post-end clamping
# ---------------------------------------------------------------------------

def test_invariant_linear_scheduler_warmup_then_decay_endpoints():
    """Closed-form: with warmup=2, total=10, end_factor=0:
        step=1: warming up → factor = 1/2 = 0.5
        step=2: warmup complete → factor = 1.0
        step=10: linear decay end → factor = 0.0
        step=20: post-end clamp → factor = 0.0 (progress clamped to 1)

    Setup: base lr=1.0 so factor == reported lr.
    """
    opt = _fresh_opt(lr=1.0)
    sched = LinearScheduler(opt, total_steps=10, end_factor=0.0, warmup_steps=2)

    sched.step()  # step 1, warming up
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=1e-6)

    sched.step()  # step 2, warmup done
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=1e-6)

    # Drive to step 10
    for _ in range(8):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)

    # Beyond total_steps: progress clamped
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)


def test_invariant_linear_scheduler_post_warmup_midpoint():
    """Closed-form: warmup=0, total=4, end_factor=0.5.
        step=2: progress = (2-0)/(4-0) = 0.5
        factor = 1.0 + (0.5 - 1.0) * 0.5 = 0.75
    """
    opt = _fresh_opt(lr=1.0)
    sched = LinearScheduler(opt, total_steps=4, end_factor=0.5, warmup_steps=0)
    sched.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------------
# WarmupCosineScheduler — endpoint, midpoint, min_lr floor
# ---------------------------------------------------------------------------

def test_invariant_warmup_cosine_warmup_phase_linear_in_step():
    """Warmup phase ramps linearly from 0 to 1 over ``warmup_steps``.

    Closed form (warmup=4, total=10): step=2 → factor = 2/4 = 0.5.
    """
    opt = _fresh_opt(lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=4, total_steps=10, min_lr_ratio=0.0)
    sched.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=1e-6)


def test_invariant_warmup_cosine_at_total_step_equals_min_lr_ratio():
    """At step=total_steps, progress=1 → cos(π) = -1 → factor = min_lr_ratio.

    Closed form (warmup=2, total=10, min_lr=0.1):
        progress at step=10 = (10-2)/(10-2) = 1.0
        cosine = 0.5 * (1 + cos(π)) = 0
        factor = 0.1 + (1 - 0.1) * 0 = 0.1
    """
    opt = _fresh_opt(lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=2, total_steps=10, min_lr_ratio=0.1)
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)


def test_invariant_warmup_cosine_midpoint_value():
    """Closed form (warmup=0, total=4, min_lr=0):
        step=2: progress = 2/4 = 0.5
        cosine = 0.5 * (1 + cos(π/2)) = 0.5
        factor = 0 + (1-0) * 0.5 = 0.5
    """
    opt = _fresh_opt(lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=4, min_lr_ratio=0.0)
    sched.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=1e-6)


def test_invariant_warmup_cosine_progress_clamped_post_total():
    """Beyond total_steps, progress clamped to 1.0 → lr stays at min_lr_ratio."""
    opt = _fresh_opt(lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=4, min_lr_ratio=0.2)
    for _ in range(50):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# WSDScheduler — 3-phase invariants
# ---------------------------------------------------------------------------

def test_invariant_wsd_warmup_phase_linear():
    """Warmup phase: step=warmup_steps/2 → factor = 0.5."""
    opt = _fresh_opt(lr=1.0)
    sched = WSDScheduler(opt, warmup_steps=4, stable_steps=8, decay_steps=4, min_lr_ratio=0.1)
    for _ in range(2):  # halfway through warmup
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5, abs=1e-6)


def test_invariant_wsd_stable_phase_holds_lr_at_base():
    """During the stable phase (after warmup, before decay), factor=1.0.

    Setup: warmup=2, stable=6, decay=4. Step to step=4 (mid-stable) and
    step=8 (last of stable).
    """
    opt = _fresh_opt(lr=1.0)
    sched = WSDScheduler(opt, warmup_steps=2, stable_steps=6, decay_steps=4, min_lr_ratio=0.0)
    # Drive past warmup into the stable phase.
    for _ in range(4):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=1e-6)
    # Still stable a few steps later
    for _ in range(4):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=1e-6)


def test_invariant_wsd_decay_phase_reaches_min_lr_ratio_at_end():
    """At the end of the decay phase (step = warmup + stable + decay), the
    factor equals ``min_lr_ratio``.

    Setup: warmup=2, stable=4, decay=4, min_lr_ratio=0.1.
        end step = 2 + 4 + 4 = 10
        progress at step 10 = (10-2-4)/4 = 1.0
        factor = 1 + (0.1 - 1) * 1.0 = 0.1
    """
    opt = _fresh_opt(lr=1.0)
    sched = WSDScheduler(opt, warmup_steps=2, stable_steps=4, decay_steps=4, min_lr_ratio=0.1)
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)


def test_invariant_wsd_post_end_clamped_to_min_lr_ratio():
    """Beyond the decay phase, progress is clamped to 1.0 → lr stays at min."""
    opt = _fresh_opt(lr=1.0)
    sched = WSDScheduler(opt, warmup_steps=2, stable_steps=4, decay_steps=4, min_lr_ratio=0.3)
    for _ in range(50):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# base_lrs capture invariant
# ---------------------------------------------------------------------------

def test_invariant_base_lrs_captured_at_attach_time():
    """Pin: ``_base_lrs`` is captured ONCE at ``attach()`` time. Mutating the
    optimizer's lr after attach does NOT affect the schedule's factor
    calculations.

    Setup: lr=1.0 at attach; after attach mutate to lr=999; tick. Expected
    lr reflects 1.0 (the captured base), not 999.
    """
    opt = _fresh_opt(lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=10, min_lr_ratio=0.0)
    # Mutate underlying optimizer's lr — sched should ignore this for scaling.
    opt.param_groups[0]["lr"] = 999.0
    sched.step()
    # Step 1 of 10 with warmup_steps=0 →
    #   progress = (1-0)/(10-0) = 0.1 → cosine = 0.5*(1 + cos(0.1π)) ≈ 0.97553
    progress = 0.1
    expected_factor = 0.0 + (1.0 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
    expected = 1.0 * expected_factor  # base_lr captured = 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------

def test_invariant_state_dict_round_trip_preserves_last_step():
    """Save scheduler state after N ticks → load into a fresh scheduler
    instance → ``last_step`` equals the saved value AND base_lrs roundtrip.
    """
    opt = _fresh_opt(lr=0.7)
    s1 = LinearScheduler(opt, total_steps=10, end_factor=0.0, warmup_steps=0)
    for _ in range(3):
        s1.step()
    sd = s1.state_dict()

    opt2 = _fresh_opt(lr=0.1)  # different lr to verify base_lrs round-trip
    s2 = LinearScheduler(opt2, total_steps=10, end_factor=0.0, warmup_steps=0)
    s2.load_state_dict(sd)
    assert s2.last_step == 3
    assert s2._base_lrs == [0.7]  # loaded from sd, not s2's own attach


# ---------------------------------------------------------------------------
# step_per_batch flag pin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sched_cls",
    [ConstantScheduler, LinearScheduler, WarmupCosineScheduler, WSDScheduler],
)
def test_pin_all_schedulers_have_step_per_batch_true(sched_cls):
    """Pin: every scheduler has ``step_per_batch=True`` (class attribute).

    Setup: each scheduler class.
    Expected: class attribute is True (drives the trainer loop's
    tick-once-per-batch dispatch).

    If you intentionally add a once-per-epoch scheduler, that class would
    have ``step_per_batch=False`` and not appear in this parametrize list.
    """
    assert sched_cls.step_per_batch is True


# ---------------------------------------------------------------------------
# Unattached scheduler is a no-op
# ---------------------------------------------------------------------------

def test_unattached_scheduler_step_is_safe_no_op():
    """``step()`` on a scheduler with ``optimizer=None`` does NOT raise —
    line 33-34 of schedulers.py early-returns from ``_set_lrs`` when
    optimizer is None.

    Goal: pin the no-op contract so a trainer can construct a scheduler
    and attach later without crashing.
    """
    sched = ConstantScheduler(optimizer=None)
    sched.step()
    assert sched.last_step == 1
    assert sched.get_lr() == []
