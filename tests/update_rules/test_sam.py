"""Adversarial tests for SAMUpdateRule — two-pass Sharpness-Aware Minimisation.

SAM algorithm:
  1. forward + backward at θ → g
  2. perturb θ → θ + ê where ê = ρ · g / ||g||
  3. forward + backward at θ + ê → ĝ
  4. restore θ
  5. optimizer.step() using ĝ (the *perturbed-state* gradient)

Key contracts we pin:
  - Two forward-backwards happen in order; perturb between, restore after.
  - Accumulation boundary skips the SAM perturbation (early-return path).
  - optimizer.step receives params at the ORIGINAL θ (not perturbed).
  - SAM does NOT use grad_sync (documented gap).
  - SAM does NOT honor SKIP_STEP (documented gap, pinned via xfail).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.update_rules.sam import SAMUpdateRule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x, **_):
        return ModelOutput(outputs={"logits": self.linear(x)})


def _simple_loss(model_output, batch, ctx):
    pred = model_output.outputs["logits"]
    return {"loss": (pred - 1.0).pow(2).mean()}


def _build_ctx(*, callbacks=None, accelerator=None, scheduler=None):
    model = _TinyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    ctx = StepContext(
        model=model,
        optimizer=optim,
        bus=EventBus(callbacks or []),
        loss_fn=_simple_loss,
        scheduler=scheduler,
        accelerator=accelerator,
    )
    return ctx, model, optim


def _batch():
    torch.manual_seed(7)
    return {"x": torch.randn(2, 4)}


class _OrderedRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def _h(name):
        def _f(self, **_kw):
            self.events.append(name)

        return _f

    on_step_begin = _h("on_step_begin")
    on_forward_post = _h("on_forward_post")
    on_loss_computed = _h("on_loss_computed")
    on_backward_pre = _h("on_backward_pre")
    on_backward_post = _h("on_backward_post")
    on_clip_grad = _h("on_clip_grad")
    on_optimizer_step_pre = _h("on_optimizer_step_pre")
    on_optimizer_step_post = _h("on_optimizer_step_post")
    on_scheduler_step = _h("on_scheduler_step")
    on_step_end = _h("on_step_end")


# ===========================================================================
# Two-pass structure & ordering
# ===========================================================================


def test_sam_two_passes_fire_perturb_then_restore_then_optimizer_step():
    """Goal: pin the SAM control-flow sequence using spy patches on
    ``_compute_perturbation`` and ``_restore``.

    Construction: patch the rule's internal methods so each records its
    invocation; also wrap ``optimizer.step`` with a spy. Then run one full
    step (no accumulation).

    Expected ordered sequence of recorded events:
      [forward1, compute_perturbation, forward2, restore, optimizer.step]

    Catches a refactor that:
      - swaps perturbation and restore order (would optimizer.step at perturbed θ)
      - drops the second forward (would compute SGD instead of SAM)
      - moves optimizer.step before restore (would step at perturbed θ)
    """
    ordered: list[str] = []
    rule = SAMUpdateRule(rho=0.05, accumulate_grad_batches=1, grad_clip=0.0)

    real_perturb = rule._compute_perturbation
    real_restore = rule._restore

    def _spy_perturb(model):
        ordered.append("compute_perturbation")
        return real_perturb(model)

    def _spy_restore(model, perts):
        ordered.append("restore")
        return real_restore(model, perts)

    rule._compute_perturbation = _spy_perturb
    rule._restore = _spy_restore

    # Track forward and optimizer.step
    ctx, model, optim = _build_ctx()
    fwd_orig = model.forward

    def _spy_forward(**kw):
        ordered.append("forward")
        return fwd_orig(**kw)

    model.forward = _spy_forward

    step_orig = optim.step

    def _spy_optim_step(*a, **kw):
        ordered.append("optimizer_step")
        return step_orig(*a, **kw)

    optim.step = _spy_optim_step

    rule.step(model, _batch(), ctx)

    expected = [
        "forward",  # pass 1
        "compute_perturbation",
        "forward",  # pass 2
        "restore",
        "optimizer_step",
    ]
    assert ordered == expected


def test_sam_lifecycle_events_fire_in_exact_order():
    """Goal: SAM's bus event sequence is shorter than Standard's — pin it.

    Construction: scheduler with step_per_batch=True attached so on_scheduler_step
    is fired.

    Expected list (NO on_forward_pre — SAM does not dispatch this hook):
      [on_step_begin, on_forward_post, on_loss_computed, on_backward_pre,
       on_backward_post, on_clip_grad, on_optimizer_step_pre,
       on_optimizer_step_post, on_scheduler_step, on_step_end]

    Catches a refactor that adds spurious events or swaps order.
    """
    rec = _OrderedRecorder()
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    ctx, model, _ = _build_ctx(callbacks=[rec], scheduler=scheduler)

    SAMUpdateRule(grad_clip=1.0).step(model, _batch(), ctx)

    expected = [
        "on_step_begin",
        "on_forward_post",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_scheduler_step",
        "on_step_end",
    ]
    assert rec.events == expected


# ===========================================================================
# Accumulation
# ===========================================================================


def test_sam_accumulation_boundary_skips_perturbation():
    """Goal: on intermediate micro-steps (not at the K-boundary), SAM skips
    its perturbation pass entirely (line 184-192).

    Input: accumulate_grad_batches=2, single micro-step ⇒ accumulating=True.

    Expected:
      - _compute_perturbation NOT called
      - second forward NOT called
      - optimizer.step NOT called

    Catches a refactor that always perturbs (would double-cost on every
    micro-step + step optimizer at wrong cadence).
    """
    rule = SAMUpdateRule(rho=0.05, accumulate_grad_batches=2)

    compute_called = [0]
    rule._compute_perturbation = lambda m: (compute_called.__setitem__(0, compute_called[0] + 1) or [])

    ctx, model, optim = _build_ctx()
    fwd_count = [0]
    fwd_orig = model.forward

    def _count_fwd(**kw):
        fwd_count[0] += 1
        return fwd_orig(**kw)

    model.forward = _count_fwd
    step_spy = MagicMock(wraps=optim.step)
    optim.step = step_spy

    rule.step(model, _batch(), ctx)

    assert compute_called[0] == 0
    assert fwd_count[0] == 1  # only pass 1 ran
    step_spy.assert_not_called()


# ===========================================================================
# Perturb / restore symmetry
# ===========================================================================


def test_sam_perturbation_restored_before_optimizer_step():
    """Goal: at the moment ``optimizer.step()`` runs, the params must be at
    the ORIGINAL θ (not perturbed). Otherwise we'd update from the
    perturbed weight — defeating SAM's design.

    Construction: snapshot θ before step. Use an ``on_optimizer_step_pre``
    callback to record a snapshot of θ at that exact moment. Compare.

    Expected: snapshot_at_optimizer_step == snapshot_before_step (exact),
    using ``torch.testing.assert_close`` with tight tolerance.

    Catches a refactor that swaps the order ``optimizer.step → _restore``.
    """
    snapshot_at_optim_step: list[torch.Tensor] = []

    class _SnapshotCb:
        def on_optimizer_step_pre(self, **_):
            snapshot_at_optim_step.append(
                ctx.model.linear.weight.detach().clone()
            )

    ctx, model, _ = _build_ctx(callbacks=[_SnapshotCb()])
    before = model.linear.weight.detach().clone()

    SAMUpdateRule(rho=0.5, grad_clip=0.0).step(model, _batch(), ctx)

    assert len(snapshot_at_optim_step) == 1
    torch.testing.assert_close(
        snapshot_at_optim_step[0], before, atol=1e-5, rtol=1e-4
    )


def test_sam_state_dict_roundtrip_preserves_micro_step():
    """Goal: micro_step persists across state_dict roundtrip for accumulation
    resumption.
    """
    rule = SAMUpdateRule(rho=0.07, accumulate_grad_batches=3, grad_clip=0.4)
    rule._micro_step = 2
    sd = rule.state_dict()
    assert sd["micro_step"] == 2
    assert sd["rho"] == 0.07
    assert sd["accumulate_grad_batches"] == 3
    assert sd["grad_clip"] == 0.4

    rule2 = SAMUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2._micro_step == 2
    assert rule2.rho == 0.07
    assert rule2.accumulate_grad_batches == 3
    assert rule2.grad_clip == 0.4


# ===========================================================================
# Documented gaps
# ===========================================================================


def test_sam_no_grad_sync_path_unsupported_gracefully():
    """Goal: pin current behavior — SAM does NOT use ctx.grad_sync; even when
    one is set, the rule runs the bare/accelerator backward path.

    Documents the GAP: SAM and DDP/FSDP are currently incompatible. A
    refactor that adds grad_sync wiring must explicitly opt-in (would
    change the answer to this test).

    Construction: set ctx.grad_sync = recording stub. Run step. Assert
    grad_sync.backward was never called.
    """
    grad_sync_calls: list[str] = []

    class _GradSyncStub:
        def backward(self, loss, model):
            grad_sync_calls.append("backward")
            loss.backward()

        def clip_grad_norm(self, model, max_norm, pctx):
            grad_sync_calls.append("clip")
            return 0.0

        def optimizer_step(self, optimizer, model):
            grad_sync_calls.append("optimizer_step")
            optimizer.step()

        def accumulate(self, model):
            from contextlib import nullcontext

            grad_sync_calls.append("accumulate")
            return nullcontext()

    ctx, model, _ = _build_ctx()
    ctx.grad_sync = _GradSyncStub()

    SAMUpdateRule(grad_clip=0.0).step(model, _batch(), ctx)

    # If a future refactor adds grad_sync wiring to SAM, this test must be
    # updated (it documents current behavior, not a bug).
    assert grad_sync_calls == []


@pytest.mark.xfail(
    strict=True, reason="SAM does not gate on Signal.SKIP_STEP (documented gap)"
)
def test_sam_skip_signal_currently_ignored_xfail():
    """Goal: when SAM is fixed to honor Signal.SKIP_STEP returned from
    ``on_loss_computed``, this xfail will XPASS and force a maintainer to
    flip the assertion.

    What we want SAM to do (asserted here as the *correct* behavior):
      - When on_loss_computed returns SKIP_STEP, the second forward and
        the optimizer.step must NOT happen.

    What SAM does today: ignores the returned signal entirely. So this test
    fails on master — which is why we mark it ``xfail(strict=True)``.
    When SAM gets the fix, the test will XPASS and CI fails loudly,
    forcing the marker to be removed.

    See: TODO/issue link tracking the SAM skip-step fix.
    """

    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    fwd_count = [0]

    ctx, model, optim = _build_ctx(callbacks=[_Skipper()])
    fwd_orig = model.forward

    def _count_fwd(**kw):
        fwd_count[0] += 1
        return fwd_orig(**kw)

    model.forward = _count_fwd
    step_spy = MagicMock(wraps=optim.step)
    optim.step = step_spy

    SAMUpdateRule().step(model, _batch(), ctx)

    # Correct behavior: only one forward, no optimizer step.
    assert fwd_count[0] == 1
    step_spy.assert_not_called()
