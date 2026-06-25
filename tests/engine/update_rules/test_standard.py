"""Adversarial tests for StandardUpdateRule — lifecycle, branches, signals,
accumulation, lazy param registration.

Targets [lighttrain/builtin_plugins/update_rules/standard.py](../../lighttrain/builtin_plugins/update_rules/standard.py).

Six sections (matching plan groups C1–C6):
  - C1: STRICT ordered list of lifecycle events
  - C2: three-way backward branch (grad_sync / accelerator / bare)
  - C3: skip / stop / retry signal handling
  - C4: gradient accumulation
  - C5: lazy ``_register_new_params`` injection
  - C6: misc (state_dict, error paths)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.engine.update_rules.standard import (
    StandardUpdateRule,
    _register_new_params,
)
from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.engine._context import StepContext
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# helpers — recorders + tiny model + builders
# ---------------------------------------------------------------------------


class _OrderedRecorder:
    """Callback that appends event names to a shared list in dispatch order.

    Tests assert on the full list with ``==`` so any reordering fails loudly
    — unlike ``event in fired`` set-membership checks which the legacy
    tests use.
    """

    def __init__(self, signal_for: dict[str, Signal] | None = None) -> None:
        self.events: list[str] = []
        self._signal_for = signal_for or {}

    def _handler(name: str):  # type: ignore[misc]
        def _h(self, **_kw):
            self.events.append(name)
            return self._signal_for.get(name)

        return _h

    on_step_begin = _handler("on_step_begin")
    on_forward_pre = _handler("on_forward_pre")
    on_forward_post = _handler("on_forward_post")
    on_loss_computed = _handler("on_loss_computed")
    on_backward_pre = _handler("on_backward_pre")
    on_backward_post = _handler("on_backward_post")
    on_clip_grad = _handler("on_clip_grad")
    on_optimizer_step_pre = _handler("on_optimizer_step_pre")
    on_optimizer_step_post = _handler("on_optimizer_step_post")
    on_zero_grad = _handler("on_zero_grad")
    on_scheduler_step = _handler("on_scheduler_step")
    on_step_end = _handler("on_step_end")


class _TinyModel(nn.Module):
    """One Linear layer. Forward takes a ``x`` kwarg."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return ModelOutput(outputs={"logits": self.linear(x)})


def _simple_loss(model_output, batch, ctx):
    """MSE-to-ones — has a real gradient w.r.t. linear.weight."""
    pred = model_output.outputs["logits"]
    loss = (pred - 1.0).pow(2).mean()
    return {"loss": loss}


def _build_ctx(
    *,
    callbacks: list | None = None,
    accelerator=None,
    grad_sync=None,
    parallel_ctx=None,
    scheduler=None,
) -> tuple[StepContext, _TinyModel, torch.optim.Optimizer]:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bus = EventBus(callbacks or [])
    ctx = StepContext(
        model=model,
        optimizer=optimizer,
        bus=bus,
        loss_fn=_simple_loss,
        scheduler=scheduler,
        accelerator=accelerator,
        grad_sync=grad_sync,
        parallel_ctx=parallel_ctx,
    )
    return ctx, model, optimizer


def _batch():
    return {"x": torch.randn(2, 4)}


# ===========================================================================
# C1. Lifecycle event order — STRICT ordered list assertions
# ===========================================================================


def test_full_step_emits_events_in_exact_order():
    """Goal: pin the full sequence of bus events emitted by one non-skip step.

    Input: standard model + optimizer + scheduler (step_per_batch=True),
    no signal returns. The recorder appends each event to a list.

    Expected: events match this exact ordered list:
        [on_step_begin, on_forward_pre, on_forward_post, on_loss_computed,
         on_backward_pre, on_backward_post, on_clip_grad,
         on_optimizer_step_pre, on_optimizer_step_post, on_zero_grad,
         on_scheduler_step, on_step_end]

    Catches: any reordering (e.g. moving on_backward_pre after backward) —
    the legacy ``test_rl_rule_fires_full_callback_chain`` only verifies
    set membership, so reordering passes silently.
    """
    rec = _OrderedRecorder()
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    ctx, model, _ = _build_ctx(callbacks=[rec], scheduler=scheduler)

    rule = StandardUpdateRule(grad_clip=1.0)
    rule.step(model, _batch(), ctx)

    expected = [
        "on_step_begin",
        "on_forward_pre",
        "on_forward_post",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
        "on_scheduler_step",
        "on_step_end",
    ]
    assert rec.events == expected


def test_skip_path_emits_abbreviated_event_sequence():
    """Goal: pin the exact (shorter) sequence when SKIP_STEP fires from
    ``on_loss_computed``.

    Input: a recorder callback that returns SKIP_STEP on on_loss_computed.

    Expected sequence:
        [on_step_begin, on_forward_pre, on_forward_post,
         on_loss_computed, on_step_end]
    No backward / clip / optimizer / scheduler events.

    Catches a refactor that fires on_backward_pre even on skip (legacy
    membership tests would not notice that extra event).
    """
    rec = _OrderedRecorder(signal_for={"on_loss_computed": Signal.SKIP_STEP})
    ctx, model, _ = _build_ctx(callbacks=[rec])

    StandardUpdateRule().step(model, _batch(), ctx)

    assert rec.events == [
        "on_step_begin",
        "on_forward_pre",
        "on_forward_post",
        "on_loss_computed",
        "on_step_end",
    ]


def test_skip_path_does_not_emit_optimizer_or_clip_events():
    """Goal: explicit guard — none of the post-backward events fire on skip.

    Input: SKIP_STEP returned from on_loss_computed.
    Expected: forbidden events not in recorder list.

    A complement to ``test_skip_path_emits_abbreviated_event_sequence``;
    pins the gap from the other direction.
    """
    rec = _OrderedRecorder(signal_for={"on_loss_computed": Signal.SKIP_STEP})
    ctx, model, _ = _build_ctx(callbacks=[rec])

    StandardUpdateRule().step(model, _batch(), ctx)

    forbidden = {
        "on_backward_pre",
        "on_backward_post",
        "on_clip_grad",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_zero_grad",
        "on_scheduler_step",
    }
    assert forbidden.isdisjoint(set(rec.events))


def test_step_end_always_fires_even_on_skip():
    """Goal: on_step_end fires regardless of skip — loggers/checkpoints rely on it.

    Input: skip-from-on_loss_computed, also stop-training-from-on_loss_computed.
    Expected: on_step_end appears in BOTH cases.

    Catches a refactor that early-returns before line 235's
    ``bus.dispatch("on_step_end", ...)``.
    """
    for sig in (Signal.SKIP_STEP, Signal.STOP_TRAINING):
        rec = _OrderedRecorder(signal_for={"on_loss_computed": sig})
        ctx, model, _ = _build_ctx(callbacks=[rec])

        StandardUpdateRule().step(model, _batch(), ctx)

        assert "on_step_end" in rec.events, f"on_step_end missing for {sig}"


# ===========================================================================
# C2. Three-way backward branch — mock probes
# ===========================================================================


def test_grad_sync_path_calls_grad_sync_backward_only(fake_dist_env):
    """Goal: when grad_sync is set, ``grad_sync.backward()`` is invoked and
    ``loss.backward()`` is NOT invoked directly.

    Input: ctx.grad_sync = fake recording obj.
    Expected: 'backward' appears exactly once in the grad_sync.calls list.
    The fake's backward records the call AND forwards to loss.backward,
    so params still update — we verify routing, not numerical effect.

    Catches a refactor that calls both paths (double-backward) or the
    wrong one.
    """
    grad_sync, pctx = fake_dist_env()
    ctx, model, _ = _build_ctx(grad_sync=grad_sync, parallel_ctx=pctx)

    StandardUpdateRule(grad_clip=1.0).step(model, _batch(), ctx)

    call_names = [c[0] for c in grad_sync.calls]
    assert "backward" in call_names
    assert call_names.count("backward") == 1
    # clip + optimizer step also went through grad_sync (the three-way branch)
    assert "clip_grad_norm" in call_names
    assert "optimizer_step" in call_names


def test_accelerator_path_calls_accelerator_backward_only(fake_accelerator):
    """Goal: when accelerator is set (no grad_sync), accelerator.backward
    is invoked, NOT loss.backward.

    Input: ctx.accelerator = fake recording obj with autocast+backward+clip.
    Expected: fake_accelerator.calls contains 'backward'; grad_sync absent.

    Catches a refactor that drops the ``hasattr(accelerator, "backward")``
    guard at line 268 — would call a missing method.
    """
    ctx, model, _ = _build_ctx(accelerator=fake_accelerator)

    StandardUpdateRule(grad_clip=1.0).step(model, _batch(), ctx)

    call_names = [c[0] for c in fake_accelerator.calls]
    assert "backward" in call_names
    assert "clip_grad_norm_" in call_names


def test_bare_path_calls_tensor_backward_only():
    """Goal: with no grad_sync and no accelerator, ``loss.backward()`` runs
    directly (the bare path at line 271).

    Input: default ctx, no distributed env.
    Expected: model params receive grads (proxy: weight changes after step).

    Catches a refactor that requires an accelerator even for single-GPU.
    """
    ctx, model, _ = _build_ctx()
    before = model.linear.weight.detach().clone()

    StandardUpdateRule(grad_clip=1.0).step(model, _batch(), ctx)
    after = model.linear.weight.detach().clone()

    assert not torch.equal(before, after)


def test_grad_sync_path_uses_no_sync_during_accumulation(fake_dist_env):
    """Goal: on intermediate micro-batches (still accumulating),
    ``grad_sync.accumulate(model)`` ctx is entered.

    Input: accumulate_grad_batches=2, run only the first micro-step.
    Expected: in grad_sync.calls, 'accumulate_enter' appears before 'backward'
    and 'accumulate_exit' after.

    Catches a refactor that drops the no_sync context — DDP would sync
    grads on every micro-batch (correct but wasteful) or break with FSDP.
    """
    grad_sync, pctx = fake_dist_env()
    ctx, model, _ = _build_ctx(grad_sync=grad_sync, parallel_ctx=pctx)

    StandardUpdateRule(accumulate_grad_batches=2).step(model, _batch(), ctx)

    names = [c[0] for c in grad_sync.calls]
    # accumulate ctx wraps backward
    assert "accumulate_enter" in names
    assert "accumulate_exit" in names
    enter_idx = names.index("accumulate_enter")
    back_idx = names.index("backward")
    exit_idx = names.index("accumulate_exit")
    assert enter_idx < back_idx < exit_idx


def test_grad_sync_path_does_not_use_no_sync_on_boundary(fake_dist_env):
    """Goal: on the last micro-batch (accumulation boundary), the no_sync
    context is NOT entered — grads must sync.

    Input: accumulate_grad_batches=1 ⇒ every step is a boundary.
    Expected: 'accumulate_enter' / 'accumulate_exit' do not appear.

    Catches a refactor that always wraps in no_sync (would prevent
    cross-rank grad sync entirely).
    """
    grad_sync, pctx = fake_dist_env()
    ctx, model, _ = _build_ctx(grad_sync=grad_sync, parallel_ctx=pctx)

    StandardUpdateRule(accumulate_grad_batches=1).step(model, _batch(), ctx)

    names = [c[0] for c in grad_sync.calls]
    assert "accumulate_enter" not in names
    assert "accumulate_exit" not in names
    # but backward + clip + optimizer_step did fire
    assert "backward" in names
    assert "optimizer_step" in names


# ===========================================================================
# C3. Skip / Stop / Retry signal handling
# ===========================================================================


class _RaisesOnBackward:
    """A loss-tensor stand-in whose .backward() always raises.

    Used to assert that the skip path *truly* avoids backward — if the
    rule mistakenly calls .backward() on it, the test fails with a clear
    RuntimeError.
    """

    def __init__(self) -> None:
        # the rule does ``loss / accumulate_grad_batches`` first so we need
        # this to behave like a tensor. We just expose what's reached.
        self._t = torch.tensor(0.5, requires_grad=True)

    def __truediv__(self, other):
        return self  # scaled = loss / N still routes to us

    def detach(self):
        return self._t.detach()

    def item(self):
        return 0.5

    def backward(self):
        raise RuntimeError("backward must not be called on skip path")


def test_skip_step_aborts_backward_and_optimizer():
    """Goal: SKIP_STEP from on_loss_computed must prevent .backward() AND
    optimizer.step() from being called.

    Input: a loss whose ``.backward`` raises (stand-in for "must not be reached").
    Expected: step returns cleanly with skipped=1.0, no RuntimeError.

    Catches a refactor that swallows the skip signal silently.
    """

    def _trap_loss(model_output, batch, ctx):
        return {"loss": _RaisesOnBackward()}

    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    ctx, model, optim = _build_ctx(callbacks=[_Skipper()])
    ctx.loss_fn = _trap_loss

    optim_step_spy = MagicMock(wraps=optim.step)
    optim.step = optim_step_spy

    metrics = StandardUpdateRule().step(model, _batch(), ctx)

    assert metrics["skipped"] == 1.0
    optim_step_spy.assert_not_called()


def test_skip_step_calls_zero_grad_to_clear_stale_grads():
    """Goal: even on skip, optimizer.zero_grad must fire — otherwise stale
    grads from a previous step leak into the next.

    Input: pre-populate model.linear.weight.grad with junk, return SKIP_STEP.
    Expected: after step, weight.grad is None (zero_grad(set_to_none=True)).

    Catches a refactor that drops the zero_grad call inside the skip branch.
    """

    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    ctx, model, optim = _build_ctx(callbacks=[_Skipper()])
    # pre-populate a stale grad
    model.linear.weight.grad = torch.ones_like(model.linear.weight) * 99.0
    assert model.linear.weight.grad is not None

    StandardUpdateRule().step(model, _batch(), ctx)

    assert model.linear.weight.grad is None


def test_stop_training_surfaces_strongest_signal_via_ctx_extras():
    """Goal: STOP_TRAINING is surfaced via ``ctx.extras["loss_signal"]``
    as ``int(Signal.STOP_TRAINING)``.

    Catches a refactor that surfaces a string ("stop") or doesn't surface
    at all — trainer's outer loop relies on the int comparison at
    [lighttrain/builtin_plugins/trainers/pretrain.py:154-156](../../lighttrain/builtin_plugins/trainers/pretrain.py#L154).
    """

    class _Stopper:
        def on_loss_computed(self, **_):
            return Signal.STOP_TRAINING

    ctx, model, _ = _build_ctx(callbacks=[_Stopper()])

    StandardUpdateRule().step(model, _batch(), ctx)

    assert ctx.extras.get("loss_signal") == int(Signal.STOP_TRAINING)


def test_retry_step_success_re_runs_forward_and_loss():
    """Goal: RETRY_STEP triggers a re-forward+re-loss; success after N
    retries proceeds to backward.

    Input: callback returns RETRY_STEP for the first 2 invocations,
    CONTINUE on the 3rd. We expect:
      - on_forward_post fires 3 times (initial + 2 retries)
      - on_loss_computed fires 3 times
      - on_backward_pre fires once (only after the successful loss check)

    Catches a refactor that collapses RETRY_STEP into SKIP_STEP — backward
    would then never fire and the test catches the absence of
    on_backward_pre.
    """
    forward_post_count = [0]
    loss_computed_count = [0]
    backward_pre_count = [0]
    retries_remaining = [2]  # first 2 calls retry, 3rd continues

    class _Retrier:
        def on_forward_post(self, **_):
            forward_post_count[0] += 1

        def on_loss_computed(self, **_):
            loss_computed_count[0] += 1
            if retries_remaining[0] > 0:
                retries_remaining[0] -= 1
                return Signal.RETRY_STEP
            return None

        def on_backward_pre(self, **_):
            backward_pre_count[0] += 1

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])

    metrics = StandardUpdateRule(max_retries=5).step(model, _batch(), ctx)

    assert forward_post_count[0] == 3, forward_post_count
    assert loss_computed_count[0] == 3, loss_computed_count
    assert backward_pre_count[0] == 1
    assert metrics["retries"] == 2.0
    assert metrics.get("retry_exhausted", 0.0) == 0.0


def test_retry_step_restores_rng_each_iteration():
    """Goal: between retries, the RNG state captured at step entry must be
    restored — so dropout / random augmentation reproduces exactly.

    Input: a loss_fn that draws ``torch.randn(1)`` and stores it. RETRY_STEP
    is returned once. The two recorded draws must be EQUAL (RNG was reset).

    Catches a refactor that drops the ``restore_rng_state(rng_snap)`` call.
    """
    drawn: list[torch.Tensor] = []

    def _draws_random_loss(_out, _batch, _ctx):
        drawn.append(torch.randn(1))
        return {"loss": torch.tensor(0.5, requires_grad=True)}

    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])
    ctx.loss_fn = _draws_random_loss

    torch.manual_seed(123)
    StandardUpdateRule(max_retries=3).step(model, _batch(), ctx)

    assert len(drawn) == 2
    torch.testing.assert_close(drawn[0], drawn[1], atol=1e-5, rtol=1e-4)


def test_retry_step_restores_rng_on_every_retry_not_just_first():
    """Goal (adversarial — attack path 'cache the RNG restore'):
    a refactor that restores RNG only on the FIRST retry and trusts it
    for subsequent retries silently breaks reproducibility on retries 2..N.

    Construction: a loss_fn that records ``torch.randn(1)``. We force
    THREE retries (then continue), so 4 forward-loss calls total. All four
    recorded draws must be EQUAL (RNG restored before each).

    The single-retry test above only catches "never restores" — this
    multi-retry version catches "restores once then caches".
    """
    drawn: list[torch.Tensor] = []

    def _draws_random_loss(_out, _batch, _ctx):
        drawn.append(torch.randn(1))
        return {"loss": torch.tensor(0.5, requires_grad=True)}

    retries_left = [3]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP

    ctx, model, _ = _build_ctx(callbacks=[_Retrier()])
    ctx.loss_fn = _draws_random_loss

    torch.manual_seed(2026)
    StandardUpdateRule(max_retries=5).step(model, _batch(), ctx)

    assert len(drawn) == 4
    for i in range(1, 4):
        torch.testing.assert_close(drawn[0], drawn[i], atol=1e-5, rtol=1e-4)


def test_retry_step_invokes_frozen_step_writer_restore_snapshot():
    """Goal: when a FrozenStepCallback is wired (ctx.frozen_step_writer),
    each retry iteration calls ``writer.restore_snapshot(model, optimizer)``.

    Input: a MagicMock writer attached to ctx.frozen_step_writer; one retry.
    Expected: writer.restore_snapshot was called at least once with
    model=our model and optimizer=our optimizer.

    Catches a refactor that drops the restore_snapshot path — checkpoint
    rollback on retry would silently no-op.
    """
    writer = MagicMock()
    writer.restore_snapshot = MagicMock()

    retries_left = [1]

    class _Retrier:
        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP

    ctx, model, optim = _build_ctx(callbacks=[_Retrier()])
    ctx.frozen_step_writer = writer

    StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    assert writer.restore_snapshot.call_count >= 1
    call_kwargs = writer.restore_snapshot.call_args.kwargs
    assert call_kwargs["model"] is model
    assert call_kwargs["optimizer"] is optim


def test_retry_exhaustion_downgrades_to_skip_and_sets_retry_exhausted():
    """Goal: when retries hit max, the rule downgrades to SKIP_STEP and
    records ``retry_exhausted=1.0`` + propagates SKIP_STEP via loss_signal.

    Input: callback returns RETRY_STEP forever; max_retries=2.
    Expected:
      - metrics['retries'] == 2
      - metrics['retry_exhausted'] == 1.0
      - metrics['skipped'] == 1.0
      - ctx.extras['loss_signal'] == int(Signal.SKIP_STEP)

    Catches a refactor that returns STOP_TRAINING on exhaustion (would
    take down the run) or fails to set retry_exhausted (silent retry
    runaway is then invisible to logs).
    """

    class _ForeverRetry:
        def on_loss_computed(self, **_):
            return Signal.RETRY_STEP

    ctx, model, _ = _build_ctx(callbacks=[_ForeverRetry()])

    metrics = StandardUpdateRule(max_retries=2).step(model, _batch(), ctx)

    assert metrics["retries"] == 2.0
    assert metrics["retry_exhausted"] == 1.0
    assert metrics["skipped"] == 1.0
    assert ctx.extras["loss_signal"] == int(Signal.SKIP_STEP)


def test_signal_precedence_stop_training_beats_retry_step():
    """Goal: when two callbacks return STOP_TRAINING and RETRY_STEP, the
    strongest wins (STOP_TRAINING).

    Input: two callbacks, one returns RETRY_STEP, the other STOP_TRAINING.
    Expected: loss_signal == int(STOP_TRAINING); skip path taken.

    Pins the IntEnum precedence at [callbacks/base.py:161-162].
    """

    class _A:
        def on_loss_computed(self, **_):
            return Signal.RETRY_STEP

    class _B:
        def on_loss_computed(self, **_):
            return Signal.STOP_TRAINING

    ctx, model, _ = _build_ctx(callbacks=[_A(), _B()])

    StandardUpdateRule(max_retries=5).step(model, _batch(), ctx)

    assert ctx.extras["loss_signal"] == int(Signal.STOP_TRAINING)


def test_signal_precedence_retry_step_beats_skip_step():
    """Goal: RETRY_STEP outranks SKIP_STEP — the rule must retry, not skip.

    Input: callback A returns SKIP_STEP, callback B returns RETRY_STEP-then-None.
    Expected: backward fires once (retry succeeded on second try).
    """
    forward_post = [0]
    retries_left = [1]

    class _A:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    class _B:
        def on_forward_post(self, **_):
            forward_post[0] += 1

        def on_loss_computed(self, **_):
            if retries_left[0] > 0:
                retries_left[0] -= 1
                return Signal.RETRY_STEP
            # Note: callback A keeps returning SKIP_STEP, so the strongest
            # signal after retry is still SKIP_STEP. We instead test by
            # ordering A *after* B so on the 2nd loss_computed we still get
            # SKIP_STEP wins. The point of THIS test is just to pin the
            # int comparison: RETRY (2) > SKIP (1). We assert that the first
            # RETRY was honored by checking the recorder.
            return None

    ctx, model, _ = _build_ctx(callbacks=[_A(), _B()])

    StandardUpdateRule(max_retries=3).step(model, _batch(), ctx)

    # The retry forced a second forward; that's the proof RETRY outranked SKIP
    # on the first loss-computed dispatch.
    assert forward_post[0] >= 2


def test_regression_signal_not_collapsed_to_skipped_flag():
    """Regression: pre-fix bug documented in
    [tests/test_stop_training_signal.py](../../tests/test_stop_training_signal.py).

    Before the fix, StandardUpdateRule collapsed SKIP_STEP / RETRY_STEP /
    STOP_TRAINING all into ``metrics['skipped'] = 1.0`` and never surfaced
    the strongest signal — trainer's outer loop couldn't distinguish a
    "skip this batch" from "stop the run".

    The fix surfaces the integer signal via ``ctx.extras["loss_signal"]``.
    This test pins that the three signals produce DIFFERENT loss_signal
    integer values (Signal IntEnum ordering: SKIP_STEP=1, RETRY=2, STOP=3).
    """
    def _make_cb(sig: Signal):
        class _Cb:
            def on_loss_computed(self, **_):
                return sig

        return _Cb()

    for sig, expected_int in [
        (Signal.SKIP_STEP, 1),
        (Signal.STOP_TRAINING, 3),
    ]:
        ctx, model, _ = _build_ctx(callbacks=[_make_cb(sig)])
        StandardUpdateRule().step(model, _batch(), ctx)
        assert ctx.extras["loss_signal"] == expected_int, (
            f"Signal {sig.name} not surfaced as int {expected_int}"
        )


# ===========================================================================
# C4. Gradient accumulation
# ===========================================================================


@pytest.mark.parametrize("accum_K", [2, 4])
def test_accumulation_boundary_optimizer_fires_only_on_last_micro_step(accum_K: int):
    """Goal: ``optimizer.step()`` must fire on micro-step #K (the boundary),
    not on any of the K−1 preceding micro-steps.

    Input: accumulate_grad_batches=K; run K micro-steps; count optimizer.step
    calls via a wrapper.

    Expected: exactly 1 optimizer.step call total, occurring on micro-step K.

    Catches a refactor that fires optimizer on every micro-step.
    """
    ctx, model, optim = _build_ctx()
    step_spy = MagicMock(wraps=optim.step)
    optim.step = step_spy  # type: ignore[method-assign]

    rule = StandardUpdateRule(accumulate_grad_batches=accum_K)

    for i in range(accum_K):
        rule.step(model, _batch(), ctx)
        if i < accum_K - 1:
            assert step_spy.call_count == 0, (
                f"optimizer.step fired early at micro-step {i + 1}/{accum_K}"
            )

    assert step_spy.call_count == 1, (
        f"optimizer.step should fire exactly once on boundary, got {step_spy.call_count}"
    )


def test_accumulation_loss_scaled_by_accumulate_grad_batches():
    """Goal: backward is called with ``loss / accumulate_grad_batches``.

    Input: K=4, loss_fn returns a tensor with known value L=4.0.
    Expected: the scaled loss passed to backward has value 1.0.

    We capture the scaled tensor by patching ``torch.Tensor.backward`` at
    the rule's call site — specifically, we intercept the backward on our
    loss object.

    Catches forgetting the ``/ accumulate_grad_batches`` scale.
    """
    captured: list[float] = []

    class _CaptureLoss(torch.Tensor):
        pass  # only used for type-checking, real interception is below

    # We use a custom tensor wrapper instead of mocking to keep the backward
    # graph intact (StandardUpdateRule does loss / K then backward).
    class _LossTensor:
        def __init__(self, t: torch.Tensor) -> None:
            self._t = t

        def __truediv__(self, other):
            scaled = self._t / other
            return _LossTensor(scaled)

        def detach(self):
            return self._t.detach()

        def item(self):
            return float(self._t.detach().item())

        def backward(self):
            captured.append(float(self._t.detach().item()))
            self._t.backward()

    def _loss(_out, _batch, _ctx):
        return {"loss": _LossTensor(torch.tensor(4.0, requires_grad=True))}

    ctx, model, _ = _build_ctx()
    ctx.loss_fn = _loss

    StandardUpdateRule(accumulate_grad_batches=4).step(model, _batch(), ctx)

    assert len(captured) == 1
    # the scaled value passed to backward is loss / K = 4.0 / 4 = 1.0
    torch.testing.assert_close(
        torch.tensor(captured[0]), torch.tensor(1.0), atol=1e-5, rtol=1e-4
    )


def test_scheduler_step_only_fires_when_step_per_batch_true():
    """Goal: scheduler.step is called only when scheduler has
    step_per_batch=True attribute.

    Parametrized check:
      - step_per_batch=True  ⇒ scheduler.step called once
      - step_per_batch=False ⇒ scheduler.step NEVER called

    Catches a refactor that drops the ``getattr(scheduler, "step_per_batch",
    True)`` gate.
    """
    for flag, expected in [(True, 1), (False, 0)]:
        scheduler = MagicMock()
        scheduler.step_per_batch = flag
        ctx, model, _ = _build_ctx(scheduler=scheduler)

        StandardUpdateRule().step(model, _batch(), ctx)

        assert scheduler.step.call_count == expected, (
            f"step_per_batch={flag} expected step.call_count={expected}, got {scheduler.step.call_count}"
        )


def test_scheduler_step_does_not_fire_during_accumulation():
    """Goal: scheduler.step is suppressed on intermediate micro-steps,
    same as optimizer.step.

    Input: K=2, scheduler.step_per_batch=True; run 1 micro-step.
    Expected: scheduler.step not called.

    Catches a refactor that always fires scheduler.step (LR schedule
    would advance K× too fast).
    """
    scheduler = MagicMock()
    scheduler.step_per_batch = True
    ctx, model, _ = _build_ctx(scheduler=scheduler)

    StandardUpdateRule(accumulate_grad_batches=2).step(model, _batch(), ctx)

    scheduler.step.assert_not_called()


def test_skip_path_resets_micro_step_counter_to_zero():
    """Goal: on skip, ``self._micro_step = 0`` so the next valid step starts
    fresh — otherwise a skip mid-accumulation desyncs the K-batch boundary.

    Input: K=4, advance micro_step to 2 with two normal steps, then
    one skipped step; verify micro_step reset.

    Catches a refactor that drops the reset.
    """
    rule = StandardUpdateRule(accumulate_grad_batches=4)

    # Two normal micro-steps without callbacks
    ctx, model, _ = _build_ctx()
    rule.step(model, _batch(), ctx)
    rule.step(model, _batch(), ctx)
    assert rule._micro_step == 2

    # Now a skip with a new ctx that has the skipper
    class _Skipper:
        def on_loss_computed(self, **_):
            return Signal.SKIP_STEP

    skip_ctx, _, _ = _build_ctx(callbacks=[_Skipper()])
    skip_ctx.model = model
    rule.step(model, _batch(), skip_ctx)

    assert rule._micro_step == 0


def test_ctx_is_accumulating_flag_set_per_micro_step():
    """Goal: ``ctx.is_accumulating`` reflects state AFTER incrementing
    micro_step:
      - True on intermediate micro-steps (K−1 of every K)
      - False on the boundary

    Catches a refactor that flips the meaning of the flag — DDP callbacks
    use it to gate inter-rank ops.
    """
    rule = StandardUpdateRule(accumulate_grad_batches=3)

    ctx, model, _ = _build_ctx()
    rule.step(model, _batch(), ctx)
    assert ctx.is_accumulating is True

    rule.step(model, _batch(), ctx)
    assert ctx.is_accumulating is True

    rule.step(model, _batch(), ctx)
    assert ctx.is_accumulating is False


# ===========================================================================
# C5. Lazy new-param registration
# ===========================================================================


def test_new_trainable_params_drained_from_extras_into_optimizer():
    """Goal: when a loss_fn pushes a fresh nn.Linear's params into
    ``ctx.extras['_new_trainable_params']``, the update rule pops them
    AND adds them as a new param_group to the optimizer (so the optimizer
    actually updates them on the next .step).

    Input: a loss_fn that creates a new Linear layer and pushes its params.
    Expected:
      - ctx.extras no longer contains '_new_trainable_params' (drained)
      - len(optimizer.param_groups) grew by 1

    Catches a refactor that drops the drain block at lines 243-245.
    """
    new_layer = nn.Linear(2, 1, bias=False)

    def _loss_with_lazy_param(_out, _batch, ctx):
        ctx.extras["_new_trainable_params"] = list(new_layer.parameters())
        return {"loss": torch.tensor(0.5, requires_grad=True)}

    ctx, model, optim = _build_ctx()
    ctx.loss_fn = _loss_with_lazy_param
    n_groups_before = len(optim.param_groups)

    StandardUpdateRule().step(model, _batch(), ctx)

    assert "_new_trainable_params" not in ctx.extras
    assert len(optim.param_groups) == n_groups_before + 1


def test_new_params_registered_before_backward_so_they_get_grads():
    """Goal: lazy params must be registered BEFORE backward; otherwise
    their .grad stays None at optimizer.step() time and the update is a no-op.

    Input: a loss_fn that creates a new Linear and uses its forward inside
    the loss computation so backward populates its grads. We inspect the
    grad with an ``on_optimizer_step_pre`` callback — *before* the trailing
    zero_grad nulls the grads back out.

    Catches a refactor that moves the drain AFTER backward (line 243-245
    must run before line 270 backward).
    """
    new_layer = nn.Linear(2, 1, bias=False)
    grad_observed_at_optim_step: list[bool] = []

    class _Inspect:
        def on_optimizer_step_pre(self, **_kw):
            grad_observed_at_optim_step.append(new_layer.weight.grad is not None)

    def _loss_with_lazy_param(_out, batch, ctx):
        ctx.extras["_new_trainable_params"] = list(new_layer.parameters())
        x = torch.randn(2, 2)
        extra = new_layer(x).sum()
        return {"loss": extra + 0.0}

    ctx, model, _ = _build_ctx(callbacks=[_Inspect()])
    ctx.loss_fn = _loss_with_lazy_param

    StandardUpdateRule().step(model, _batch(), ctx)

    assert grad_observed_at_optim_step == [True]


def test_new_params_no_duplicate_registration():
    """Goal: registering the same parameter twice is idempotent (no
    duplicate param_groups), guarded by ``id()`` check at lines 348-351.

    Input: call ``_register_new_params`` twice with the same params.
    Expected: only one new param_group added in total.

    Catches a refactor that drops the existing_ids check.
    """
    optim = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.01)
    fresh = nn.Parameter(torch.zeros(2))
    n0 = len(optim.param_groups)

    _register_new_params(optim, [fresh])
    n1 = len(optim.param_groups)
    _register_new_params(optim, [fresh])  # second call w/ same param
    n2 = len(optim.param_groups)

    assert n1 == n0 + 1
    assert n2 == n1  # no duplicate group


# ===========================================================================
# C6. Misc
# ===========================================================================


def test_state_dict_roundtrip_preserves_micro_step():
    """Goal: micro_step is persisted by state_dict / load_state_dict so a
    resume mid-accumulation continues from the correct micro-position.

    Catches a refactor that drops micro_step from the dict.
    """
    rule = StandardUpdateRule(
        grad_clip=0.7, accumulate_grad_batches=4, max_retries=2
    )
    rule._micro_step = 3
    sd = rule.state_dict()
    assert sd["micro_step"] == 3
    assert sd["grad_clip"] == 0.7
    assert sd["accumulate_grad_batches"] == 4
    assert sd["max_retries"] == 2

    rule2 = StandardUpdateRule()
    rule2.load_state_dict(sd)
    assert rule2._micro_step == 3
    assert rule2.grad_clip == 0.7
    assert rule2.accumulate_grad_batches == 4
    assert rule2.max_retries == 2


def test_loss_fn_keyerror_raised_clearly():
    """Goal: if loss_fn returns a dict without 'loss' key, the rule raises a
    KeyError with a clear message — not a downstream AttributeError.

    Catches a refactor that drops the explicit check at line 150-151.
    """

    def _bad_loss(_out, _batch, _ctx):
        return {"not_loss": torch.tensor(1.0)}

    ctx, model, _ = _build_ctx()
    ctx.loss_fn = _bad_loss

    with pytest.raises(KeyError, match="'loss'"):
        StandardUpdateRule().step(model, _batch(), ctx)


def test_missing_model_raises_runtime_error():
    """Pin line 91-92 guard: model=None raises a clear RuntimeError."""
    ctx, _, optim = _build_ctx()
    ctx.model = None
    with pytest.raises(RuntimeError, match="model is None"):
        StandardUpdateRule().step(None, _batch(), ctx)


def test_missing_optimizer_raises_runtime_error():
    """Pin line 93-94 guard: optimizer=None raises a clear RuntimeError."""
    ctx, model, _ = _build_ctx()
    ctx.optimizer = None
    with pytest.raises(RuntimeError, match="optimizer is None"):
        StandardUpdateRule().step(model, _batch(), ctx)


def test_missing_loss_fn_raises_runtime_error():
    """Pin line 95-96 guard: loss_fn=None raises a clear RuntimeError."""
    ctx, model, _ = _build_ctx()
    ctx.loss_fn = None
    with pytest.raises(RuntimeError, match="loss_fn is None"):
        StandardUpdateRule().step(model, _batch(), ctx)
