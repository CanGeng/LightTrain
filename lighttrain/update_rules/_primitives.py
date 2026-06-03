"""Public, re-entrant training primitives shared across update rules.

``apply_update`` is the ``clip / step / scheduler / AMP / accumulation`` backward half
lifted verbatim out of :meth:`StandardUpdateRule.step`. It is factored so that
*any* custom forward (multi-model distillation, RL surrogate losses, a bare
per-layer loop) can reuse the mature backward path — gradient accumulation,
the three-way backward dispatch (grad_sync / accelerator / bare), grad clip,
and the full callback lifecycle — instead of re-implementing ``loss.backward()``.

Re-entrancy: the function holds **no** state. Per-call accumulation position
lives in the caller-owned :class:`MicroState`, so the same primitive can drive
many independent optimizers in one process (e.g. an Axis-C per-layer loop).
"""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from typing import Any

import torch


def make_autocast(accelerator: Any):
    """Return a *fresh* AMP autocast context manager for one forward pass.

    ``accelerator.autocast()`` returns a single-use ``_GeneratorContextManager``
    (re-entering the same object raises ``AttributeError: '_GeneratorContextManager'
    object has no attribute 'args'``). Update rules that run multiple forwards per
    step (RETRY_STEP replay, SAM two-pass, MeZO ±perturbation) must call this each
    forward instead of caching the context object.
    """
    if accelerator is not None and hasattr(accelerator, "autocast"):
        return accelerator.autocast()
    return _nullcontext()


def _current_lr(optimizer: Any) -> float:
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if not groups:
        return 0.0
    return float(groups[0].get("lr", 0.0))


def _register_new_params(optimizer: Any, new_params: Any) -> None:
    """Add fresh trainable params to ``optimizer`` as a new param_group.

    Mirrors the defaults (lr / weight_decay / betas / ...) of the first
    existing param group so the projection layer trains under the same
    hyperparameters as the model proper. Filters duplicates by ``id`` so a
    repeated drain (shouldn't happen, but cheap to guard) is idempotent.
    """
    inner = getattr(optimizer, "optimizer", optimizer)
    if not hasattr(inner, "add_param_group") or not hasattr(inner, "param_groups"):
        return
    existing_ids: set[int] = set()
    for g in inner.param_groups:
        for p in g.get("params", []):
            existing_ids.add(id(p))
    fresh = [p for p in new_params if id(p) not in existing_ids]
    if not fresh:
        return
    if inner.param_groups:
        defaults = {
            k: v for k, v in inner.param_groups[0].items() if k != "params"
        }
    else:
        defaults = {}
    inner.add_param_group({"params": list(fresh), **defaults})


@dataclass
class MicroState:
    """Caller-owned gradient-accumulation cursor passed to :func:`apply_update`.

    Holds the running micro-batch counter so the primitive itself stays
    stateless. Each independent optimizer/loop owns one instance.
    """

    micro_step: int = 0


def apply_update(
    *,
    loss: Any,
    model: Any,
    optimizer: Any,
    ctx: Any,
    micro_state: MicroState,
    scheduler: Any | None = None,
    accelerator: Any | None = None,
    grad_clip: float = 0.0,
    accumulate_grad_batches: int = 1,
    bus: Any | None = None,
) -> float:
    """Run the backward / clip / optimizer / scheduler half of a training step.

    Lifted verbatim from :meth:`StandardUpdateRule.step` so the pretrain path
    is numerically a no-op after the extraction. Writes ``grad_norm`` / ``lr`` /
    ``skipped`` into ``ctx.metrics`` and returns ``grad_norm``. Does **not**
    dispatch ``on_step_end`` — that stays a step-level concern owned by the
    caller (it also fires on the skip path, which never reaches here).

    ``grad_sync`` / ``parallel_ctx`` are read from ``ctx`` exactly as before.
    """
    # Drain newly-created trainable params (e.g. HiddenStatesMSELoss lazy
    # projections) into the optimizer as a fresh param_group **before**
    # backward, so subsequent ``optimizer.step()`` actually updates them.
    _new_params = ctx.extras.pop("_new_trainable_params", None)
    if _new_params:
        _register_new_params(optimizer, _new_params)

    # Backward.
    if bus is not None:
        bus.dispatch("on_backward_pre", step=ctx.step, loss=loss)

    grad_sync = getattr(ctx, "grad_sync", None)
    parallel_ctx = getattr(ctx, "parallel_ctx", None)

    # Determine accumulation state BEFORE incrementing micro_step so we can
    # suppress inter-rank gradient sync (DDP/FSDP no_sync) on all but the
    # last micro-batch.
    pre_accumulating = (
        (micro_state.micro_step + 1) % accumulate_grad_batches
    ) != 0

    scaled = loss / accumulate_grad_batches
    accum_ctx = (
        grad_sync.accumulate(model) if (grad_sync and pre_accumulating) else _nullcontext()
    )

    with accum_ctx:
        if grad_sync is not None:
            grad_sync.backward(scaled, model)
        elif accelerator is not None and hasattr(accelerator, "backward"):
            accelerator.backward(scaled)
        else:
            scaled.backward()

    micro_state.micro_step += 1
    accumulating = (micro_state.micro_step % accumulate_grad_batches) != 0
    ctx.is_accumulating = accumulating

    if bus is not None:
        bus.dispatch("on_backward_post", step=ctx.step, loss=loss)

    grad_norm = 0.0
    if not accumulating:
        if grad_clip and grad_clip > 0:
            if grad_sync is not None:
                grad_norm = grad_sync.clip_grad_norm(model, grad_clip, parallel_ctx)
            else:
                params = [p for p in model.parameters() if p.grad is not None]
                if params:
                    if accelerator is not None and hasattr(accelerator, "clip_grad_norm_"):
                        gn = accelerator.clip_grad_norm_(params, max_norm=grad_clip)
                    else:
                        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                    grad_norm = float(gn)
        if bus is not None:
            bus.dispatch("on_clip_grad", step=ctx.step, grad_norm=grad_norm)
            bus.dispatch("on_optimizer_step_pre", step=ctx.step)

        if grad_sync is not None:
            grad_sync.optimizer_step(optimizer, model)
        else:
            optimizer.step()

        if bus is not None:
            bus.dispatch("on_optimizer_step_post", step=ctx.step, model=model)

        optimizer.zero_grad(set_to_none=True)
        if bus is not None:
            bus.dispatch("on_zero_grad", step=ctx.step)

        if scheduler is not None and getattr(scheduler, "step_per_batch", True):
            scheduler.step()
            if bus is not None:
                bus.dispatch("on_scheduler_step", step=ctx.step)

    ctx.metrics["grad_norm"] = grad_norm
    ctx.metrics["lr"] = _current_lr(optimizer)
    ctx.metrics["skipped"] = 0.0
    return grad_norm


__all__ = ["MicroState", "apply_update", "make_autocast", "_current_lr", "_register_new_params"]
