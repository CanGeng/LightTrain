"""StandardUpdateRule — the canonical SGD/Adam-style step.

Sequence::

    on_step_begin → forward → on_forward_post → loss → on_loss_computed
       → backward → on_backward_post → (clip+step+zero_grad) → on_optimizer_step_post
       → scheduler.step → on_scheduler_step → on_step_end

Returns a metrics dict the trainer / callbacks can read. Honors callback
``Signal.SKIP_STEP`` returned from ``on_loss_computed`` (e.g. NaNSkip)
by aborting the backward path before parameters are touched.

**RETRY_STEP true replay**: when ``on_loss_computed`` returns
``Signal.RETRY_STEP`` we re-run forward+loss on the *same batch* with the
RNG state captured at step entry. If a ``FrozenStepCallback`` is attached
(``ctx.frozen_step_writer``) its on-step snapshot is also used to restore
model parameters; without that snapshot the retry still runs but only on
RNG (parameters are unchanged anyway because RETRY fires before backward).
The retry is bounded by ``max_retries`` to prevent infinite loops;
exhaustion falls back to a soft SKIP_STEP and writes
``ctx.metrics['retry_exhausted'] = 1``.
"""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from typing import Any, Mapping

import torch

from ..callbacks.base import Signal
from ..protocols import LossContext, ModelOutput
from ..registry import register
from ..utils.seed import restore_rng_state, rng_state


def _to_metric(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@register("update_rule", "standard")
class StandardUpdateRule:
    def __init__(
        self,
        *,
        grad_clip: float = 1.0,
        accumulate_grad_batches: int = 1,
        max_retries: int = 3,
    ) -> None:
        self.grad_clip = float(grad_clip)
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.max_retries = max(0, int(max_retries))
        self._micro_step = 0

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "grad_clip": self.grad_clip,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "max_retries": self.max_retries,
            "micro_step": self._micro_step,
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.grad_clip = float(sd.get("grad_clip", self.grad_clip))
        self.accumulate_grad_batches = int(
            sd.get("accumulate_grad_batches", self.accumulate_grad_batches)
        )
        self.max_retries = int(sd.get("max_retries", self.max_retries))
        self._micro_step = int(sd.get("micro_step", 0))

    def step(
        self,
        model: Any,
        batch: Mapping[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        bus = ctx.bus
        accelerator = ctx.accelerator
        loss_fn = ctx.loss_fn
        optimizer = ctx.optimizer
        scheduler = ctx.scheduler

        if model is None:
            raise RuntimeError("StandardUpdateRule.step: ctx.model is None.")
        if optimizer is None:
            raise RuntimeError("StandardUpdateRule.step: ctx.optimizer is None.")
        if loss_fn is None:
            raise RuntimeError("StandardUpdateRule.step: ctx.loss_fn is None.")

        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        # Snapshot RNG at step entry so RETRY_STEP can replay forward+loss with
        # the same RNG-driven side effects (dropout, sampler, augmentation).
        # ``rng_state`` returns a dict; cheap (tens of bytes for CPU + a few KB
        # for CUDA), so we do it on every step instead of opt-in.
        try:
            rng_snap = rng_state()
        except Exception:  # noqa: BLE001
            rng_snap = None

        # Forward.
        if bus is not None:
            bus.dispatch("on_forward_pre", step=ctx.step, batch=batch, model=model)

        # AMP autocast wraps forward + loss when an Accelerator is wired.
        # Backward stays outside autocast (per HF Accelerate convention).
        if accelerator is not None and hasattr(accelerator, "autocast"):
            autocast_ctx = accelerator.autocast()
        else:
            from contextlib import nullcontext

            autocast_ctx = nullcontext()

        # Publish the model into ``ctx.extras`` so loss fns that need to
        # attach learned submodules (e.g. HiddenStatesMSELoss(project=True)
        # lazy ``nn.Linear`` projection) can reach it without breaking the
        # LossFn(model_output, batch, ctx) signature.
        ctx.extras["model"] = model

        def _run_forward_and_loss() -> tuple[Any, Any, dict[str, Any]]:
            with autocast_ctx:
                _outputs = model(**batch)
                if not isinstance(_outputs, ModelOutput):
                    _outputs = ModelOutput(
                        outputs=dict(_outputs)
                        if isinstance(_outputs, Mapping)
                        else {"logits": _outputs}
                    )
                # Share ``extras`` between StepContext and LossContext so loss
                # fns can push side-channel state back (e.g.
                # HiddenStatesMSELoss(project=True) appending freshly-created
                # ``nn.Linear`` parameters to ``_new_trainable_params`` for the
                # update rule to register with the optimizer).
                _loss_ctx = LossContext(
                    step=ctx.step,
                    epoch=ctx.epoch,
                    metrics=ctx.metrics,
                    extras=ctx.extras,
                )
                _loss_dict = loss_fn(_outputs, batch, _loss_ctx)
            if "loss" not in _loss_dict:
                raise KeyError("LossFn must return a dict containing 'loss'.")
            return _outputs, _loss_dict["loss"], _loss_dict

        outputs, loss, loss_dict = _run_forward_and_loss()

        if bus is not None:
            bus.dispatch("on_forward_post", step=ctx.step, outputs=outputs, model=model)

        skip = False
        loss_signal = Signal.CONTINUE
        retry_count = 0
        if bus is not None:
            sig = bus.dispatch(
                "on_loss_computed",
                step=ctx.step,
                loss=loss,
                outputs=outputs,
                batch=batch,
                model=model,
                metrics=ctx.metrics,
            )
            loss_signal = sig
            # RETRY_STEP true replay: re-run forward+loss on the same batch
            # with restored RNG; bounded by ``max_retries``; on exhaustion
            # fall back to a soft skip.
            while sig == Signal.RETRY_STEP and retry_count < self.max_retries:
                retry_count += 1
                # Restore model params from FrozenStepCallback snapshot, if any.
                writer = getattr(ctx, "frozen_step_writer", None)
                if writer is not None and hasattr(writer, "restore_snapshot"):
                    try:
                        writer.restore_snapshot(model=model, optimizer=optimizer)
                    except Exception:  # noqa: BLE001
                        pass
                if rng_snap is not None:
                    try:
                        restore_rng_state(rng_snap)
                    except Exception:  # noqa: BLE001
                        pass
                outputs, loss, loss_dict = _run_forward_and_loss()
                if bus is not None:
                    bus.dispatch(
                        "on_forward_post",
                        step=ctx.step,
                        outputs=outputs,
                        model=model,
                    )
                    sig = bus.dispatch(
                        "on_loss_computed",
                        step=ctx.step,
                        loss=loss,
                        outputs=outputs,
                        batch=batch,
                        model=model,
                        metrics=ctx.metrics,
                    )
                    loss_signal = sig
            if sig == Signal.RETRY_STEP and retry_count >= self.max_retries:
                # Exhausted — degrade to SKIP_STEP.
                loss_signal = Signal.SKIP_STEP
                sig = Signal.SKIP_STEP
                ctx.metrics["retry_exhausted"] = 1.0
            if sig in (Signal.SKIP_STEP, Signal.STOP_TRAINING):
                skip = True
        ctx.metrics["retries"] = float(retry_count)
        # Propagate the strongest non-CONTINUE signal up to the trainer via
        # ``ctx.extras`` so STOP_TRAINING actually stops the loop.
        if loss_signal != Signal.CONTINUE:
            ctx.extras["loss_signal"] = int(loss_signal)

        ctx.metrics["loss"] = _to_metric(loss)
        for k, v in loss_dict.items():
            if k == "loss":
                continue
            if isinstance(v, torch.Tensor):
                ctx.metrics[k] = _to_metric(v)

        if skip:
            optimizer.zero_grad(set_to_none=True)
            self._micro_step = 0
            ctx.metrics.setdefault("grad_norm", 0.0)
            ctx.metrics["lr"] = _current_lr(optimizer)
            ctx.metrics["skipped"] = 1.0
            if bus is not None:
                bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)
            return dict(ctx.metrics)

        # Drain newly-created trainable params (e.g. HiddenStatesMSELoss lazy
        # projections) into the optimizer as a fresh param_group **before**
        # backward, so subsequent ``optimizer.step()`` actually updates them.
        # Loss fns push parameters into ``ctx.extras["_new_trainable_params"]``
        # at create-time; we pop here so it's a one-shot delivery.
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
        # last micro-batch.  After increment: accumulating = True means keep
        # accumulating; False means this was the last micro-step — do optimizer.
        pre_accumulating = (
            (self._micro_step + 1) % self.accumulate_grad_batches
        ) != 0

        scaled = loss / self.accumulate_grad_batches
        accum_ctx = grad_sync.accumulate(model) if (grad_sync and pre_accumulating) else _nullcontext()

        with accum_ctx:
            if grad_sync is not None:
                grad_sync.backward(scaled, model)
            elif accelerator is not None and hasattr(accelerator, "backward"):
                accelerator.backward(scaled)
            else:
                scaled.backward()

        self._micro_step += 1
        accumulating = (self._micro_step % self.accumulate_grad_batches) != 0
        ctx.is_accumulating = accumulating

        if bus is not None:
            bus.dispatch("on_backward_post", step=ctx.step, loss=loss)

        grad_norm = 0.0
        if not accumulating:
            if self.grad_clip and self.grad_clip > 0:
                if grad_sync is not None:
                    grad_norm = grad_sync.clip_grad_norm(model, self.grad_clip, parallel_ctx)
                else:
                    params = [p for p in model.parameters() if p.grad is not None]
                    if params:
                        if accelerator is not None and hasattr(accelerator, "clip_grad_norm_"):
                            gn = accelerator.clip_grad_norm_(params, max_norm=self.grad_clip)
                        else:
                            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip)
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

        if bus is not None:
            bus.dispatch(
                "on_step_end",
                step=ctx.step,
                metrics=ctx.metrics,
                batch=batch,
                model=model,
            )
        return dict(ctx.metrics)


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


__all__ = ["StandardUpdateRule"]
