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

import logging
from collections.abc import Mapping
from typing import Any

import torch

from lighttrain.callbacks.base import Signal
from lighttrain.engine.update_rules._primitives import (  # noqa: F401  (_register_new_params re-exported for back-compat)
    MicroState,
    _current_lr,
    _register_new_params,
    apply_update,
    make_autocast,
)
from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register
from lighttrain.utils.log import warn_once
from lighttrain.utils.seed import restore_rng_state, rng_state

_log = logging.getLogger(__name__)


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
        # Gradient-accumulation cursor lives in a caller-shareable holder so the
        # backward half can be the stateless ``apply_update`` primitive.
        self._micro = MicroState()
        # Keys of per-step failure warnings already emitted, so a persistently
        # failing hot-loop site (RNG snapshot/restore) warns once, not per step.
        self._warned: set[str] = set()

    @property
    def _micro_step(self) -> int:
        return self._micro.micro_step

    @_micro_step.setter
    def _micro_step(self, value: int) -> None:
        self._micro.micro_step = int(value)

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
            warn_once(
                self._warned,
                "rng_snapshot_failed",
                _log,
                "update_rule.step: RNG snapshot failed; RETRY_STEP replay will skip RNG restore",
                exc_info=True,
            )
            rng_snap = None

        # Forward.
        if bus is not None:
            bus.dispatch("on_forward_pre", step=ctx.step, batch=batch, model=model)

        # AMP autocast wraps forward + loss when an Accelerator is wired.
        # Backward stays outside autocast (per HF Accelerate convention).
        # A fresh autocast CM is built per forward inside ``_run_forward_and_loss``
        # (RETRY_STEP replays forward, and ``accelerator.autocast()`` is single-use).

        # Publish the model into ``ctx.extras`` so loss fns that need to
        # attach learned submodules (e.g. HiddenStatesMSELoss(project=True)
        # lazy ``nn.Linear`` projection) can reach it without breaking the
        # LossFn(model_output, batch, ctx) signature.
        ctx.extras["model"] = model

        def _run_forward_and_loss() -> tuple[Any, Any, dict[str, Any]]:
            with make_autocast(accelerator):
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
                        warn_once(
                            self._warned,
                            "retry_snapshot_restore_failed",
                            _log,
                            "update_rule.step: RETRY_STEP snapshot restore failed; replaying on unrestored params",
                            exc_info=True,
                        )
                if rng_snap is not None:
                    try:
                        restore_rng_state(rng_snap)
                    except Exception:  # noqa: BLE001
                        warn_once(
                            self._warned,
                            "retry_rng_restore_failed",
                            _log,
                            "update_rule.step: RETRY_STEP RNG restore failed; replaying with current RNG state",
                            exc_info=True,
                        )
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

        # Backward / clip / optimizer / scheduler — the re-usable backward half.
        # Loss fns push fresh trainable params into
        # ``ctx.extras["_new_trainable_params"]`` (drained inside apply_update).
        apply_update(
            loss=loss,
            model=model,
            optimizer=optimizer,
            ctx=ctx,
            micro_state=self._micro,
            scheduler=scheduler,
            accelerator=accelerator,
            grad_clip=self.grad_clip,
            accumulate_grad_batches=self.accumulate_grad_batches,
            bus=bus,
        )

        if bus is not None:
            bus.dispatch(
                "on_step_end",
                step=ctx.step,
                metrics=ctx.metrics,
                batch=batch,
                model=model,
            )
        return dict(ctx.metrics)


# ``_current_lr`` / ``_register_new_params`` now live in ._primitives and are
# imported above; re-exported here for back-compat (rl.py imports _current_lr
# from .standard).
__all__ = ["StandardUpdateRule"]
