"""RLUpdateRule — backward/clip/step rule for RL trainers (GRPO/PPO/Preference).

Unlike StandardUpdateRule, this rule does NOT call ``model(**batch)`` in forward.
The trainer is responsible for:

1. Running the model-specific forward pass to compute RL quantities
   (log_probs_new, advantages, chosen_logps, etc.)
2. Populating ``ctx.extras`` with the pre-computed tensors
3. Setting ``ctx.loss_fn`` to the RL loss function
4. Calling ``self._rl_rule.step(model, batch, ctx)``

This rule then calls ``ctx.loss_fn(dummy, batch, ctx)`` to obtain the loss
and runs the full backward / clip / optimizer / callback sequence,
mirroring the three-path logic of StandardUpdateRule:
  - grad_sync path   (DDP/FSDP/ZeRO)
  - accelerator path (HuggingFace Accelerate / GradScaler)
  - bare path        (single-GPU, no AMP)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..callbacks.base import Signal
from ..protocols import LossContext, ModelOutput
from ..registry import register
from ._primitives import MicroState, _current_lr, apply_update


@register("update_rule", "rl")
class RLUpdateRule:
    """Backward/clip/step rule shared by GRPO, PPO, and Preference trainers."""

    def __init__(self, *, grad_clip: float = 1.0) -> None:
        self.grad_clip = float(grad_clip)
        # RL does not accumulate gradients; a private cursor lets the backward
        # half be the shared, stateless ``apply_update`` primitive.
        self._micro = MicroState()

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"grad_clip": self.grad_clip}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.grad_clip = float(sd.get("grad_clip", self.grad_clip))

    def step(
        self,
        model: Any,
        batch: Mapping[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        bus = ctx.bus
        optimizer = ctx.optimizer
        scheduler = ctx.scheduler
        loss_fn = ctx.loss_fn
        accelerator = ctx.accelerator

        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        # Loss — no model forward; RL data already in ctx.extras / batch
        loss_ctx = LossContext(step=ctx.step, epoch=ctx.epoch, extras=ctx.extras)
        loss_dict = loss_fn(ModelOutput(outputs={}), batch, loss_ctx)
        if "loss" not in loss_dict:
            raise KeyError("LossFn must return a dict containing 'loss'.")
        loss: torch.Tensor = loss_dict["loss"]

        ctx.metrics.update({
            k: float(v.detach()) if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        })

        # SKIP_STEP / STOP_TRAINING
        skip = False
        if bus is not None:
            sig = bus.dispatch(
                "on_loss_computed",
                step=ctx.step,
                loss=loss,
                batch=batch,
                model=model,
                metrics=ctx.metrics,
            )
            if sig in (Signal.SKIP_STEP, Signal.STOP_TRAINING):
                skip = True
                ctx.extras["loss_signal"] = int(sig)

        if skip:
            optimizer.zero_grad(set_to_none=True)
            ctx.metrics.setdefault("grad_norm", 0.0)
            ctx.metrics["lr"] = _current_lr(optimizer)
            ctx.metrics["skipped"] = 1.0
            if bus is not None:
                bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics,
                             batch=batch, model=model)
            return dict(ctx.metrics)

        # Backward / clip / optimizer / scheduler — the shared backward half.
        # RL has no gradient accumulation, so accumulate_grad_batches=1 with a
        # private MicroState makes this numerically identical to the previous
        # inline three-path implementation. grad_clip / grad_sync / accelerator
        # are honoured exactly as before (read from ctx inside apply_update).
        apply_update(
            loss=loss,
            model=model,
            optimizer=optimizer,
            ctx=ctx,
            micro_state=self._micro,
            scheduler=scheduler,
            accelerator=accelerator,
            grad_clip=self.grad_clip,
            accumulate_grad_batches=1,
            bus=bus,
        )

        if bus is not None:
            bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics,
                         batch=batch, model=model)
        return dict(ctx.metrics)


__all__ = ["RLUpdateRule"]
