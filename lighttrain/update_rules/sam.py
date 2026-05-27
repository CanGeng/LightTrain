"""SAMUpdateRule — Sharpness-Aware Minimisation.

Implements the two-pass SAM optimiser (Foret et al., 2021).

Algorithm per step:
    1.  Forward + backward → compute gradient g at θ.
    2.  Compute normalised perturbation: ê = ρ · g / (||g|| + ε_norm)
    3.  Perturb weights: θ_hat = θ + ê
    4.  Forward + backward at θ_hat → compute gradient g_hat.
    5.  Restore weights: θ = θ_hat − ê
    6.  ``optimizer.step()`` with g_hat.

SAM finds flat minima (low sharpness) and typically improves generalisation
at the cost of exactly 2× forward/backward passes per step.

Compatible with gradient accumulation (inner loop respects accumulate_grad_batches);
SAM perturbation is only applied at the *final* accumulation step.

Registered as ``@register("update_rule", "sam")``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch

from ..protocols import LossContext, ModelOutput
from ..registry import register


def _to_metric(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _current_lr(optimizer: Any) -> float:
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if not groups:
        return 0.0
    return float(groups[0].get("lr", 0.0))


@register("update_rule", "sam")
class SAMUpdateRule:
    """Sharpness-Aware Minimisation update rule.

    Args:
        rho:                  Perturbation radius.
        eps_norm:             Small constant for numerical stability in ê.
        accumulate_grad_batches: Gradient accumulation steps.
        grad_clip:            Max gradient norm (0 = no clipping).
    """

    def __init__(
        self,
        rho: float = 0.05,
        eps_norm: float = 1e-12,
        accumulate_grad_batches: int = 1,
        grad_clip: float = 1.0,
    ) -> None:
        self.rho = float(rho)
        self.eps_norm = float(eps_norm)
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.grad_clip = float(grad_clip)
        self._micro_step = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "grad_clip": self.grad_clip,
            "micro_step": self._micro_step,
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.rho = float(sd.get("rho", self.rho))
        self.accumulate_grad_batches = int(sd.get("accumulate_grad_batches", self.accumulate_grad_batches))
        self.grad_clip = float(sd.get("grad_clip", self.grad_clip))
        self._micro_step = int(sd.get("micro_step", 0))

    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------

    def _compute_perturbation(self, model: Any) -> list[torch.Tensor]:
        """Compute ê and store per-parameter perturbation tensors."""
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return []
        grad_norm = torch.stack([g.norm(2) for g in grads]).norm(2)
        scale = self.rho / (grad_norm + self.eps_norm)
        perturbations = []
        for p in model.parameters():
            if p.grad is not None:
                e = p.grad * scale
                p.data.add_(e)
                perturbations.append(e)
            else:
                perturbations.append(None)
        return perturbations

    def _restore(self, model: Any, perturbations: list[torch.Tensor]) -> None:
        it = iter(perturbations)
        for p in model.parameters():
            e = next(it)
            if e is not None:
                p.data.sub_(e)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

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

        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        autocast_ctx = (
            accelerator.autocast()
            if accelerator is not None and hasattr(accelerator, "autocast")
            else nullcontext()
        )

        ctx.extras["model"] = model

        def _forward_loss(b: Any) -> tuple[Any, torch.Tensor, dict]:
            with autocast_ctx:
                _out = model(**b)
                if not isinstance(_out, ModelOutput):
                    _out = ModelOutput(
                        outputs=dict(_out) if isinstance(_out, Mapping) else {"logits": _out}
                    )
                _lctx = LossContext(step=ctx.step, epoch=ctx.epoch, metrics=ctx.metrics, extras=ctx.extras)
                _ld = loss_fn(_out, b, _lctx)
            if "loss" not in _ld:
                raise KeyError("LossFn must return a dict containing 'loss'.")
            return _out, _ld["loss"], _ld

        # --- Pass 1: forward + backward at θ ---------------------------
        outputs, loss, loss_dict = _forward_loss(batch)

        if bus is not None:
            bus.dispatch("on_forward_post", step=ctx.step, outputs=outputs, model=model)
            bus.dispatch("on_loss_computed", step=ctx.step, loss=loss, outputs=outputs,
                         batch=batch, model=model, metrics=ctx.metrics)
            bus.dispatch("on_backward_pre", step=ctx.step, loss=loss)

        scaled = loss / self.accumulate_grad_batches
        if accelerator is not None and hasattr(accelerator, "backward"):
            accelerator.backward(scaled)
        else:
            scaled.backward()

        self._micro_step += 1
        accumulating = (self._micro_step % self.accumulate_grad_batches) != 0

        if bus is not None:
            bus.dispatch("on_backward_post", step=ctx.step, loss=loss)

        if accumulating:
            # Not yet at accumulation boundary — skip SAM perturbation
            ctx.metrics["loss"] = _to_metric(loss)
            ctx.metrics["lr"] = _current_lr(optimizer)
            ctx.metrics.setdefault("grad_norm", 0.0)
            ctx.metrics["skipped"] = 0.0
            if bus is not None:
                bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)
            return dict(ctx.metrics)

        # --- SAM perturbation + Pass 2 ---------------------------------
        perturbations = self._compute_perturbation(model)
        optimizer.zero_grad(set_to_none=True)

        _, loss2, _ = _forward_loss(batch)
        scaled2 = loss2 / self.accumulate_grad_batches
        if accelerator is not None and hasattr(accelerator, "backward"):
            accelerator.backward(scaled2)
        else:
            scaled2.backward()

        # Restore θ before optimizer step
        self._restore(model, perturbations)

        # --- Grad clip + optimizer step --------------------------------
        grad_norm = 0.0
        if self.grad_clip and self.grad_clip > 0:
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

        optimizer.step()

        if bus is not None:
            bus.dispatch("on_optimizer_step_post", step=ctx.step, model=model)

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and getattr(scheduler, "step_per_batch", True):
            scheduler.step()
            if bus is not None:
                bus.dispatch("on_scheduler_step", step=ctx.step)

        ctx.metrics["loss"] = _to_metric(loss)
        ctx.metrics["grad_norm"] = grad_norm
        ctx.metrics["lr"] = _current_lr(optimizer)
        ctx.metrics["skipped"] = 0.0
        for k, v in loss_dict.items():
            if k != "loss":
                ctx.metrics[k] = _to_metric(v)

        if bus is not None:
            bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)

        return dict(ctx.metrics)


__all__ = ["SAMUpdateRule"]
