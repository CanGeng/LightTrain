"""MeZOUpdateRule — Memory-Efficient Zeroth-Order optimisation.

Implements the SPSA-style gradient-free optimiser from Zhang et al., 2023
("Fine-Tuning Language Models with Just Forward Passes").

Algorithm (per step):
    1.  Sample a random seed; draw perturbation z ~ N(0,1) matching all
        trainable parameters (using the seed for reproducibility).
    2.  Perturb θ ← θ + ε·z  ;  compute L₊ = loss(θ + ε·z)
    3.  Restore θ ← θ − ε·z  ;  compute L₋ = loss(θ − ε·z)
    4.  Re-perturb with same seed and apply update:
        θ ← θ − lr · grad_est · z   where grad_est = (L₊ − L₋) / (2ε)

Key properties:
    * No backward pass — ``p.grad`` is always ``None``.
    * Memory cost O(n) for parameters only (no activations saved).
    * Compatible with any loss fn; does NOT require ``ctx.accelerator``.

Registered as ``@register("update_rule", "mezo")``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register
from lighttrain.update_rules._primitives import make_autocast


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


@register("update_rule", "mezo")
class MeZOUpdateRule:
    """Zeroth-order SFT update rule — no backward pass, gradient-free.

    Args:
        eps:           Perturbation scale ε.
        seed_per_step: Use a fresh random seed each step (True) or a fixed
                       seed (False, for debugging).
    """

    def __init__(
        self,
        eps: float = 1e-3,
        seed_per_step: bool = True,
    ) -> None:
        self.eps = float(eps)
        self.seed_per_step = bool(seed_per_step)
        self._step_count = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def setup(self, model: Any, sample: Any) -> None:  # noqa: ARG002
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"eps": self.eps, "seed_per_step": self.seed_per_step, "step_count": self._step_count}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.eps = float(sd.get("eps", self.eps))
        self.seed_per_step = bool(sd.get("seed_per_step", self.seed_per_step))
        self._step_count = int(sd.get("step_count", 0))

    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------

    def _seed(self) -> int:
        if self.seed_per_step:
            return int(torch.randint(0, 2**31, (1,)).item())
        return 42

    def _perturb(self, model: Any, seed: int, sign: float) -> None:
        """Add sign * eps * z to all trainable parameters in-place."""
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn(p.shape, generator=rng)
                p.data.add_(sign * self.eps * z.to(p.device))

    def _apply_update(self, model: Any, optimizer: Any, grad_est: float, seed: int) -> None:
        """θ ← θ − lr · grad_est · z  (no torch optimizer step needed)."""
        lr = _current_lr(optimizer)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn(p.shape, generator=rng)
                p.data.add_(-lr * grad_est * z.to(p.device))

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
        loss_fn = ctx.loss_fn
        optimizer = ctx.optimizer
        accelerator = ctx.accelerator

        if bus is not None:
            bus.dispatch("on_step_begin", step=ctx.step, ctx=ctx, batch=batch)

        seed = self._seed()
        self._step_count += 1

        ctx.extras["model"] = model

        # MeZO runs two forward passes (L+ / L-) per step; build a fresh autocast
        # CM each pass (accelerator.autocast() is single-use — caching crashes L-).
        def _forward(b: Any) -> tuple[Any, torch.Tensor, dict]:
            with make_autocast(accelerator):
                _out = model(**b)
                if not isinstance(_out, ModelOutput):
                    _out = ModelOutput(outputs={"logits": _out} if not isinstance(_out, Mapping) else dict(_out))
                _lctx = LossContext(step=ctx.step, epoch=ctx.epoch, metrics=ctx.metrics, extras=ctx.extras)
                _ld = loss_fn(_out, b, _lctx)
            if "loss" not in _ld:
                raise KeyError("LossFn must return a dict containing 'loss'.")
            return _out, _ld["loss"], _ld

        # L+  (θ + ε·z)
        self._perturb(model, seed, +1.0)
        _, loss_plus, loss_dict = _forward(batch)
        lp = float(loss_plus.detach().item())

        # L-  (θ − 2ε·z, net = θ − ε·z relative to original)
        self._perturb(model, seed, -2.0)
        _, loss_minus, _ = _forward(batch)
        lm = float(loss_minus.detach().item())

        # Restore to original θ
        self._perturb(model, seed, +1.0)

        grad_est = (lp - lm) / (2.0 * self.eps)

        if bus is not None:
            bus.dispatch("on_forward_post", step=ctx.step, outputs=None, model=model)

        # Gradient-free update: θ ← θ − lr·grad_est·z
        self._apply_update(model, optimizer, grad_est, seed)

        ctx.metrics["loss"] = (lp + lm) / 2.0  # report average loss
        ctx.metrics["grad_est"] = grad_est
        ctx.metrics["lr"] = _current_lr(optimizer)
        ctx.metrics["grad_norm"] = math.fabs(grad_est)  # surrogate
        ctx.metrics["skipped"] = 0.0

        for k, v in loss_dict.items():
            if k != "loss":
                ctx.metrics[k] = _to_metric(v)

        if bus is not None:
            bus.dispatch("on_step_end", step=ctx.step, metrics=ctx.metrics, batch=batch, model=model)

        return dict(ctx.metrics)


__all__ = ["MeZOUpdateRule"]
