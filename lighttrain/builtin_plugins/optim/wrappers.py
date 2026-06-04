"""Optimizer wrappers — AdamW / Lion (concrete impls).

The param-group DSL (``ParamGroupSpec`` / ``_split_param_groups``) and the
``OptimizerWrapperBase`` abstraction live in ``lighttrain.optim.base`` (core);
these registered optimizers subclass it (DESIGN §3.3).
"""

from __future__ import annotations

from typing import Any

import torch

from lighttrain.optim.base import OptimizerWrapperBase, _split_param_groups
from lighttrain.registry import register


@register("optimizer", "adamw")
class AdamWWrapper(OptimizerWrapperBase):
    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        self._check_unbuilt()
        groups = _split_param_groups(model, self.param_groups, self._kwargs)
        self.optimizer = torch.optim.AdamW(groups)
        self._built = True
        return self.optimizer


@register("optimizer", "lion")
class LionWrapper(OptimizerWrapperBase):
    """Lion optimizer (Chen et al. 2023). Pure-PyTorch reference impl."""

    def _moments_per_param(self) -> int:
        return 1  # Lion keeps a single momentum buffer

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        self._check_unbuilt()
        groups = _split_param_groups(model, self.param_groups, self._kwargs)
        self.optimizer = _Lion(groups)
        self._built = True
        return self.optimizer


class _Lion(torch.optim.Optimizer):
    """Reference Lion. Single-GPU, no fused kernels."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = (exp_avg * beta1 + grad * (1 - beta1)).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


__all__ = ["AdamWWrapper", "LionWrapper"]
