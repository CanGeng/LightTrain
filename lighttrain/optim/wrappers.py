"""Optimizer wrappers — param-group DSL + AdamW/Lion.

The wrapper exposes ``.optimizer`` (a ``torch.optim.Optimizer``) plus the
usual ``step / zero_grad / state_dict / load_state_dict`` so calling code
stays ignorant of the wrapper layer. ``.build(model)`` is invoked once by
the trainer; calling it again raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from ..registry import register


@dataclass
class ParamGroupSpec:
    """Regex-based parameter-group selector (first match wins).

    ``pattern`` is a Python regex matched against fully-qualified parameter
    names (``layer.0.weight``). Any extra keys (lr / weight_decay / ...)
    override the optimizer defaults for matched parameters.
    """

    pattern: str
    options: dict[str, Any] = field(default_factory=dict)

    def match(self, name: str) -> bool:
        return re.search(self.pattern, name) is not None


def _split_param_groups(
    model: torch.nn.Module,
    specs: list[ParamGroupSpec] | None,
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    if not specs:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Model has no trainable parameters.")
        return [{"params": params, **defaults}]

    buckets: list[dict[str, Any]] = [{"params": [], **defaults, **s.options} for s in specs]
    fallback: dict[str, Any] = {"params": [], **defaults}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for i, s in enumerate(specs):
            if s.match(name):
                buckets[i]["params"].append(param)
                break
        else:
            fallback["params"].append(param)

    if fallback["params"]:
        buckets.append(fallback)
    buckets = [b for b in buckets if b["params"]]
    if not buckets:
        raise ValueError("No parameters matched any param-group spec.")
    return buckets


class _BaseWrapper:
    """Common bookkeeping for the wrappers below."""

    optimizer: torch.optim.Optimizer

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)
        groups_raw = self._kwargs.pop("param_groups", None)
        self.param_groups: list[ParamGroupSpec] | None = None
        if groups_raw:
            self.param_groups = [
                ParamGroupSpec(pattern=g["pattern"], options={k: v for k, v in g.items() if k != "pattern"})
                for g in groups_raw
            ]
        self._built = False
        self.optimizer = None  # type: ignore[assignment]

    def _check_unbuilt(self) -> None:
        if self._built:
            raise RuntimeError("Optimizer already built; rebuild not supported.")

    def step(self, *a: Any, **kw: Any) -> Any:
        return self.optimizer.step(*a, **kw)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(sd)

    @property
    def param_groups_list(self) -> list[dict[str, Any]]:
        return list(self.optimizer.param_groups)


@register("optimizer", "adamw")
class AdamWWrapper(_BaseWrapper):
    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        self._check_unbuilt()
        groups = _split_param_groups(model, self.param_groups, self._kwargs)
        self.optimizer = torch.optim.AdamW(groups)
        self._built = True
        return self.optimizer


@register("optimizer", "lion")
class LionWrapper(_BaseWrapper):
    """Lion optimizer (Chen et al. 2023). Pure-PyTorch reference impl."""

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
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
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


__all__ = ["AdamWWrapper", "LionWrapper", "ParamGroupSpec"]
