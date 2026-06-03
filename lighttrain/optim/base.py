"""Optimizer-wrapper / scheduler base classes + the param-group DSL.

These are the core abstraction/extension points (DESIGN §3.3): concrete
optimizers and schedulers in ``lighttrain.builtin_plugins.optim`` subclass
``OptimizerWrapperBase`` / ``SchedulerBase``. ``ParamGroupSpec`` (+ the
``_split_param_groups`` helper) is the shared param-group selection DSL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

_BYTES_PER_DTYPE: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
}


def _trainable_param_bytes(model: torch.nn.Module) -> int:
    return sum(
        p.numel() * _BYTES_PER_DTYPE.get(p.dtype, 4)
        for p in model.parameters()
        if p.requires_grad
    )


@dataclass
class ParamGroupSpec:
    """Regex-based parameter-group selector (first match wins).

    ``pattern`` is a Python regex matched against fully-qualified parameter
    names (``layer.0.weight``). Any extra keys (lr / weight_decay / ...)
    override the optimizer defaults for matched parameters.

    Optional **additive** predicates (applied *after* the name regex; default
    ``None`` = name-only, the legacy behavior):

    * ``min_ndim`` — only params whose tensor has ``ndim >= min_ndim`` (e.g.
      ``2`` selects weight matrices and excludes 1-D bias/norm vectors).
    * ``module_type`` — only params owned by a module whose class name equals
      this string (e.g. ``"Linear"`` selects ``nn.Linear`` params). Matches
      against ``type(module).__name__``.

    These let the built-in DSL express selections like GaLore's "Linear
    weights, ndim>=2" without dropping to a custom ``build()``.
    """

    pattern: str
    options: dict[str, Any] = field(default_factory=dict)
    min_ndim: int | None = None
    module_type: str | None = None

    def match(
        self,
        name: str,
        param: "torch.Tensor | None" = None,
        module: "torch.nn.Module | None" = None,
    ) -> bool:
        if re.search(self.pattern, name) is None:
            return False
        if self.min_ndim is not None and param is not None and param.ndim < self.min_ndim:
            return False
        if self.module_type is not None and module is not None:
            if type(module).__name__ != self.module_type:
                return False
        return True


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

    # Map each param to its owning module so specs can filter by module_type.
    param_to_module: dict[int, torch.nn.Module] = {}
    for mod in model.modules():
        for p in mod.parameters(recurse=False):
            param_to_module[id(p)] = mod

    buckets: list[dict[str, Any]] = [{"params": [], **defaults, **s.options} for s in specs]
    fallback: dict[str, Any] = {"params": [], **defaults}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module = param_to_module.get(id(param))
        for i, s in enumerate(specs):
            if s.match(name, param, module):
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


class OptimizerWrapperBase:
    """Common bookkeeping for optimizer wrappers (the core extension point).

    Concrete wrappers (``lighttrain.builtin_plugins.optim.wrappers``) subclass
    this and implement ``build(model) -> torch.optim.Optimizer``. The wrapper
    exposes ``.optimizer`` plus the usual ``step / zero_grad / state_dict /
    load_state_dict`` so calling code stays ignorant of the wrapper layer.
    """

    optimizer: torch.optim.Optimizer

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)
        groups_raw = self._kwargs.pop("param_groups", None)
        self.param_groups: list[ParamGroupSpec] | None = None
        if groups_raw:
            _reserved = ("pattern", "min_ndim", "module_type")
            self.param_groups = [
                ParamGroupSpec(
                    pattern=g["pattern"],
                    min_ndim=g.get("min_ndim"),
                    module_type=g.get("module_type"),
                    options={k: v for k, v in g.items() if k not in _reserved},
                )
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

    def _safe_state_dict(
        self,
        convert: "Callable[[Any, Any], Any] | None" = None,
    ) -> dict[str, Any]:
        """A ``state_dict()`` whose inner per-param state dicts are **copies** —
        safe to mutate.

        ⚠️ Aliasing trap: ``torch.optim.Optimizer.state_dict()`` returns the
        *same* inner ``state[param]`` dict objects that the live optimizer uses.
        So an override that rewrites custom state in place (e.g. serializing a
        non-tensor object like a projector to plain tensors for a portable
        checkpoint) would corrupt the running optimizer — the next ``.step()``
        finds your serialized form instead of the live object. Always copy
        first; this helper does that for you.

        Pass ``convert(key, value) -> value`` to transform individual state
        entries on the copy. Example (portable custom state)::

            def state_dict(self):
                def conv(k, v):
                    return v.as_tensors() if k == "projector" else v
                return self._safe_state_dict(conv)
        """
        sd = self.optimizer.state_dict()
        new_state: dict[Any, Any] = {}
        for pid, st in sd.get("state", {}).items():
            if isinstance(st, dict):
                st = dict(st)  # copy so caller-side mutation can't alias self.state
                if convert is not None:
                    for k in list(st):
                        st[k] = convert(k, st[k])
            new_state[pid] = st
        sd["state"] = new_state
        return sd

    @property
    def param_groups_list(self) -> list[dict[str, Any]]:
        return list(self.optimizer.param_groups)

    def optim_state_bytes(self, model: torch.nn.Module) -> int:
        """Per-step optimizer-state footprint, in bytes.

        Default = ``2 × trainable_param_bytes`` (Adam's ``m`` + ``v``). Called
        by ``lab.estimate`` (via the optional protocol hook) so memory-efficient
        optimizers can report their real saving by overriding this. Computed
        from the model + the wrapper's own kwargs; does **not** require
        ``build()``.
        """
        return self._moments_per_param() * _trainable_param_bytes(model)

    def _moments_per_param(self) -> int:
        """How many full-size moment buffers this optimizer keeps per param."""
        return 2


class SchedulerBase:
    """Per-step LR scheduler base (the core extension point).

    Concrete schedulers (``lighttrain.builtin_plugins.optim.schedulers``)
    subclass this and implement ``_factor(step) -> float``. All schedulers tick
    once per optimizer step (``step_per_batch = True``).
    """

    step_per_batch: bool = True

    def __init__(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        self.optimizer: torch.optim.Optimizer | None = optimizer
        self.last_step: int = 0
        self._base_lrs: list[float] = []
        if optimizer is not None:
            self.attach(optimizer)

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._base_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]

    def _set_lrs(self, factor: float) -> None:
        if self.optimizer is None or not self._base_lrs:
            return
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base * factor

    def step(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.last_step += 1
        self._set_lrs(self._factor(self.last_step))

    def _factor(self, step: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_lr(self) -> list[float]:
        if self.optimizer is None:
            return []
        return [g.get("lr", 0.0) for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "base_lrs": list(self._base_lrs)}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.last_step = int(sd.get("last_step", 0))
        if "base_lrs" in sd:
            self._base_lrs = list(sd["base_lrs"])


__all__ = [
    "OptimizerWrapperBase",
    "ParamGroupSpec",
    "SchedulerBase",
    "_split_param_groups",
    "_trainable_param_bytes",
]
