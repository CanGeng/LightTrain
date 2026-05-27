"""``OptimizerCPUOffloadWrapper``.

Wraps any registered ``optimizer`` short name. State (m / v / momentum /
etc.) is kept fp32 on host; params on the device live as a bf16 / fp16
working copy. On ``step()`` we:

1. copy each device ``param.grad`` to the host fp32 master grad slot,
2. invoke the *base* optimizer.step() entirely on host fp32 master params,
3. cast master back to the device working precision so subsequent forwards
   see the updated weights.

This is the "host fp32 master + device low-precision working copy" recipe.

Single-GPU friendly: when no CUDA device is present, the wrapper acts as
an identity over the base optimizer (still legal; useful for unit tests).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from lighttrain.registry import register


@register("optimizer", "cpu_offload")
class OptimizerCPUOffloadWrapper:
    """Lighttrain ``OptimizerWrapperProtocol`` host-side step wrapper."""

    def __init__(
        self,
        *,
        base: Mapping[str, Any] | None = None,
        master_dtype: str = "float32",
        working_dtype: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        from lighttrain.config._resolver import resolve as _resolve

        if base is None:
            base = {"name": "adamw"}
        base_spec = dict(base)
        # Allow flat ``optim.lr=...`` style: forward extras to base.
        for k, v in base_kwargs.items():
            base_spec.setdefault(k, v)
        self._base_factory = _resolve(base_spec, category="optimizer", instantiate=False)
        self._base_kwargs = {k: v for k, v in base_spec.items()
                             if k not in ("name", "_target_", "params")}
        # Honor inner ComponentSpec.params if present
        if isinstance(base_spec, dict) and "params" in base_spec:
            self._base_kwargs.update(base_spec.get("params") or {})

        self.master_dtype = _resolve_dtype(master_dtype)
        self.working_dtype = (
            _resolve_dtype(working_dtype) if working_dtype else None
        )
        self.optimizer: torch.optim.Optimizer | None = None
        self._host_master: dict[int, torch.Tensor] = {}
        self._wrapper = None  # the resolved inner OptimizerWrapper

    # ---- build / parameter-group plumbing ------------------------------

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        # Construct the base optimizer (must satisfy OptimizerWrapperProtocol).
        wrapper = self._base_factory(**self._base_kwargs)
        self._wrapper = wrapper
        # Note: we hand the *device* model to the base wrapper; it builds
        # its own optimizer over device params. We then replace each
        # device param's data ptr with a host fp32 master clone for state.
        inner_opt = wrapper.build(model)
        self.optimizer = inner_opt

        device = next(model.parameters()).device
        for p in (
            p for g in inner_opt.param_groups for p in g["params"]
            if p.requires_grad
        ):
            with torch.no_grad():
                master = p.detach().to(dtype=self.master_dtype, device="cpu").clone()
                # Pin if CUDA is available — speeds up D2H grad transfer.
                if device.type == "cuda":
                    try:
                        master = master.pin_memory()
                    except Exception:  # noqa: BLE001
                        pass
                self._host_master[id(p)] = master
        return inner_opt

    @property
    def param_groups_list(self) -> list[dict[str, Any]]:
        if self.optimizer is None:
            return []
        return list(self.optimizer.param_groups)

    # ---- step / zero_grad / state_dict ---------------------------------

    def step(self, *args: Any, **kwargs: Any) -> Any:
        if self.optimizer is None:
            raise RuntimeError("OptimizerCPUOffloadWrapper.step before build()")
        # 1) Stage grads to host fp32 master grad slots.
        for p in (
            p for g in self.optimizer.param_groups for p in g["params"]
        ):
            master = self._host_master.get(id(p))
            if master is None:
                continue
            if p.grad is None:
                master.grad = None
                continue
            with torch.no_grad():
                host_grad = p.grad.detach().to(
                    dtype=self.master_dtype, device="cpu"
                )
                if master.grad is None:
                    master.grad = host_grad.clone()
                else:
                    master.grad.copy_(host_grad)
        # 2) Promote any plain host tensor to ``nn.Parameter`` once, and
        # temporarily rebind optimizer.param_groups to those masters so the
        # base step() operates entirely on host fp32.
        originals: list[tuple[dict, list[torch.nn.Parameter], list[torch.nn.Parameter]]] = []
        for g in self.optimizer.param_groups:
            origs = list(g["params"])
            masters: list[torch.nn.Parameter] = []
            for p in origs:
                master = self._host_master.get(id(p))
                if master is None:
                    masters.append(p)
                    continue
                if not isinstance(master, torch.nn.Parameter):
                    grad_save = getattr(master, "grad", None)
                    master_param = torch.nn.Parameter(master, requires_grad=False)
                    master_param.grad = grad_save
                    self._host_master[id(p)] = master_param
                    master = master_param
                masters.append(master)
                # 2.5) Migrate any pre-existing state key from orig → master so
                # consecutive steps accumulate momentum / variance correctly.
                if p in self.optimizer.state:
                    self.optimizer.state[master] = self.optimizer.state.pop(p)
            g["params"] = masters
            originals.append((g, origs, masters))

        try:
            result = self.optimizer.step(*args, **kwargs)
        finally:
            # 3) Cast master back to working precision and copy to device.
            # 4) Restore original param objects in param_groups *and* re-key
            # optimizer.state so future state_dict / load_state_dict work.
            for g, origs, masters in originals:
                for orig, master in zip(origs, masters):
                    with torch.no_grad():
                        target_dtype = self.working_dtype or orig.dtype
                        orig.data.copy_(master.detach().to(
                            dtype=target_dtype, device=orig.device
                        ))
                    if master in self.optimizer.state:
                        self.optimizer.state[orig] = self.optimizer.state.pop(master)
                g["params"] = origs
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.optimizer is None:
            return
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for master in self._host_master.values():
            if hasattr(master, "grad") and master.grad is not None:
                master.grad = None if set_to_none else torch.zeros_like(master)

    def state_dict(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {}
        return {
            "inner": self.optimizer.state_dict(),
            "master_dtype": str(self.master_dtype),
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Build the optimizer before loading state_dict.")
        inner = sd.get("inner") if isinstance(sd, Mapping) else sd
        if inner is not None:
            self.optimizer.load_state_dict(inner)


def _resolve_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    n = str(name).lower()
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if n not in mapping:
        raise ValueError(f"OptimizerCPUOffloadWrapper: unknown dtype {name!r}")
    return mapping[n]


__all__ = ["OptimizerCPUOffloadWrapper"]
