"""ZeROStrategy — DeepSpeed ZeRO-1/2/3/Infinity.

DeepSpeed's engine merges backward + clip + step into a single
``engine.step()`` call, so the protocol methods map as follows:
  backward()        → engine.backward(loss)
  clip_grad_norm()  → engine.get_global_grad_norm()  (already clipped by engine.step)
  optimizer_step()  → engine.step()

The ``optimizer`` argument passed to ``optimizer_step`` is the DeepSpeed
engine itself (returned as the ``optimizer`` slot from ``prepare()``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


@register("grad_sync_strategy", "deepspeed")
class ZeROStrategy:
    """DeepSpeed ZeRO gradient synchronisation and optimizer sharding."""

    def __init__(
        self,
        *,
        zero_stage: int = 2,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        fp16: bool = False,
        bf16: bool = True,
        gradient_clipping: float = 1.0,
        config_file: str | None = None,
    ) -> None:
        self.zero_stage = int(zero_stage)
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.fp16 = fp16
        self.bf16 = bf16
        self.gradient_clipping = float(gradient_clipping)
        self.config_file = config_file
        self._engine: Any = None

    def _build_ds_config(self, train_batch_size: int = 1) -> dict[str, Any]:
        if self.config_file:
            import json
            with open(self.config_file) as f:
                return json.load(f)

        cfg: dict[str, Any] = {
            "train_batch_size": train_batch_size,
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {"stage": self.zero_stage},
        }
        if self.offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        if self.offload_param and self.zero_stage >= 3:
            cfg["zero_optimization"]["offload_param"] = {"device": "cpu"}
        if self.fp16:
            cfg["fp16"] = {"enabled": True}
        elif self.bf16:
            cfg["bf16"] = {"enabled": True}
        return cfg

    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: ParallelContext,
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]:
        import deepspeed

        raw_optimizer = optimizer_factory(model)
        inner_opt = getattr(raw_optimizer, "optimizer", raw_optimizer)
        ds_config = self._build_ds_config()

        engine, ds_opt, _, _ = deepspeed.initialize(
            model=model,
            optimizer=inner_opt,
            config=ds_config,
        )
        self._engine = engine
        # Return engine as both model AND optimizer so dispatch points work.
        return engine, engine, loader

    def accumulate(self, model: nn.Module) -> Any:
        from contextlib import nullcontext
        return nullcontext()  # ZeRO handles gradient accumulation internally

    def backward(self, loss: torch.Tensor, model: nn.Module) -> None:
        model.backward(loss)  # type: ignore[attr-defined]  # model is DS engine

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        parallel_ctx: ParallelContext,
    ) -> float:
        # DeepSpeed clips internally during engine.step(); retrieve the norm.
        try:
            return float(model.get_global_grad_norm())  # type: ignore[attr-defined]
        except AttributeError:
            return 0.0

    def optimizer_step(self, optimizer: Any, model: nn.Module) -> None:
        model.step()  # type: ignore[attr-defined]  # model is DS engine; step() = clip+step+zero_grad

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module  # type: ignore[attr-defined]

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(path), tag=f"step_{step}")  # type: ignore[attr-defined]

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        model.load_checkpoint(str(path))  # type: ignore[attr-defined]

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "deepspeed",
            "zero_stage": self.zero_stage,
            "offload_optimizer": self.offload_optimizer,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


__all__ = ["ZeROStrategy"]
