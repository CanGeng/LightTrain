"""FSDPStrategy — FullyShardedDataParallel (PyTorch native).

Key constraints vs DDP:
- Optimizer MUST be created after model wrapping.
- Gradient clipping uses ``model.clip_grad_norm_()`` (FSDP-aware).
- ``unwrap_model()`` returns the FSDP wrapper itself; use StateDictType to
  control what state_dict returns (full vs sharded).

Checkpoint modes (``state_dict_type``):
- ``"full"``    — gather to rank-0, write ``model.safetensors`` (slow but portable)
- ``"sharded"`` — each rank writes its own shard (fast, requires resharding on resume)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register

_log = logging.getLogger(__name__)


@register("grad_sync_strategy", "fsdp")
class FSDPStrategy:
    """FullyShardedDataParallel gradient synchronisation strategy."""

    def __init__(
        self,
        *,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision_policy: str | None = None,
        auto_wrap_policy: str = "transformer_layer",
        activation_checkpointing: bool = False,
        cpu_offload: bool = False,
        state_dict_type: Literal["full", "sharded"] = "full",
        min_num_params: int = 100_000,
    ) -> None:
        self.sharding_strategy_name = sharding_strategy
        self.mixed_precision_policy_name = mixed_precision_policy
        self.auto_wrap_policy_name = auto_wrap_policy
        self.activation_checkpointing = activation_checkpointing
        self.cpu_offload = cpu_offload
        self.state_dict_type = state_dict_type
        self.min_num_params = min_num_params

    def _fsdp_kwargs(self, parallel_ctx: ParallelContext, device: torch.device) -> dict[str, Any]:
        import functools

        from torch.distributed.fsdp import (
            CPUOffload,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        kwargs: dict[str, Any] = {
            "process_group": parallel_ctx.dp_group,
            "device_id": device if device.type == "cuda" else None,
            "sharding_strategy": getattr(ShardingStrategy, self.sharding_strategy_name, ShardingStrategy.FULL_SHARD),
        }

        if self.cpu_offload:
            kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        if self.mixed_precision_policy_name:
            dtype = torch.bfloat16 if "bf16" in self.mixed_precision_policy_name else torch.float16
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )

        kwargs["auto_wrap_policy"] = functools.partial(
            size_based_auto_wrap_policy, min_num_params=self.min_num_params
        )

        return kwargs

    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: ParallelContext,
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp_kwargs = self._fsdp_kwargs(parallel_ctx, device)
        wrapped = FSDP(model, **fsdp_kwargs)

        if self.activation_checkpointing:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    apply_activation_checkpointing,
                )
                apply_activation_checkpointing(wrapped)
            except Exception:  # noqa: BLE001
                _log.warning("fsdp.prepare: activation checkpointing wrap failed; continuing without it", exc_info=True)

        # Optimizer MUST be built after FSDP wrapping.
        optimizer = optimizer_factory(wrapped)
        return wrapped, optimizer, loader

    def accumulate(self, model: nn.Module) -> Any:
        return cast(Any, model).no_sync()  # FSDP-only method, not on base nn.Module

    def backward(self, loss: torch.Tensor, model: nn.Module) -> None:
        loss.backward()

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        parallel_ctx: ParallelContext,
    ) -> float:
        return float(cast(Any, model).clip_grad_norm_(max_norm))  # FSDP-only method

    def optimizer_step(self, optimizer: Any, model: nn.Module) -> None:
        inner = getattr(optimizer, "optimizer", optimizer)
        inner.step()

    def zero_grad(self, optimizer: Any) -> None:
        inner = getattr(optimizer, "optimizer", optimizer)
        inner.zero_grad(set_to_none=True)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model  # FSDP wrapper is the canonical state-dict carrier

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        path.mkdir(parents=True, exist_ok=True)

        if self.state_dict_type == "full":
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                sd = model.state_dict()
            if parallel_ctx.is_main_process:
                from safetensors.torch import save_file
                save_file(
                    {k: v.detach().cpu().clone() for k, v in sd.items()},
                    str(path / "model.safetensors"),
                )
        else:
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                sd = model.state_dict()
            torch.save(sd, str(path / f"shard_{parallel_ctx.rank}.pt"))

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self.state_dict_type == "full":
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                from safetensors.torch import load_file
                sd = load_file(str(path / "model.safetensors"))
                model.load_state_dict(sd, strict=False)
        else:
            # Sharded loading — each rank reads its shard.
            sd = torch.load(str(path / f"shard_{parallel_ctx.rank}.pt"), map_location="cpu")
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                model.load_state_dict(sd, strict=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "fsdp",
            "sharding_strategy": self.sharding_strategy_name,
            "state_dict_type": self.state_dict_type,
            "activation_checkpointing": self.activation_checkpointing,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


__all__ = ["FSDPStrategy"]
