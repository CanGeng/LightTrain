"""DDPStrategy — DistributedDataParallel gradient synchronisation.

Wraps the model with torch's DDP.  The optimizer is created AFTER wrapping
so that parameter references are stable (though for DDP this is not strictly
necessary, it maintains API consistency with FSDP).

Checkpoint strategy:
- rank-0 writes a single ``model.safetensors``  (all ranks have identical params)
- optimizer state: rank-0 writes ``optimizer.pt``
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file as _stensors_save

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


@register("grad_sync_strategy", "ddp")
class DDPStrategy:
    """DistributedDataParallel gradient synchronisation strategy."""

    def __init__(
        self,
        *,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        broadcast_buffers: bool = True,
    ) -> None:
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.broadcast_buffers = broadcast_buffers

    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: ParallelContext,
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]:
        from torch.nn.parallel import DistributedDataParallel

        model = model.to(device)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[parallel_ctx.local_rank] if device.type == "cuda" else None,
            process_group=parallel_ctx.dp_group,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            broadcast_buffers=self.broadcast_buffers,
        )
        optimizer = optimizer_factory(wrapped)
        return wrapped, optimizer, loader

    def accumulate(self, model: nn.Module) -> Any:
        return model.no_sync()

    def backward(self, loss: torch.Tensor, model: nn.Module) -> None:
        loss.backward()

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        parallel_ctx: ParallelContext,
    ) -> float:
        return float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        )

    def optimizer_step(self, optimizer: Any, model: nn.Module) -> None:
        inner = getattr(optimizer, "optimizer", optimizer)
        inner.step()

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
        if not parallel_ctx.is_main_process:
            return
        path.mkdir(parents=True, exist_ok=True)
        sd = self.unwrap_model(model).state_dict()
        _stensors_save(
            {k: v.detach().cpu().clone() for k, v in sd.items()},
            str(path / "model.safetensors"),
        )

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        from safetensors.torch import load_file
        sd = load_file(str(path / "model.safetensors"))
        self.unwrap_model(model).load_state_dict(sd, strict=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "ddp",
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


__all__ = ["DDPStrategy"]
