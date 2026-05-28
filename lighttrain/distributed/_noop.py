"""NoopGradSyncStrategy — single-GPU passthrough.

Satisfies the GradSyncStrategy protocol with zero overhead. Used when
``parallel.grad_sync.name == "noop"`` (the default for world_size == 1).
No torch.distributed calls are made anywhere in this class.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from ._context import ParallelContext
from ..registry._core import register


@register("grad_sync_strategy", "noop")
class NoopGradSyncStrategy:
    """Single-GPU GradSyncStrategy — direct passthrough, no distributed I/O."""

    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: ParallelContext,
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]:
        model = model.to(device)
        optimizer = optimizer_factory(model)
        return model, optimizer, loader

    def accumulate(self, model: nn.Module) -> Any:
        return nullcontext()

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
        optimizer.step()

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        pass  # delegated to CheckpointManager; noop here

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        pass  # delegated to CheckpointManager; noop here

    def state_dict(self) -> dict[str, Any]:
        return {"name": "noop"}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


__all__ = ["NoopGradSyncStrategy"]
