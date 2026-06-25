"""Distributed strategy protocols.

* ``GradSyncStrategy`` — model-agnostic DP: DDP / FSDP / ZeRO

Implementations live in ``builtin_plugins/distributed/``.  The protocol
itself lives here (core) so that ``StandardEngine``, ``StandardUpdateRule``,
and ``CheckpointManager`` can type-check against it without pulling in
heavy optional dependencies (torch.distributed, deepspeed, etc.).

The protocol uses structural subtyping (duck-typing) — no inheritance
required.  The ``runtime_checkable`` decorator allows ``isinstance`` checks
in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ._context import ParallelContext


@runtime_checkable
class GradSyncStrategy(Protocol):
    """Model-agnostic gradient synchronisation: DDP, FSDP, DeepSpeed ZeRO.

    Implementations must handle:
    - ``prepare``: wrap model, create optimizer, inject DistributedSampler
    - ``accumulate``: suppress gradient sync during micro-steps
    - ``backward`` / ``clip_grad_norm`` / ``optimizer_step``: backend-specific
    - ``unwrap_model``: strip wrapper for checkpoint / surgery access
    - ``save_checkpoint`` / ``load_checkpoint``: rank-aware I/O
    """

    def prepare(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[nn.Module], Any],
        loader: Any,
        parallel_ctx: ParallelContext,
        *,
        device: torch.device,
    ) -> tuple[nn.Module, Any, Any]:
        """Return ``(wrapped_model, optimizer, loader)``.

        ``optimizer_factory`` is a callable ``model -> optimizer`` so that
        FSDP (which requires the optimizer to be built *after* wrapping)
        can create it internally.
        """
        ...

    def accumulate(self, model: nn.Module) -> Any:
        """Context manager that suppresses inter-rank gradient sync.

        DDP/FSDP → ``model.no_sync()``; ZeRO → ``nullcontext()``.
        """
        ...

    def backward(self, loss: torch.Tensor, model: nn.Module) -> None:
        """Backend-aware backward pass."""
        ...

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        parallel_ctx: ParallelContext,
    ) -> float:
        """Backend-aware gradient clipping. Returns the global gradient norm."""
        ...

    def optimizer_step(self, optimizer: Any, model: nn.Module) -> None:
        """Backend-aware optimizer step (ZeRO: engine.step() replaces opt.step())."""
        ...

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Strip DDP/FSDP wrapper to get the original ``nn.Module``."""
        ...

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        """Write rank-aware checkpoint. Implementations decide sharded vs. full."""
        ...

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        parallel_ctx: ParallelContext,
        path: Path,
    ) -> None:
        """Load checkpoint, handling topology mismatches via resharding."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Strategy hyperparameter snapshot for run lineage."""
        ...

    def load_state_dict(self, sd: dict[str, Any]) -> None: ...


__all__ = [
    "GradSyncStrategy",
]
