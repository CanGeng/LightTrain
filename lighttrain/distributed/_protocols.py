"""Distributed strategy protocols.

Three protocols cover the full parallelism spectrum:

* ``GradSyncStrategy``      — model-agnostic DP: DDP / FSDP / ZeRO
* ``ModelParallelStrategy`` — model-aware intra-op: TP / SP / EP
* ``PipelineSchedule``      — layer-structure-aware: PP (1F1B / GPipe)

Implementations live in ``plugins/distributed/``.  The protocols
themselves live here (core) so that ``StandardEngine``, ``StandardUpdateRule``,
and ``CheckpointManager`` can type-check against them without pulling in
heavy optional dependencies (torch.distributed, deepspeed, etc.).

All three protocols use structural subtyping (duck-typing) — no inheritance
required.  The ``runtime_checkable`` decorator allows ``isinstance`` checks
in tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

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


@runtime_checkable
class ModelParallelStrategy(Protocol):
    """Model-aware parallelism surgery: TP, SP, EP.

    Must be applied **before** ``GradSyncStrategy.prepare()`` and before
    any FSDP wrapping, because FSDP needs to see already-sharded parameters.
    """

    def apply(self, model: nn.Module, parallel_ctx: ParallelContext) -> nn.Module:
        """Return the model with TP/SP/EP surgery applied.

        For TP: replaces ``nn.Linear`` layers with DTensor column/row shards.
        For SP: replaces sequence-consuming layers with sequence-sharded variants.
        For EP: routes MoE expert layers to assigned ranks and installs all-to-all hooks.
        """
        ...

    def is_stateless(self) -> bool:
        """True for strategies that only reshard weights (TP, SP).
        False for EP where routing state is rank-specific.
        """
        ...


@runtime_checkable
class PipelineSchedule(Protocol):
    """Layer-structure-aware pipeline parallelism: 1F1B, GPipe, Interleaved.

    Must be applied **after** ``ModelParallelStrategy.apply()`` (TP surgery)
    and **before** ``GradSyncStrategy.prepare()`` (FSDP/DDP wrapping on DP dim).
    """

    def prepare(
        self, model: nn.Module, parallel_ctx: ParallelContext
    ) -> Any:
        """Split model into PP stages and return the local stage module.

        The returned object is what ``ctx.model`` will be set to for the
        duration of training on this rank.
        """
        ...

    def run_step(
        self, stage: Any, microbatches: list[dict[str, Any]], ctx: Any
    ) -> torch.Tensor:
        """Execute one global step via micro-batch schedule.

        Returns the loss scalar (valid only on ``pp_last_stage`` rank;
        other ranks return a zero tensor that should not be logged).
        """
        ...


__all__ = [
    "GradSyncStrategy",
    "ModelParallelStrategy",
    "PipelineSchedule",
]
