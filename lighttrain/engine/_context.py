"""StepContext — the shared state passed to engine / update_rule / callbacks.

Kept deliberately concrete (a dataclass, not a Protocol) so callbacks can
mutate ``metrics`` in place without going through indirection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..distributed._context import ParallelContext


@dataclass
class StepContext:
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    # Authoritative count of batches consumed in the *current* epoch. The
    # training loop bumps it once per ``next(iterator)`` (so it tracks what the
    # trainer actually consumed, NOT what the sampler/prefetch has yielded) and
    # resets it to 0 on epoch rollover. Persisted in the checkpoint and used to
    # resume the data sampler step-exactly mid-epoch (BUG-1), independent of
    # DataLoader prefetch depth.
    batch_in_epoch: int = 0
    is_accumulating: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # injected by the trainer:
    model: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    loss_fn: Any | None = None
    accelerator: Any | None = None
    bus: Any | None = None  # EventBus
    logger: Any | None = None  # LoggerBus
    # Lineage is a soft dependency.
    lineage_store: Any | None = None
    run_id: str | None = None
    # ``run_dir`` lets diagnostics callbacks (nan_hunter / loss_attribution /
    # frozen_step / crash_bundle / file_signals) drop artifacts under
    # <run_dir>/diagnostics/* without rediscovering the path. ``mode`` is the
    # lab/prod switch that toggles diagnostics defaults.
    # ``frozen_step_writer`` is set by FrozenStepCallback so the StandardUpdate
    # Rule can borrow its per-step snapshot for RETRY_STEP replay.
    # ``diagnostics`` is a free-form scratch dict — diagnostic callbacks stash
    # state here without touching ``extras`` (which trainer treats as signals).
    run_dir: Path | None = None
    mode: str = "lab"
    frozen_step_writer: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Distributed fields — always present; default to single-GPU values.
    # parallel_ctx is ParallelContext.single_gpu() when not distributed.
    # grad_sync is None for single-GPU (the update rule uses loss.backward()
    # directly); set to a GradSyncStrategy instance for DDP/FSDP/ZeRO.
    parallel_ctx: ParallelContext | None = None
    grad_sync: Any | None = None


__all__ = ["StepContext"]
