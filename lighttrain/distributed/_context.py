"""ParallelContext — distributed topology carrier.

Always present: single-GPU is ``ParallelContext.single_gpu()`` with all
degrees == 1 and rank == 0. Callers that don't know about distributed can
safely read ``parallel_ctx.is_main_process`` (always True on single-GPU)
and ``parallel_ctx.local_device`` (cuda:0 or cpu).

``torch.distributed`` is lazily imported: ``ParallelContext.single_gpu()``
works without NCCL or any dist initialization.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..config._schema import ParallelSection

_log = logging.getLogger(__name__)


@dataclass
class ParallelContext:
    # Global identity
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    # Data-parallel rank/size (0/1 for single-GPU)
    dp_rank: int = 0
    dp_degree: int = 1

    force_cpu: bool = False

    # torch.distributed objects — None on single-GPU
    device_mesh: Any | None = None   # torch.distributed.DeviceMesh
    dp_group: Any | None = None      # dist.ProcessGroup

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def single_gpu(cls) -> ParallelContext:
        """Single-GPU context. No torch.distributed calls; safe without NCCL."""
        return cls()

    @classmethod
    def from_env(cls, cfg: ParallelSection) -> ParallelContext:
        """Initialize the data-parallel process group from torchrun env vars.

        Expects LOCAL_RANK / RANK / WORLD_SIZE to be set by the launcher.
        Builds a single-dimension (dp,) DeviceMesh; all ranks form one
        data-parallel group.
        """
        import torch.distributed as dist

        backend = str(getattr(cfg, "backend", "nccl"))
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = dist.get_world_size()

        dp = int(getattr(cfg, "dp", 1))
        force_cpu = bool(getattr(cfg, "force_cpu", False))

        if dp != world_size:
            raise ValueError(
                f"dp({dp}) != world_size({world_size}). "
                "Adjust parallel.dp so it equals the total GPU count."
            )

        # Build a 1-D DeviceMesh over the data-parallel dimension.
        # When force_cpu=True, skip the CUDA DeviceMesh entirely and use a plain
        # process group so gloo+CPU runs work without any CUDA context.
        try:
            if force_cpu:
                raise RuntimeError("force_cpu=True: using manual process group (no CUDA mesh)")
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cuda", (dp,), mesh_dim_names=("dp",))
            dp_group = mesh.get_group("dp")
            dp_rank = mesh.get_local_rank("dp")
        except Exception:  # noqa: BLE001
            # Fallback for older PyTorch or force_cpu=True
            _log.warning("parallel context: CUDA DeviceMesh init failed; falling back to a manual process group", exc_info=True)
            mesh = None
            dp_rank = rank
            dp_group = dist.new_group(list(range(dp)))

        return cls(
            rank=rank, local_rank=local_rank, world_size=world_size,
            dp_rank=dp_rank, dp_degree=dp,
            force_cpu=force_cpu,
            device_mesh=mesh,
            dp_group=dp_group,
        )

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def local_device(self) -> torch.device:
        if self.force_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        """True only on global rank 0 (the process that owns logs/checkpoints)."""
        return self.rank == 0

    @property
    def is_dp_rank0(self) -> bool:
        """True on the rank-0 replica of the data-parallel group."""
        return self.dp_rank == 0

    def __repr__(self) -> str:
        return (
            f"ParallelContext(rank={self.rank}/{self.world_size}, "
            f"dp={self.dp_rank}/{self.dp_degree})"
        )


__all__ = ["ParallelContext"]
