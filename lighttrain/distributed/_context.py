"""ParallelContext — distributed topology carrier.

Always present: single-GPU is ``ParallelContext.single_gpu()`` with all
degrees == 1 and rank == 0. Callers that don't know about distributed can
safely read ``parallel_ctx.is_main_process`` (always True on single-GPU)
and ``parallel_ctx.local_device`` (cuda:0 or cpu).

``torch.distributed`` is lazily imported: ``ParallelContext.single_gpu()``
works without NCCL or any dist initialization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..config._schema import ParallelSection


@dataclass
class ParallelContext:
    # Global identity
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    # Per-dimension ranks (all 0 for single-GPU)
    dp_rank: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    ep_rank: int = 0

    # Per-dimension sizes (all 1 for single-GPU)
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1

    sp_enabled: bool = False
    force_cpu: bool = False

    # torch.distributed objects — None on single-GPU
    device_mesh: Any | None = None   # torch.distributed.DeviceMesh
    dp_group: Any | None = None      # dist.ProcessGroup
    tp_group: Any | None = None
    pp_group: Any | None = None
    ep_group: Any | None = None

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def single_gpu(cls) -> "ParallelContext":
        """Single-GPU context. No torch.distributed calls; safe without NCCL."""
        return cls()

    @classmethod
    def from_env(cls, cfg: "ParallelSection") -> "ParallelContext":
        """Initialize process groups from torchrun environment variables.

        Expects LOCAL_RANK / RANK / WORLD_SIZE to be set by the launcher.
        Builds a DeviceMesh with dimensions (dp, tp, pp) and derives
        per-dimension process groups from it.
        """
        import torch.distributed as dist

        backend = str(getattr(cfg, "backend", "nccl"))
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = dist.get_world_size()

        dp = int(getattr(cfg, "dp", 1))
        tp = int(getattr(cfg, "tp", 1))
        pp = int(getattr(cfg, "pp", 1))
        ep = int(getattr(cfg, "ep", 1))
        sp = bool(getattr(cfg, "sp", False))
        force_cpu = bool(getattr(cfg, "force_cpu", False))

        if dp * tp * pp != world_size:
            raise ValueError(
                f"dp({dp}) × tp({tp}) × pp({pp}) = {dp*tp*pp} "
                f"!= world_size({world_size}). "
                "Adjust parallel.dp/tp/pp so their product equals the total GPU count."
            )

        # Build DeviceMesh with named dimensions.
        # When force_cpu=True, skip CUDA DeviceMesh entirely and fall directly to
        # _create_groups_manual so gloo+CPU runs work without any CUDA context.
        try:
            if force_cpu:
                raise RuntimeError("force_cpu=True: using manual process groups (no CUDA mesh)")
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(
                "cuda",
                (dp, tp, pp),
                mesh_dim_names=("dp", "tp", "pp"),
            )
            dp_group = mesh.get_group("dp")
            tp_group = mesh.get_group("tp")
            pp_group = mesh.get_group("pp")
            dp_rank = mesh.get_local_rank("dp")
            tp_rank = mesh.get_local_rank("tp")
            pp_rank = mesh.get_local_rank("pp")
        except Exception:
            # Fallback for older PyTorch or force_cpu=True
            mesh = None
            dp_rank, tp_rank, pp_rank = _compute_ranks(rank, dp, tp, pp)
            dp_group, tp_group, pp_group = _create_groups_manual(dp, tp, pp, rank)

        # EP groups are formed as sub-groups within the DP dimension
        ep_rank = 0
        ep_group = None
        if ep > 1:
            ep_rank, ep_group = _create_ep_groups(dp_rank, ep, dp_group)

        return cls(
            rank=rank, local_rank=local_rank, world_size=world_size,
            dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank,
            dp_degree=dp, tp_degree=tp, pp_degree=pp, ep_degree=ep,
            sp_enabled=sp,
            force_cpu=force_cpu,
            device_mesh=mesh,
            dp_group=dp_group, tp_group=tp_group,
            pp_group=pp_group, ep_group=ep_group,
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
        """True on the rank-0 replica within each TP×PP local group.

        Used by checkpoint writers that need one writer per TP/PP group
        but not necessarily global rank 0.
        """
        return self.dp_rank == 0

    @property
    def is_pp_last_stage(self) -> bool:
        """True on the pipeline stage that holds the final layers and the loss."""
        return self.pp_rank == self.pp_degree - 1

    def __repr__(self) -> str:
        return (
            f"ParallelContext(rank={self.rank}/{self.world_size}, "
            f"dp={self.dp_rank}/{self.dp_degree}, "
            f"tp={self.tp_rank}/{self.tp_degree}, "
            f"pp={self.pp_rank}/{self.pp_degree})"
        )


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _compute_ranks(
    rank: int, dp: int, tp: int, pp: int
) -> tuple[int, int, int]:
    """Map global rank to (dp_rank, tp_rank, pp_rank) for a (dp, tp, pp) mesh.

    Layout: rank = dp_rank * (tp * pp) + tp_rank * pp + pp_rank
    """
    tp_pp = tp * pp
    dp_rank = rank // tp_pp
    remainder = rank % tp_pp
    tp_rank = remainder // pp
    pp_rank = remainder % pp
    return dp_rank, tp_rank, pp_rank


def _create_groups_manual(
    dp: int, tp: int, pp: int, rank: int
) -> tuple[Any, Any, Any]:
    """Create DP/TP/PP process groups without DeviceMesh (PyTorch < 2.0 fallback)."""
    import torch.distributed as dist

    # DP groups: all ranks with same tp_rank and pp_rank
    dp_group = None
    for tp_r in range(tp):
        for pp_r in range(pp):
            members = [
                dp_r * tp * pp + tp_r * pp + pp_r
                for dp_r in range(dp)
            ]
            g = dist.new_group(members)
            if rank in members:
                dp_group = g

    # TP groups: all ranks with same dp_rank and pp_rank
    tp_group = None
    for dp_r in range(dp):
        for pp_r in range(pp):
            members = [
                dp_r * tp * pp + tp_r * pp + pp_r
                for tp_r in range(tp)
            ]
            g = dist.new_group(members)
            if rank in members:
                tp_group = g

    # PP groups: all ranks with same dp_rank and tp_rank
    pp_group = None
    for dp_r in range(dp):
        for tp_r in range(tp):
            members = [
                dp_r * tp * pp + tp_r * pp + pp_r
                for pp_r in range(pp)
            ]
            g = dist.new_group(members)
            if rank in members:
                pp_group = g

    return dp_group, tp_group, pp_group


def _create_ep_groups(dp_rank: int, ep: int, dp_group: Any) -> tuple[int, Any]:
    """Expert parallel groups are sub-groups of the DP group.

    ep_size must divide dp_degree. Each EP group has ep_size members.
    """
    import torch.distributed as dist

    if dp_group is None:
        return 0, None

    # dist.new_group requires GLOBAL ranks, not DP-local indices.
    dp_global_ranks = dist.get_process_group_ranks(dp_group)
    my_global_rank = dist.get_rank()
    ep_rank = dp_rank % ep
    ep_group = None
    for start in range(0, len(dp_global_ranks), ep):
        members = dp_global_ranks[start:start + ep]
        g = dist.new_group(members)
        if my_global_rank in members:
            ep_group = g
    return ep_rank, ep_group


__all__ = ["ParallelContext"]
