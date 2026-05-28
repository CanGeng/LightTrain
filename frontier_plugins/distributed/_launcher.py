"""Distributed launcher helpers.

For torchrun / DeepSpeed launching, these functions validate the topology
and emit helpful error messages before torch.distributed initialises.

Usage (in a training script, not usually needed with lighttrain CLI):

    from frontier_plugins.distributed._launcher import validate_topology, launch_torchrun

    validate_topology(dp=4, tp=2, pp=2, world_size=16)
    # → prints: "Topology OK: 4×2×2=16 GPUs"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def validate_topology(
    dp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    world_size: int | None = None,
) -> None:
    """Raise ValueError if dp × tp × pp ≠ world_size (when world_size is given)."""
    product = dp * tp * pp
    if world_size is not None and product != world_size:
        raise ValueError(
            f"Topology error: dp({dp}) × tp({tp}) × pp({pp}) = {product} "
            f"≠ world_size({world_size}). "
            "Adjust parallel.dp/tp/pp so their product equals the total GPU count."
        )
    if ep > 1 and ep > dp:
        raise ValueError(
            f"ep({ep}) must be ≤ dp({dp}): EP groups are sub-groups of the DP dimension."
        )
    if ep > 1 and dp % ep != 0:
        raise ValueError(f"ep({ep}) must divide dp({dp}) evenly.")


def get_world_size_from_env() -> int:
    """Read WORLD_SIZE from environment (set by torchrun)."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank_from_env() -> int:
    """Read LOCAL_RANK from environment (set by torchrun)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def launch_torchrun(
    config_path: str | Path,
    *,
    nproc_per_node: int,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    extra_overrides: list[str] | None = None,
) -> int:
    """Programmatically launch ``torchrun lighttrain.cli train`` (subprocess).

    Returns the subprocess return code.  Raises RuntimeError if torchrun
    is not on PATH.  Prefer the shell command for production launches.
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m", "lighttrain.cli",
        "train",
        "-c", str(config_path),
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    result = subprocess.run(cmd)
    return result.returncode


__all__ = ["validate_topology", "get_world_size_from_env", "get_local_rank_from_env", "launch_torchrun"]
