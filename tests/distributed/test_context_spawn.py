"""Real multi-process integration tests for ``ParallelContext.from_env``.

Spawns worker processes via ``torch.multiprocessing.spawn`` over the ``gloo``
CPU backend so the data-parallel manual fallback path is exercised end-to-end
(DeviceMesh requires CUDA). ``cfg.force_cpu=True`` bypasses the CUDA DeviceMesh
branch so the manual ``dist.new_group`` path actually runs.

Marked ``@pytest.mark.slow``: skip with ``pytest -m 'not slow'`` for fast
local iteration.

Workers boot a process group, construct ``ParallelContext.from_env(cfg)``,
perform a collective over ``dp_group`` (or pin the broadcast run dir), and
write per-rank results to files the parent test asserts on.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- #
# Worker entry points                                                         #
# (Must be top-level so torch.multiprocessing.spawn can pickle them.)         #
#                                                                             #
# We use ``init_method="file://..."`` rather than TCP rendezvous so that      #
# consecutive tests cannot collide on a TCPStore port that lingers in         #
# TIME_WAIT after the previous test's process group teardown.                 #
# --------------------------------------------------------------------------- #


def _init_dist(rank: int, world_size: int, rendezvous_file: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )


def _worker_dp2_broadcast(rank: int, world_size: int, rendezvous_file: str, out_dir: str) -> None:
    """2-proc DP=2: rank-0 broadcasts a known tensor; rank-1 must receive it.

    Each rank writes the received tensor to ``out_dir/rank{rank}.pt`` so the
    parent process can compare against the analytical expectation.
    """
    from lighttrain.distributed import ParallelContext

    _init_dist(rank, world_size, rendezvous_file)
    cfg = SimpleNamespace(backend="gloo", dp=2, force_cpu=True)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    if rank == 0:
        # Known payload that rank-1 cannot produce locally.
        t = torch.tensor([10.0, 20.0, 30.0])
    else:
        t = torch.zeros(3)

    dist.broadcast(t, src=0, group=ctx.dp_group)

    info = {
        "rank": ctx.rank,
        "dp_rank": ctx.dp_rank,
        "world_size": ctx.world_size,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(info), encoding="utf-8")
    torch.save(t, Path(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _worker_run_dir_agreement(rank: int, world_size: int, rendezvous_file: str, out_dir: str) -> None:
    """All ranks must end up with rank-0's run dir, never their own timestamp.

    ``factory`` returns a ``datetime.now()`` path suffixed with the rank, and
    rank 0 sleeps past a one-second boundary first — so a naive per-rank path
    would diverge on both the timestamp *and* the suffix. ``broadcast_run_dir``
    must make every rank report rank 0's path (``-r0``).
    """
    from lighttrain.utils.run_dir import broadcast_run_dir

    _init_dist(rank, world_size, rendezvous_file)

    def factory() -> Path:
        if rank == 0:
            time.sleep(1.2)  # force a cross-second skew to amplify the race
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path(out_dir) / f"run-{ts}-r{rank}"

    run_dir = broadcast_run_dir(
        factory,
        world_size=world_size,
        is_main=(rank == 0),
        device=torch.device("cpu"),
    )
    Path(out_dir, f"rank{rank}.json").write_text(
        json.dumps({"run_dir": str(run_dir)}), encoding="utf-8"
    )
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Test harness                                                                #
# --------------------------------------------------------------------------- #


def _spawn(target, world_size: int, out_dir: Path) -> None:
    # Use a fresh file-based rendezvous per spawn — no TCP port collisions
    # between consecutive tests.
    rendezvous = out_dir / "rendezvous.lock"
    if rendezvous.exists():
        rendezvous.unlink()
    mp.spawn(
        target,
        args=(world_size, str(rendezvous), str(out_dir)),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def test_spawn_2procs_dp2_broadcast(tmp_path: Path) -> None:
    """Both ranks see the broadcast tensor [10,20,30] after rank-0 sends it.

    Input: 2 procs, dp=2; rank-0 emits ``[10,20,30]``, rank-1 starts with
    zeros. Analytical: post-broadcast rank-1's tensor must equal the source.
    """
    _spawn(_worker_dp2_broadcast, 2, tmp_path)
    t0 = torch.load(tmp_path / "rank0.pt")
    t1 = torch.load(tmp_path / "rank1.pt")
    expected = torch.tensor([10.0, 20.0, 30.0])
    torch.testing.assert_close(t0, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(t1, expected, atol=1e-5, rtol=1e-4)
    # Rank arithmetic sanity.
    info0 = json.loads((tmp_path / "rank0.json").read_text(encoding="utf-8"))
    info1 = json.loads((tmp_path / "rank1.json").read_text(encoding="utf-8"))
    assert info0["dp_rank"] == 0
    assert info1["dp_rank"] == 1


def test_regression_run_dir_agreement_spawn_4procs(tmp_path: Path) -> None:
    """All 4 ranks share rank-0's run dir despite a forced timestamp skew.

    Pins the run-dir timestamp race: ``make_run_dir`` stamps
    ``datetime.now()`` per rank, so a launch crossing a one-second boundary
    used to split ranks across sibling dirs (same config hash, different
    ``HHMMSS``). ``broadcast_run_dir`` has rank 0 own the path and broadcast it.

    Input: 4 procs; ``factory`` returns ``run-<ts>-r<rank>`` and rank 0 sleeps
    1.2s first. Pre-fix every rank keeps its own ``-r<rank>`` path → 4 distinct
    dirs. Post-fix all report rank 0's ``-r0`` path → exactly one dir.
    """
    _spawn(_worker_run_dir_agreement, 4, tmp_path)
    dirs = [
        json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8"))["run_dir"]
        for r in range(4)
    ]
    assert len(set(dirs)) == 1, f"ranks disagreed on run_dir: {dirs}"
    assert dirs[0].endswith("-r0"), f"expected rank-0's path, got {dirs[0]}"
