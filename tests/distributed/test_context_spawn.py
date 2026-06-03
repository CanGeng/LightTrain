"""Real multi-process integration tests for ``ParallelContext.from_env``.

Spawns 2–4 worker processes via ``torch.multiprocessing.spawn`` over the
``gloo`` CPU backend so that the manual fallback path in ``_create_groups_manual``
and ``_create_ep_groups`` is exercised end-to-end (DeviceMesh requires CUDA).

Marked ``@pytest.mark.slow``: skip with ``pytest -m 'not slow'`` for fast
local iteration.

Each test boots N processes that:
  1. Call ``dist.init_process_group(backend="gloo", ...)`` with the
     ``LOCAL_RANK``/``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``
     env vars set by the harness.
  2. Construct ``ParallelContext.from_env(cfg)`` and write its observable
     fields to a shared per-rank file. ``cfg.force_cpu=True`` bypasses the
     CUDA DeviceMesh branch so the manual helpers actually run.
  3. Perform a collective (broadcast / all_reduce) over the resulting
     ``dp_group`` / ``ep_group`` whose result is predictable from rank
     arithmetic.

The parent test then reads the per-rank files and asserts the collective
result + the rank arithmetic.
"""
from __future__ import annotations

import json
import os
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
    cfg = SimpleNamespace(
        backend="gloo", dp=2, tp=1, pp=1, ep=1, sp=False, force_cpu=True
    )
    ctx = ParallelContext.from_env(cfg)

    if rank == 0:
        # Known payload that rank-1 cannot produce locally.
        t = torch.tensor([10.0, 20.0, 30.0])
    else:
        t = torch.zeros(3)

    dist.broadcast(t, src=0, group=ctx.dp_group)

    info = {
        "rank": ctx.rank,
        "dp_rank": ctx.dp_rank,
        "tp_rank": ctx.tp_rank,
        "pp_rank": ctx.pp_rank,
        "world_size": ctx.world_size,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(info), encoding="utf-8")
    torch.save(t, Path(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def _worker_dp2_tp2(rank: int, world_size: int, rendezvous_file: str, out_dir: str) -> None:
    """4-proc dp=2 tp=2: assert (dp_rank, tp_rank) matches the layout formula.

    The layout is ``rank = dp_r * (tp*pp) + tp_r * pp + pp_r`` with pp=1, so
    ``dp_rank = rank // 2`` and ``tp_rank = rank % 2``.
    """
    from lighttrain.distributed import ParallelContext

    _init_dist(rank, world_size, rendezvous_file)
    cfg = SimpleNamespace(
        backend="gloo", dp=2, tp=2, pp=1, ep=1, sp=False, force_cpu=True
    )
    ctx = ParallelContext.from_env(cfg)
    info = {
        "rank": ctx.rank,
        "dp_rank": ctx.dp_rank,
        "tp_rank": ctx.tp_rank,
        "pp_rank": ctx.pp_rank,
        "world_size": ctx.world_size,
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(info), encoding="utf-8")
    dist.destroy_process_group()


def _worker_dp4_ep2(rank: int, world_size: int, rendezvous_file: str, out_dir: str) -> None:
    """4-proc dp=4 ep=2: ``ep_group`` all_reduce sums must match the analytical answer.

    With dp=4, ep=2 the DP global ranks are [0,1,2,3]. ``_create_ep_groups``
    slices them into ``[[0,1], [2,3]]``. ep_rank = dp_rank % 2 (= rank % 2
    here, because tp=pp=1 → dp_rank == rank).

    Each rank contributes ``rank+1`` to an all_reduce on the ep_group, so
    the sums must be ``1+2 = 3`` for the {0,1} group and ``3+4 = 7`` for
    {2,3}. Pre-fix (DIST_EP_01) the EP groups would have been mis-formed
    (using local positional indices passed to ``new_group``) which would
    yield a different rank-pairing — for any wrong pairing the sum would
    differ from {3, 3, 7, 7}.
    """
    from lighttrain.distributed import ParallelContext

    _init_dist(rank, world_size, rendezvous_file)
    cfg = SimpleNamespace(
        backend="gloo", dp=4, tp=1, pp=1, ep=2, sp=False, force_cpu=True
    )
    ctx = ParallelContext.from_env(cfg)

    contribution = torch.tensor([float(rank + 1)])
    dist.all_reduce(contribution, op=dist.ReduceOp.SUM, group=ctx.ep_group)

    info = {
        "rank": ctx.rank,
        "dp_rank": ctx.dp_rank,
        "ep_rank": ctx.ep_rank,
        "ep_sum": float(contribution.item()),
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(info), encoding="utf-8")
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


def test_spawn_4procs_dp2_tp2_local_ranks(tmp_path: Path) -> None:
    """Each rank's ``(dp_rank, tp_rank)`` matches the analytical layout formula.

    Input: 4 procs, dp=2 tp=2 pp=1. Analytical:
        rank 0 → (dp=0, tp=0)
        rank 1 → (dp=0, tp=1)
        rank 2 → (dp=1, tp=0)
        rank 3 → (dp=1, tp=1)
    """
    _spawn(_worker_dp2_tp2, 4, tmp_path)
    expected = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    for r in range(4):
        info = json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8"))
        assert (info["dp_rank"], info["tp_rank"]) == expected[r], (
            f"rank {r} got dp={info['dp_rank']}, tp={info['tp_rank']}"
        )


def test_regression_DIST_EP_01_spawn_4procs_dp4_ep2_real_groups(tmp_path: Path) -> None:
    """Integration-level pin of DIST_EP_01 over real gloo processes.

    Pre-fix bug: ``_create_ep_groups`` used DP-positional indices instead of
    real global ranks when calling ``dist.new_group``; ``dp_degree>1,ep>1``
    EP groups would have wrong members and the all_reduce sums would not
    match the analytical {3, 3, 7, 7} pattern (see docs/changelog/v0.1.4:
    'EP 组用局部 rank 调用 dist.new_group').

    Input: 4 procs, dp=4 ep=2. Each rank ``r`` contributes ``r+1`` to an
    all_reduce summed over its ep_group.

    Analytical solution:
        EP groups (post-fix)  = {0,1}, {2,3}
        sum for {0,1}         = 1+2 = 3
        sum for {2,3}         = 3+4 = 7
        ep_rank               = rank % 2 (because dp_rank==rank here)

    Verifies rank 0 and 1 see sum=3, rank 2 and 3 see sum=7.
    """
    _spawn(_worker_dp4_ep2, 4, tmp_path)
    sums = {}
    ep_ranks = {}
    for r in range(4):
        info = json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8"))
        sums[r] = info["ep_sum"]
        ep_ranks[r] = info["ep_rank"]

    torch.testing.assert_close(
        torch.tensor([sums[0], sums[1]]),
        torch.tensor([3.0, 3.0]),
        atol=1e-5, rtol=1e-4,
    )
    torch.testing.assert_close(
        torch.tensor([sums[2], sums[3]]),
        torch.tensor([7.0, 7.0]),
        atol=1e-5, rtol=1e-4,
    )
    # ep_rank = dp_rank % ep ; here dp_rank == rank
    assert ep_ranks == {0: 0, 1: 1, 2: 0, 3: 1}
