"""Adversarial tests for ``lighttrain.distributed._context``.

Targets that the legacy ``tests/test_distributed_context.py`` misses:
  * Off-by-one in ``_compute_ranks`` for non-power-of-2 mesh shapes
  * Exact group membership produced by ``_create_groups_manual`` (the
    legacy test only exercises ``_compute_ranks``)
  * Equivalence between the DeviceMesh code path and the manual fallback
  * The P0 historical bug ``DIST_EP_01``: ``_create_ep_groups`` must call
    ``dist.new_group`` with **global** ranks from ``get_process_group_ranks``,
    not DP-positional indices (see docs/changelog/v0.1.4).

Numeric assertions are not relevant here (these are integer rank-arithmetic
tests), so ``torch.testing.assert_close`` is not used in this file. Instead
exact integer/list equality is the right comparator.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from lighttrain.distributed._context import (
    ParallelContext,
    _compute_ranks,
    _create_ep_groups,
    _create_groups_manual,
)


# --------------------------------------------------------------------------- #
# _compute_ranks: bijection over (rank → dp_r, tp_r, pp_r)                    #
# --------------------------------------------------------------------------- #


_MESH_PARAMS = [
    (dp, tp, pp)
    for dp in (1, 2, 3, 4)
    for tp in (1, 2, 3)
    for pp in (1, 2)
]


@pytest.mark.parametrize("dp,tp,pp", _MESH_PARAMS)
def test_compute_ranks_parametrized(dp: int, tp: int, pp: int) -> None:
    """``rank → (dp_r, tp_r, pp_r)`` matches the documented layout formula.

    Input: every global rank in ``range(dp*tp*pp)`` for each non-trivial mesh
    shape including non-power-of-2 axes (dp=3, tp=3) where off-by-one would
    surface.

    Analytical reference:
        ``rank = dp_r * (tp*pp) + tp_r * pp + pp_r``
        ``dp_r = rank // (tp*pp)``
        ``tp_r = (rank % (tp*pp)) // pp``
        ``pp_r = rank % pp``
    """
    for dp_r, tp_r, pp_r in itertools.product(range(dp), range(tp), range(pp)):
        rank = dp_r * (tp * pp) + tp_r * pp + pp_r
        assert _compute_ranks(rank, dp, tp, pp) == (dp_r, tp_r, pp_r), (
            f"mesh=({dp},{tp},{pp}) rank={rank}"
        )


@pytest.mark.parametrize("dp,tp,pp", _MESH_PARAMS)
def test_compute_ranks_inverse_is_bijection(dp: int, tp: int, pp: int) -> None:
    """Every rank maps to a unique (dp_r, tp_r, pp_r) triple, all in range.

    Input: ``range(dp*tp*pp)``. Analytical: there are exactly ``dp*tp*pp``
    unique triples in the cartesian product, so the mapping is a bijection.
    """
    seen: set[tuple[int, int, int]] = set()
    for rank in range(dp * tp * pp):
        triple = _compute_ranks(rank, dp, tp, pp)
        assert triple not in seen, f"duplicate triple {triple} at rank={rank}"
        seen.add(triple)
        dp_r, tp_r, pp_r = triple
        assert 0 <= dp_r < dp
        assert 0 <= tp_r < tp
        assert 0 <= pp_r < pp
    assert len(seen) == dp * tp * pp


# --------------------------------------------------------------------------- #
# _create_groups_manual: pin exact group memberships                          #
# --------------------------------------------------------------------------- #


def _dp_members_for(dp: int, tp: int, pp: int, tp_r: int, pp_r: int) -> list[int]:
    return [dp_r * tp * pp + tp_r * pp + pp_r for dp_r in range(dp)]


def _tp_members_for(dp: int, tp: int, pp: int, dp_r: int, pp_r: int) -> list[int]:
    return [dp_r * tp * pp + tp_r * pp + pp_r for tp_r in range(tp)]


def _pp_members_for(dp: int, tp: int, pp: int, dp_r: int, tp_r: int) -> list[int]:
    return [dp_r * tp * pp + tp_r * pp + pp_r for pp_r in range(pp)]


_GROUP_MEM_PARAMS = [(2, 2, 2), (3, 2, 1), (4, 2, 1), (2, 1, 2), (1, 2, 3)]


@pytest.mark.parametrize("dp,tp,pp", _GROUP_MEM_PARAMS)
def test_create_groups_manual_dp_membership(
    dp: int, tp: int, pp: int, dist_mock
) -> None:
    """Every DP group contains exactly the ranks with the same (tp_r, pp_r).

    Input: rank=0, world=dp*tp*pp; capture ``new_group`` calls.
    Analytical: ``_create_groups_manual`` iterates ``(tp_r, pp_r)`` and emits
    one DP group per combination → ``tp*pp`` DP groups; members are
    ``[dp_r * tp * pp + tp_r * pp + pp_r for dp_r in range(dp)]``.
    """
    handle = dist_mock(rank=0, world_size=dp * tp * pp)
    _create_groups_manual(dp, tp, pp, rank=0)
    # _create_groups_manual emits DP, TP, PP groups in that order. Slice them.
    dp_calls = handle.calls[: tp * pp]
    expected_dp = [
        _dp_members_for(dp, tp, pp, tp_r, pp_r)
        for tp_r in range(tp)
        for pp_r in range(pp)
    ]
    assert dp_calls == expected_dp


@pytest.mark.parametrize("dp,tp,pp", _GROUP_MEM_PARAMS)
def test_create_groups_manual_tp_membership(
    dp: int, tp: int, pp: int, dist_mock
) -> None:
    """Every TP group contains exactly the ranks with the same (dp_r, pp_r)."""
    handle = dist_mock(rank=0, world_size=dp * tp * pp)
    _create_groups_manual(dp, tp, pp, rank=0)
    tp_offset = tp * pp
    tp_calls = handle.calls[tp_offset : tp_offset + dp * pp]
    expected_tp = [
        _tp_members_for(dp, tp, pp, dp_r, pp_r)
        for dp_r in range(dp)
        for pp_r in range(pp)
    ]
    assert tp_calls == expected_tp


@pytest.mark.parametrize("dp,tp,pp", _GROUP_MEM_PARAMS)
def test_create_groups_manual_pp_membership(
    dp: int, tp: int, pp: int, dist_mock
) -> None:
    """Every PP group contains exactly the ranks with the same (dp_r, tp_r)."""
    handle = dist_mock(rank=0, world_size=dp * tp * pp)
    _create_groups_manual(dp, tp, pp, rank=0)
    pp_offset = tp * pp + dp * pp
    pp_calls = handle.calls[pp_offset : pp_offset + dp * tp]
    expected_pp = [
        _pp_members_for(dp, tp, pp, dp_r, tp_r)
        for dp_r in range(dp)
        for tp_r in range(tp)
    ]
    assert pp_calls == expected_pp


def test_create_groups_manual_partitions_world(dist_mock) -> None:
    """Union of DP groups passing through every rank covers ``range(world)`` once.

    Input: dp=2,tp=2,pp=2 → world=8. Each rank r belongs to exactly one DP
    group (those sharing its (tp_r, pp_r)). Across all DP groups, every rank
    must appear exactly ``1`` time per dimension.
    """
    dp, tp, pp = 2, 2, 2
    handle = dist_mock(rank=0, world_size=dp * tp * pp)
    _create_groups_manual(dp, tp, pp, rank=0)
    dp_calls = handle.calls[: tp * pp]
    flat = sorted([r for members in dp_calls for r in members])
    assert flat == sorted(range(dp * tp * pp))


# --------------------------------------------------------------------------- #
# DIST_EP_01 — regression for the local-vs-global rank bug in EP groups       #
# --------------------------------------------------------------------------- #


def test_regression_DIST_EP_01_global_ranks_in_new_group(dist_mock) -> None:
    """Pre-fix bug: ``_create_ep_groups`` passed local DP-positional indices to
    ``dist.new_group`` instead of the real global ranks returned by
    ``get_process_group_ranks(dp_group)``. Under ``dp_degree>1, ep>1`` the
    resulting EP groups had wrong members (see docs/changelog/v0.1.4:
    'EP 组用局部 rank 调用 dist.new_group').

    Input: dp_group with global ranks [2,3,6,7], my dp_rank=2, ep=2, caller
    global rank=6.

    Analytical solution:
        ep_rank      = dp_rank % ep         = 2 % 2 = 0  (still local DP rank)
        ep_partition = [[2,3], [6,7]]                    (slices of length ep)
        my chunk     = [6,7] (because my global rank 6 ∈ [6,7])

    A pre-fix implementation would have called ``new_group([0,1])`` and
    ``new_group([2,3])`` (treating positional indices as ranks). Post-fix,
    the calls must use the global ranks ``[2,3]`` and ``[6,7]``.
    """
    handle = dist_mock(
        rank=6,
        world_size=8,
        dp_group_global_ranks=[2, 3, 6, 7],
    )
    ep_rank, ep_group = _create_ep_groups(dp_rank=2, ep=2, dp_group=handle.dp_group)

    # Local-DP-rank arithmetic is unchanged by the fix.
    assert ep_rank == 0

    # The critical pin: new_group was called with GLOBAL ranks, in order.
    assert handle.calls == [[2, 3], [6, 7]], (
        "Pre-fix would emit local positional indices like [[0,1],[2,3]]; "
        f"got {handle.calls}"
    )

    # The returned ep_group must be the second one (since my global rank=6 ∈ [6,7]).
    assert ep_group is handle.new_group_returns[1]


def test_invariant_create_ep_groups_ep_eq_one(dist_mock) -> None:
    """``ep=1`` is the degenerate path: each EP group has a single member.

    Contract: returns ``ep_rank=0`` and an ep_group object without raising;
    every group emitted is a singleton.
    """
    handle = dist_mock(rank=2, world_size=4, dp_group_global_ranks=[0, 1, 2, 3])
    ep_rank, ep_group = _create_ep_groups(dp_rank=2, ep=1, dp_group=handle.dp_group)
    assert ep_rank == 0
    assert handle.calls == [[0], [1], [2], [3]]
    # My global rank is 2 → I belong to the singleton group [2] (3rd call).
    assert ep_group is handle.new_group_returns[2]


def test_create_ep_groups_dp_group_none_returns_zero(dist_mock) -> None:
    """If ``dp_group`` is None (no DP), return ``(0, None)`` without any dist call.

    Contract: degenerate path (no DP) yields trivial EP topology.
    """
    handle = dist_mock(rank=0, world_size=1)
    ep_rank, ep_group = _create_ep_groups(dp_rank=0, ep=2, dp_group=None)
    assert ep_rank == 0
    assert ep_group is None
    assert handle.calls == []  # never even touched dist


@pytest.mark.parametrize("dp,ep", [(2, 2), (4, 2), (6, 2), (4, 1), (2, 1)])
def test_create_ep_groups_membership_partition(dp: int, ep: int, dist_mock) -> None:
    """All ``new_group`` calls partition the DP group's global ranks into chunks of ``ep``.

    Input: dp_group global ranks = ``range(dp)`` (i.e. DP is the leading
    dimension of the mesh), tp=pp=1. Caller dp_rank=0.

    Analytical: ``_create_ep_groups`` slices ``dp_global_ranks`` into
    contiguous chunks of length ``ep``. There are ``dp // ep`` chunks
    (assumes ``ep | dp``); their union equals the input ranks and they are
    pairwise disjoint.
    """
    dp_globals = list(range(dp))
    handle = dist_mock(rank=0, world_size=dp, dp_group_global_ranks=dp_globals)
    _create_ep_groups(dp_rank=0, ep=ep, dp_group=handle.dp_group)
    expected = [dp_globals[s : s + ep] for s in range(0, dp, ep)]
    assert handle.calls == expected
    flat = sorted(r for chunk in handle.calls for r in chunk)
    assert flat == sorted(dp_globals)


# --------------------------------------------------------------------------- #
# DeviceMesh path vs manual fallback equivalence                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dp,tp,pp", [(2, 2, 2), (4, 1, 1), (2, 2, 1)])
def test_invariant_devicemesh_manual_fallback_equivalent(
    dp: int, tp: int, pp: int, dist_mock
) -> None:
    """Manual fallback emits the same group memberships as ``init_device_mesh`` would.

    Contract: the two code paths in ``ParallelContext.from_env`` (DeviceMesh
    vs ``_create_groups_manual``) must agree on which ranks belong to which
    DP/TP/PP group, otherwise switching between them silently changes
    collective semantics.

    We simulate the DeviceMesh by constructing the analytical group
    partitions (the formula DeviceMesh would use for a row-major
    ``(dp, tp, pp)`` mesh with the same layout) and compare to what
    ``_create_groups_manual`` actually emits.
    """
    handle = dist_mock(rank=0, world_size=dp * tp * pp)
    _create_groups_manual(dp, tp, pp, rank=0)

    # Manual emits DP, then TP, then PP. Build the DeviceMesh-equivalent
    # partitions from the layout formula and compare.
    dp_calls = handle.calls[: tp * pp]
    tp_calls = handle.calls[tp * pp : tp * pp + dp * pp]
    pp_calls = handle.calls[tp * pp + dp * pp :]

    expected_dp = [
        _dp_members_for(dp, tp, pp, tp_r, pp_r)
        for tp_r in range(tp)
        for pp_r in range(pp)
    ]
    expected_tp = [
        _tp_members_for(dp, tp, pp, dp_r, pp_r)
        for dp_r in range(dp)
        for pp_r in range(pp)
    ]
    expected_pp = [
        _pp_members_for(dp, tp, pp, dp_r, tp_r)
        for dp_r in range(dp)
        for tp_r in range(tp)
    ]

    assert dp_calls == expected_dp
    assert tp_calls == expected_tp
    assert pp_calls == expected_pp


# --------------------------------------------------------------------------- #
# single_gpu purity + force_cpu dominance + mismatch guard                    #
# --------------------------------------------------------------------------- #


def test_single_gpu_is_pure_no_dist_calls(monkeypatch) -> None:
    """``single_gpu()`` must not touch ``torch.distributed`` at all.

    Contract: replacing every dist symbol with a raising stub must not affect
    construction of the single-GPU context — single-GPU users without NCCL
    rely on this purity.
    """
    def _boom(*_a, **_kw):
        raise AssertionError("single_gpu() must not touch torch.distributed")

    for name in (
        "init_process_group",
        "is_initialized",
        "get_rank",
        "get_world_size",
        "new_group",
    ):
        monkeypatch.setattr(f"torch.distributed.{name}", _boom)

    ctx = ParallelContext.single_gpu()
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.device_mesh is None
    assert ctx.dp_group is None
    assert ctx.tp_group is None
    assert ctx.pp_group is None
    assert ctx.ep_group is None


@pytest.mark.parametrize("local_rank", [0, 1, 7])
def test_local_device_force_cpu_dominates(local_rank: int) -> None:
    """``force_cpu=True`` always returns CPU regardless of CUDA / local_rank.

    Contract: the force_cpu kill-switch must short-circuit any CUDA selection.
    Pre-checked: rank 0/1/7 all hit the same branch.
    """
    ctx = ParallelContext(force_cpu=True, local_rank=local_rank)
    assert ctx.local_device == torch.device("cpu")


def test_dp_times_tp_times_pp_mismatch_raises(monkeypatch) -> None:
    """``from_env`` raises ``ValueError`` when ``dp*tp*pp != world_size``.

    Input: dp=2, tp=2, pp=2 but world_size=7 → product 8 ≠ 7.
    Contract: misconfigured launchers fail loud, not silently degrade.
    """
    cfg = SimpleNamespace(
        backend="gloo", dp=2, tp=2, pp=2, ep=1, sp=False, force_cpu=True
    )

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **_kw: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 7)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    with pytest.raises(ValueError, match="world_size"):
        ParallelContext.from_env(cfg)
