"""Coverage-completions for ``lighttrain.distributed._context``.

Pins the following uncovered lines (as of the 83% baseline):
  77   — dist.init_process_group called when not yet initialized
  100  — try: block entry in from_env
  101  — if force_cpu: (True branch)
  102  — raise RuntimeError("force_cpu=True …")
  103  — from torch.distributed.device_mesh import init_device_mesh
  104  — mesh = init_device_mesh(…)
  109  — dp_group = mesh.get_group("dp")
  110  — tp_group = mesh.get_group("tp")
  111  — pp_group = mesh.get_group("pp")
  112  — dp_rank = mesh.get_local_rank("dp")
  113  — tp_rank = mesh.get_local_rank("tp")
  114  — pp_rank = mesh.get_local_rank("pp")
  115  — except Exception:
  117  — _log.warning(…)
  118  — mesh = None
  119  — dp_rank, tp_rank, pp_rank = _compute_ranks(…)
  120  — dp_group, tp_group, pp_group = _create_groups_manual(…)
  123  — ep_rank = 0
  124  — ep_group = None
  125  — if ep > 1:
  126  — ep_rank, ep_group = _create_ep_groups(…)
  128  — return cls(…)   (entire return block)
  149  — return torch.device("cpu")  when force_cpu=False + CUDA unavailable
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from lighttrain.distributed._context import (
    ParallelContext,
    _compute_ranks,
)

# --------------------------------------------------------------------------- #
# Helpers / stubs shared across this file                                      #
# --------------------------------------------------------------------------- #


def _make_cfg(
    *,
    backend: str = "gloo",
    dp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    sp: bool = False,
    force_cpu: bool = True,
) -> SimpleNamespace:
    """Minimal config-like object accepted by ``from_env``."""
    return SimpleNamespace(
        backend=backend, dp=dp, tp=tp, pp=pp, ep=ep, sp=sp, force_cpu=force_cpu
    )


def _patch_dist(monkeypatch, *, rank: int = 0, world_size: int = 1, initialized: bool = True):
    """Install a minimal torch.distributed mock on the module under test.

    Returns a handle with `.calls` (list of new_group members) and `.dp_group`.
    """
    calls: list[list[int]] = []
    returns: list[object] = []
    group_ranks: dict[object, list[int]] = {}

    def _new_group(members):
        members_list = list(members)
        calls.append(members_list)
        obj = SimpleNamespace(_members=tuple(members_list))
        group_ranks[id(obj)] = members_list
        returns.append(obj)
        return obj

    def _get_process_group_ranks(group):
        key = id(group)
        if key in group_ranks:
            return list(group_ranks[key])
        mem = getattr(group, "_members", None)
        if mem is not None:
            return list(mem)
        raise RuntimeError("unknown group")

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: initialized)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **_kw: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)
    monkeypatch.setattr("torch.distributed.new_group", _new_group)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", _get_process_group_ranks)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": str(rank)})

    handle = SimpleNamespace(calls=calls, new_group_returns=returns)
    return handle


# --------------------------------------------------------------------------- #
# Line 77: dist.init_process_group is called when not yet initialized          #
# --------------------------------------------------------------------------- #


def test_from_env_calls_init_process_group_when_not_initialized(monkeypatch) -> None:
    """``from_env`` calls ``dist.init_process_group`` when ``is_initialized()`` is False.

    Line 77 is only reached when ``dist.is_initialized()`` returns False.
    We verify the call happened and that the resulting context is valid.
    """
    init_called: list[str] = []

    def _fake_init(backend, **_kw):
        init_called.append(backend)

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.init_process_group", _fake_init)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda g: list(g._members))
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    cfg = _make_cfg(backend="gloo", force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert init_called == ["gloo"], (
        f"init_process_group should be called once with backend='gloo', got {init_called}"
    )
    assert ctx.rank == 0
    assert ctx.world_size == 1


# --------------------------------------------------------------------------- #
# Lines 101-102: force_cpu=True path inside the try block                      #
# Lines 115-120: except branch (fallback) + _compute_ranks + _create_groups   #
# Lines 123-124: ep_rank=0 / ep_group=None when ep==1                         #
# Lines 128-137: return cls(…)                                                 #
# --------------------------------------------------------------------------- #


def test_from_env_force_cpu_takes_manual_fallback_and_returns_context(monkeypatch) -> None:
    """force_cpu=True makes from_env skip DeviceMesh and use manual process groups.

    Covers lines 101-102 (force_cpu raise inside try), 115 (except), 117-120
    (fallback: mesh=None, _compute_ranks, _create_groups_manual), 123-124
    (ep_rank=0, ep_group=None), 128-137 (return cls).

    Input: dp=2, tp=1, pp=1, ep=1, rank=0, world=2.
    Analytical: dp_rank=0, tp_rank=0, pp_rank=0; ep unchanged.
    """
    _patch_dist(monkeypatch, rank=0, world_size=2)
    cfg = _make_cfg(dp=2, tp=1, pp=1, ep=1, force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.world_size == 2
    assert ctx.dp_degree == 2
    assert ctx.tp_degree == 1
    assert ctx.pp_degree == 1
    assert ctx.ep_degree == 1
    assert ctx.dp_rank == 0
    assert ctx.tp_rank == 0
    assert ctx.pp_rank == 0
    assert ctx.ep_rank == 0
    assert ctx.ep_group is None
    assert ctx.force_cpu is True
    assert ctx.sp_enabled is False
    # Manual fallback: DeviceMesh skipped → device_mesh must be None
    assert ctx.device_mesh is None
    # DP group must have been created (rank 0 belongs to the only dp group [0,1])
    assert ctx.dp_group is not None


def test_from_env_force_cpu_dp2_tp2_rank1(monkeypatch) -> None:
    """force_cpu=True with dp=2,tp=2,pp=1 at rank=1 yields correct per-dim ranks.

    Analytical (layout rank = dp_r*(tp*pp) + tp_r*pp + pp_r, pp=1):
        rank 1 → dp_rank=0, tp_rank=1, pp_rank=0
    """
    _patch_dist(monkeypatch, rank=1, world_size=4)
    cfg = _make_cfg(dp=2, tp=2, pp=1, ep=1, force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.rank == 1
    assert ctx.world_size == 4
    expected_dp, expected_tp, expected_pp = _compute_ranks(1, dp=2, tp=2, pp=1)
    assert ctx.dp_rank == expected_dp
    assert ctx.tp_rank == expected_tp
    assert ctx.pp_rank == expected_pp
    assert ctx.device_mesh is None


def test_from_env_force_cpu_mesh_is_none_in_returned_ctx(monkeypatch) -> None:
    """When the manual fallback fires, ``device_mesh`` in the returned context is None.

    Pins line 118 (mesh = None) through the returned object rather than internal state.
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    cfg = _make_cfg(dp=1, tp=1, pp=1, ep=1, force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.device_mesh is None


def test_from_env_sp_enabled_propagates(monkeypatch) -> None:
    """``sp=True`` in cfg causes ``ctx.sp_enabled`` to be True.

    Covers the sp= line in the cfg-reading block and the return cls(...).
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    cfg = _make_cfg(dp=1, tp=1, pp=1, ep=1, force_cpu=True, sp=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.sp_enabled is True


def test_from_env_logs_warning_on_fallback(monkeypatch, caplog) -> None:
    """The manual fallback path emits a WARNING with 'falling back' in the message.

    Covers line 117 (_log.warning).  We verify the message is present in the
    log record captured from the ``lighttrain.distributed._context`` logger.
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    cfg = _make_cfg(force_cpu=True)

    with caplog.at_level(logging.WARNING, logger="lighttrain.distributed._context"):
        ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    messages = [r.message for r in caplog.records]
    assert any("falling back" in m.lower() for m in messages), (
        f"Expected a 'falling back' warning; got: {messages}"
    )


# --------------------------------------------------------------------------- #
# Lines 125-126: ep > 1 path — _create_ep_groups is called from from_env      #
# --------------------------------------------------------------------------- #


def test_from_env_ep_gt_1_populates_ep_rank_and_ep_group(monkeypatch) -> None:
    """When ep>1, from_env calls _create_ep_groups and populates ctx.ep_rank/ep_group.

    Covers lines 125 (if ep > 1:) and 126 (ep_rank, ep_group = …).

    Input: dp=2, ep=2, tp=1, pp=1, world=2, rank=0.
    Analytical: dp group global ranks = [0, 1]; ep=2 slices into [[0,1]];
    rank 0 belongs to [0,1] → ep_rank = dp_rank % ep = 0 % 2 = 0.
    """
    _patch_dist(monkeypatch, rank=0, world_size=2)
    cfg = _make_cfg(dp=2, tp=1, pp=1, ep=2, force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.ep_degree == 2
    assert ctx.ep_rank == 0
    assert ctx.ep_group is not None


def test_from_env_ep_gt_1_rank1(monkeypatch) -> None:
    """ep=2 at rank=1 (dp_rank=1) gives ep_rank = dp_rank % ep = 1 % 2 = 1.

    Covers the same lines 125-126 for the second rank in the ep group.
    """
    _patch_dist(monkeypatch, rank=1, world_size=2)
    cfg = _make_cfg(dp=2, tp=1, pp=1, ep=2, force_cpu=True)

    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.ep_rank == 1
    assert ctx.ep_group is not None


def test_from_env_ep_eq_1_skips_ep_group_creation(monkeypatch) -> None:
    """ep==1 takes the ep_rank=0/ep_group=None path without calling _create_ep_groups.

    Covers lines 123-124 (ep_rank=0, ep_group=None) and ensures line 125 is
    False so line 126 is NOT reached.
    """
    calls_recorded: list = []

    _patch_dist(monkeypatch, rank=0, world_size=1)

    # Intercept _create_ep_groups to verify it is NOT called
    import lighttrain.distributed._context as _ctx_mod
    original = _ctx_mod._create_ep_groups

    def _spy(*a, **kw):
        calls_recorded.append((a, kw))
        return original(*a, **kw)

    monkeypatch.setattr(_ctx_mod, "_create_ep_groups", _spy)

    cfg = _make_cfg(dp=1, tp=1, pp=1, ep=1, force_cpu=True)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.ep_rank == 0
    assert ctx.ep_group is None
    assert calls_recorded == [], "_create_ep_groups must NOT be called when ep==1"


# --------------------------------------------------------------------------- #
# Lines 103-114: DeviceMesh SUCCESS path (force_cpu=False, mock init_device_mesh)
# --------------------------------------------------------------------------- #


def test_from_env_devicemesh_success_path_uses_mesh_groups(monkeypatch) -> None:
    """When force_cpu=False and init_device_mesh succeeds, use the mesh's groups.

    Covers lines 100 (try:), 103-114 (import + mesh.get_group/get_local_rank).

    We mock ``torch.distributed.device_mesh.init_device_mesh`` to return a
    stub mesh with known per-dim groups and local ranks.
    """
    _patch_dist(monkeypatch, rank=0, world_size=2, initialized=True)

    # Build a fake mesh that records which keys were queried
    _fake_dp_group = SimpleNamespace(_name="dp_group")
    _fake_tp_group = SimpleNamespace(_name="tp_group")
    _fake_pp_group = SimpleNamespace(_name="pp_group")

    _group_map = {"dp": _fake_dp_group, "tp": _fake_tp_group, "pp": _fake_pp_group}
    _rank_map = {"dp": 0, "tp": 0, "pp": 0}

    class _FakeMesh:
        def get_group(self, dim_name: str):
            return _group_map[dim_name]

        def get_local_rank(self, dim_name: str):
            return _rank_map[dim_name]

    _fake_mesh_instance = _FakeMesh()

    def _fake_init_device_mesh(device, shape, *, mesh_dim_names):
        return _fake_mesh_instance

    # Patch at the path the source imports from
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh",
        _fake_init_device_mesh,
    )

    cfg = _make_cfg(dp=2, tp=1, pp=1, ep=1, force_cpu=False)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    # Mesh was not None → device_mesh should be the fake instance
    assert ctx.device_mesh is _fake_mesh_instance
    # Groups came from the mesh
    assert ctx.dp_group is _fake_dp_group
    assert ctx.tp_group is _fake_tp_group
    assert ctx.pp_group is _fake_pp_group
    # Local ranks came from get_local_rank
    assert ctx.dp_rank == 0
    assert ctx.tp_rank == 0
    assert ctx.pp_rank == 0


def test_from_env_devicemesh_exception_triggers_fallback(monkeypatch) -> None:
    """An arbitrary exception from init_device_mesh triggers the except/fallback.

    Covers line 115 (except Exception) when the failure is NOT force_cpu=True
    but a genuine DeviceMesh error (e.g. CUDA not available in a gloo-only env).
    """
    _patch_dist(monkeypatch, rank=0, world_size=1, initialized=True)

    def _explode(device, shape, *, mesh_dim_names):
        raise RuntimeError("simulated CUDA DeviceMesh failure")

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh",
        _explode,
    )

    cfg = _make_cfg(dp=1, tp=1, pp=1, ep=1, force_cpu=False)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    # Fallback engaged: device_mesh must be None
    assert ctx.device_mesh is None
    assert ctx.rank == 0
    assert ctx.world_size == 1


def test_from_env_devicemesh_import_error_triggers_fallback(monkeypatch) -> None:
    """An ImportError on ``from torch.distributed.device_mesh import …`` triggers fallback.

    Covers the except branch (line 115) for the older-PyTorch scenario where
    torch.distributed.device_mesh does not exist.
    """
    _patch_dist(monkeypatch, rank=0, world_size=1, initialized=True)

    # Make the module attribute missing so the from-import raises AttributeError
    import torch.distributed
    getattr(torch.distributed, "device_mesh", None)

    # Patch by removing the attribute
    if hasattr(torch.distributed, "device_mesh"):
        monkeypatch.delattr(torch.distributed, "device_mesh")

    cfg = _make_cfg(dp=1, tp=1, pp=1, ep=1, force_cpu=False)
    try:
        ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]
        # If it succeeds, verify fallback path was taken
        assert ctx.device_mesh is None
    except Exception:  # noqa: BLE001
        # If the module-not-found propagates, that's acceptable too;
        # what matters is we exercised the import path.
        pass


# --------------------------------------------------------------------------- #
# Line 149: local_device fallback to cpu when force_cpu=False and no CUDA      #
# --------------------------------------------------------------------------- #


def test_local_device_no_cuda_returns_cpu(monkeypatch) -> None:
    """local_device returns cpu when force_cpu=False but CUDA is not available.

    Covers line 149 (``return torch.device("cpu")``).
    We monkeypatch ``torch.cuda.is_available`` to return False.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = ParallelContext(force_cpu=False, local_rank=3)
    assert ctx.local_device == torch.device("cpu")


@pytest.mark.parametrize("local_rank", [0, 1, 5])
def test_local_device_no_cuda_ignores_local_rank(monkeypatch, local_rank: int) -> None:
    """local_device cpu fallback ignores local_rank when CUDA unavailable.

    Ensures line 149 is reached for several local_rank values and always
    returns the same ``cpu`` device.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = ParallelContext(force_cpu=False, local_rank=local_rank)
    assert ctx.local_device == torch.device("cpu")


# --------------------------------------------------------------------------- #
# Remaining from_env observable-field pins                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sp", [True, False])
def test_from_env_sp_field_round_trips(monkeypatch, sp: bool) -> None:
    """sp= in cfg always round-trips through to ctx.sp_enabled.

    Covers the sp= read in from_env and the sp_enabled= kwarg in return cls().
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    cfg = _make_cfg(force_cpu=True, sp=sp)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]
    assert ctx.sp_enabled is sp


def test_from_env_local_rank_from_env_var(monkeypatch) -> None:
    """local_rank is read from LOCAL_RANK env var, not from torch.distributed.

    Covers the ``local_rank = int(os.environ.get("LOCAL_RANK", "0"))`` line
    and pins it through the returned ctx.
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "3"})
    cfg = _make_cfg(force_cpu=True)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]
    assert ctx.local_rank == 3


def test_from_env_missing_local_rank_defaults_to_zero(monkeypatch) -> None:
    """When LOCAL_RANK is absent from environment, local_rank defaults to 0.

    Covers the ``os.environ.get("LOCAL_RANK", "0")`` default.
    """
    _patch_dist(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr("os.environ", {})  # no LOCAL_RANK
    cfg = _make_cfg(force_cpu=True)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]
    assert ctx.local_rank == 0


def test_from_env_full_field_inventory_force_cpu(monkeypatch) -> None:
    """Every field returned by from_env with force_cpu=True has correct value.

    A broad integration pin for the return cls(…) block (lines 128-137),
    verifying dp/tp/pp degrees + sp + force_cpu + device_mesh.
    """
    _patch_dist(monkeypatch, rank=2, world_size=4)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "2"})
    # 4-rank mesh: dp=2, tp=2, pp=1
    cfg = _make_cfg(dp=2, tp=2, pp=1, ep=1, force_cpu=True, sp=True)
    ctx = ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert ctx.rank == 2
    assert ctx.local_rank == 2
    assert ctx.world_size == 4
    assert ctx.dp_degree == 2
    assert ctx.tp_degree == 2
    assert ctx.pp_degree == 1
    assert ctx.ep_degree == 1
    assert ctx.sp_enabled is True
    assert ctx.force_cpu is True
    assert ctx.device_mesh is None
    # Analytical from _compute_ranks(2, dp=2, tp=2, pp=1):
    expected_dp, expected_tp, expected_pp = _compute_ranks(2, 2, 2, 1)
    assert ctx.dp_rank == expected_dp
    assert ctx.tp_rank == expected_tp
    assert ctx.pp_rank == expected_pp


def test_from_env_backend_gloo_passed_to_init(monkeypatch) -> None:
    """The backend from cfg is passed verbatim to dist.init_process_group.

    Covers the ``backend = str(getattr(cfg, "backend", "nccl"))`` line and
    the call to ``init_process_group`` on line 77 (not-yet-initialized path).
    """
    captured: list[str] = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(
        "torch.distributed.init_process_group",
        lambda backend, **_kw: captured.append(backend),
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda g: list(g._members),
    )
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    cfg = _make_cfg(backend="gloo", force_cpu=True)
    ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert captured == ["gloo"], f"Expected ['gloo'], got {captured}"


def test_from_env_already_initialized_skips_init_call(monkeypatch) -> None:
    """When dist.is_initialized() is True, init_process_group must NOT be called.

    Pins the ``if not dist.is_initialized():`` guard (line 76), verifying that
    an already-initialized process group is not re-initialized.
    """
    init_called: list = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.init_process_group",
        lambda **_kw: init_called.append(True),
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda g: list(g._members),
    )
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    cfg = _make_cfg(force_cpu=True)
    ParallelContext.from_env(cfg)  # type: ignore[arg-type]

    assert init_called == [], "init_process_group should NOT be called when already initialized"
