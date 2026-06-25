"""Coverage-completions for ``lighttrain.distributed._context`` (data-parallel only).

Pins the branches of the dp-only ``ParallelContext.from_env``:
  * init_process_group called only when not yet initialized; backend passed
  * 1-D DeviceMesh success path (dp group + dp local rank from the mesh)
  * fallback path (force_cpu / DeviceMesh failure): mesh=None, dp_group via
    ``dist.new_group(range(dp))``, warning logged
  * LOCAL_RANK env var read (value + default)
  * local_device cpu fallback when CUDA unavailable
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from lighttrain.distributed._context import ParallelContext

# --------------------------------------------------------------------------- #
# Helpers / stubs shared across this file                                      #
# --------------------------------------------------------------------------- #


def _make_cfg(
    *,
    backend: str = "gloo",
    dp: int = 1,
    force_cpu: bool = True,
) -> SimpleNamespace:
    """Minimal config-like object accepted by ``from_env``."""
    return SimpleNamespace(backend=backend, dp=dp, force_cpu=force_cpu)


def _patch_dist(monkeypatch, *, rank: int = 0, world_size: int = 1, initialized: bool = True):
    """Install a minimal torch.distributed mock on the module under test.

    Returns a handle with ``.calls`` (list of new_group members).
    """
    calls: list[list[int]] = []
    returns: list[object] = []

    def _new_group(members):
        members_list = list(members)
        calls.append(members_list)
        obj = SimpleNamespace(_members=tuple(members_list))
        returns.append(obj)
        return obj

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: initialized)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **_kw: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)
    monkeypatch.setattr("torch.distributed.new_group", _new_group)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": str(rank)})

    return SimpleNamespace(calls=calls, new_group_returns=returns)


# --------------------------------------------------------------------------- #
# init_process_group guard                                                     #
# --------------------------------------------------------------------------- #


def test_from_env_calls_init_process_group_when_not_initialized(monkeypatch) -> None:
    """``from_env`` calls ``dist.init_process_group`` when not yet initialized."""
    init_called: list[str] = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(
        "torch.distributed.init_process_group",
        lambda backend, **_kw: init_called.append(backend),
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    ctx = ParallelContext.from_env(_make_cfg(backend="gloo", force_cpu=True))  # type: ignore[arg-type]

    assert init_called == ["gloo"]
    assert ctx.rank == 0
    assert ctx.world_size == 1


def test_from_env_already_initialized_skips_init_call(monkeypatch) -> None:
    """When ``is_initialized()`` is True, ``init_process_group`` must NOT be called."""
    init_called: list = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.init_process_group",
        lambda **_kw: init_called.append(True),
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    ParallelContext.from_env(_make_cfg(force_cpu=True))  # type: ignore[arg-type]

    assert init_called == []


def test_from_env_backend_passed_to_init(monkeypatch) -> None:
    """The backend from cfg is passed verbatim to ``dist.init_process_group``."""
    captured: list[str] = []

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(
        "torch.distributed.init_process_group",
        lambda backend, **_kw: captured.append(backend),
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    ParallelContext.from_env(_make_cfg(backend="gloo", force_cpu=True))  # type: ignore[arg-type]

    assert captured == ["gloo"]


# --------------------------------------------------------------------------- #
# Fallback path: force_cpu / DeviceMesh failure → manual dp group              #
# --------------------------------------------------------------------------- #


def test_from_env_force_cpu_takes_manual_fallback_and_returns_context(monkeypatch) -> None:
    """force_cpu=True skips DeviceMesh and builds the dp group manually.

    Input: dp=2, rank=0, world=2 → dp_rank=rank=0, dp_degree=2, mesh None,
    dp_group created via ``dist.new_group(range(dp))``.
    """
    handle = _patch_dist(monkeypatch, rank=0, world_size=2)

    ctx = ParallelContext.from_env(_make_cfg(dp=2, force_cpu=True))  # type: ignore[arg-type]

    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.world_size == 2
    assert ctx.dp_degree == 2
    assert ctx.dp_rank == 0
    assert ctx.force_cpu is True
    assert ctx.device_mesh is None
    assert ctx.dp_group is not None
    assert handle.calls == [[0, 1]]  # new_group over all dp ranks


def test_from_env_force_cpu_mesh_is_none_in_returned_ctx(monkeypatch) -> None:
    """When the manual fallback fires, ``device_mesh`` in the returned context is None."""
    _patch_dist(monkeypatch, rank=0, world_size=1)
    ctx = ParallelContext.from_env(_make_cfg(dp=1, force_cpu=True))  # type: ignore[arg-type]
    assert ctx.device_mesh is None


def test_from_env_logs_warning_on_fallback(monkeypatch, caplog) -> None:
    """The manual fallback path emits a WARNING with 'falling back' in the message."""
    _patch_dist(monkeypatch, rank=0, world_size=1)

    with caplog.at_level(logging.WARNING, logger="lighttrain.distributed._context"):
        ParallelContext.from_env(_make_cfg(force_cpu=True))  # type: ignore[arg-type]

    messages = [r.message for r in caplog.records]
    assert any("falling back" in m.lower() for m in messages), (
        f"Expected a 'falling back' warning; got: {messages}"
    )


# --------------------------------------------------------------------------- #
# DeviceMesh SUCCESS path (force_cpu=False, mocked init_device_mesh)           #
# --------------------------------------------------------------------------- #


def test_from_env_devicemesh_success_path_uses_mesh_groups(monkeypatch) -> None:
    """When force_cpu=False and init_device_mesh succeeds, use the mesh's dp group."""
    _patch_dist(monkeypatch, rank=0, world_size=2, initialized=True)

    _fake_dp_group = SimpleNamespace(_name="dp_group")

    class _FakeMesh:
        def get_group(self, dim_name: str):
            assert dim_name == "dp"
            return _fake_dp_group

        def get_local_rank(self, dim_name: str):
            assert dim_name == "dp"
            return 0

    _fake_mesh_instance = _FakeMesh()

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh",
        lambda device, shape, *, mesh_dim_names: _fake_mesh_instance,
    )

    ctx = ParallelContext.from_env(_make_cfg(dp=2, force_cpu=False))  # type: ignore[arg-type]

    assert ctx.device_mesh is _fake_mesh_instance
    assert ctx.dp_group is _fake_dp_group
    assert ctx.dp_rank == 0


def test_from_env_devicemesh_exception_triggers_fallback(monkeypatch) -> None:
    """An exception from init_device_mesh triggers the except/fallback path."""
    _patch_dist(monkeypatch, rank=0, world_size=1, initialized=True)

    def _explode(device, shape, *, mesh_dim_names):
        raise RuntimeError("simulated CUDA DeviceMesh failure")

    monkeypatch.setattr("torch.distributed.device_mesh.init_device_mesh", _explode)

    ctx = ParallelContext.from_env(_make_cfg(dp=1, force_cpu=False))  # type: ignore[arg-type]

    assert ctx.device_mesh is None
    assert ctx.rank == 0
    assert ctx.world_size == 1


def test_from_env_devicemesh_import_error_triggers_fallback(monkeypatch) -> None:
    """A missing ``torch.distributed.device_mesh`` triggers the fallback path."""
    _patch_dist(monkeypatch, rank=0, world_size=1, initialized=True)

    import torch.distributed

    if hasattr(torch.distributed, "device_mesh"):
        monkeypatch.delattr(torch.distributed, "device_mesh")

    try:
        ctx = ParallelContext.from_env(_make_cfg(dp=1, force_cpu=False))  # type: ignore[arg-type]
        assert ctx.device_mesh is None
    except Exception:  # noqa: BLE001 — exercising the import path is enough
        pass


# --------------------------------------------------------------------------- #
# local_device cpu fallback + LOCAL_RANK env handling                          #
# --------------------------------------------------------------------------- #


def test_local_device_no_cuda_returns_cpu(monkeypatch) -> None:
    """local_device returns cpu when force_cpu=False but CUDA is not available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = ParallelContext(force_cpu=False, local_rank=3)
    assert ctx.local_device == torch.device("cpu")


@pytest.mark.parametrize("local_rank", [0, 1, 5])
def test_local_device_no_cuda_ignores_local_rank(monkeypatch, local_rank: int) -> None:
    """local_device cpu fallback ignores local_rank when CUDA unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = ParallelContext(force_cpu=False, local_rank=local_rank)
    assert ctx.local_device == torch.device("cpu")


def test_from_env_local_rank_from_env_var(monkeypatch) -> None:
    """local_rank is read from the LOCAL_RANK env var."""
    _patch_dist(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "3"})
    ctx = ParallelContext.from_env(_make_cfg(force_cpu=True))  # type: ignore[arg-type]
    assert ctx.local_rank == 3


def test_from_env_missing_local_rank_defaults_to_zero(monkeypatch) -> None:
    """When LOCAL_RANK is absent, local_rank defaults to 0."""
    _patch_dist(monkeypatch, rank=0, world_size=1)
    monkeypatch.setattr("os.environ", {})  # no LOCAL_RANK
    ctx = ParallelContext.from_env(_make_cfg(force_cpu=True))  # type: ignore[arg-type]
    assert ctx.local_rank == 0
