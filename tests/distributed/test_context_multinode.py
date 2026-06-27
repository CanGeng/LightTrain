"""Multi-node fail-loud guard for ``ParallelContext.from_env``.

lighttrain validates only single-node multi-GPU (DDP / FSDP / ZeRO). These
tests pin the new guard that raises ``RuntimeError`` on multi-node launches
(see PLAN_v0.5.5.md Block A).

NNODES check runs *before* ``dist.init_process_group`` so multi-node launches
fail fast instead of hanging on rendezvous.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from lighttrain.distributed._context import ParallelContext


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(backend="gloo", dp=1, force_cpu=True)


def test_multi_node_raises_on_nnodes_env(monkeypatch) -> None:
    """``from_env`` raises ``RuntimeError`` when ``NNODES>1`` before touching dist.

    Contract: the guard must fire *before* ``dist.init_process_group`` so a
    multi-node torchrun fails immediately rather than hanging on rendezvous.
    The test installs dist stubs that would *succeed* if reached — proving
    the raise happens upstream.
    """
    monkeypatch.setenv("NNODES", "2")

    dist_called: list[str] = []

    def _boom(*_a, **_kw):
        dist_called.append("called")
        raise AssertionError("dist.init_process_group must not be reached")

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.init_process_group", _boom)

    with pytest.raises(RuntimeError, match="Multi-node training is not supported"):
        ParallelContext.from_env(_make_cfg())  # type: ignore[arg-type]

    assert dist_called == [], "NNODES guard must fire before any dist call"


def test_multi_node_raises_on_local_world_size_split(monkeypatch) -> None:
    """``from_env`` raises when ``LOCAL_WORLD_SIZE < WORLD_SIZE``.

    Double-signal defense: torchrun can split the world across nodes without
    setting NNODES explicitly (e.g. via placement-derived env vars). The check
    runs *after* ``init_process_group`` and must raise even when NNODES==1.
    """
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **_kw: None)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))

    with pytest.raises(RuntimeError, match="Multi-node detected"):
        ParallelContext.from_env(_make_cfg())  # type: ignore[arg-type]


def test_single_node_unchanged(monkeypatch) -> None:
    """Single-node launch (NNODES==1, LOCAL_WORLD_SIZE==WORLD_SIZE) passes the guard.

    Regression: verifies the new guard does not trigger on legitimate
    single-node launches — the existing cov tests cover the rest of from_env,
    here we just ensure both guards are inert.
    """
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.new_group", lambda m: SimpleNamespace(_members=tuple(m)))

    ctx = ParallelContext.from_env(_make_cfg())  # type: ignore[arg-type]
    assert ctx.rank == 0
    assert ctx.world_size == 1
