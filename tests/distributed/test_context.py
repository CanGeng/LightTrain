"""Adversarial tests for ``lighttrain.distributed._context``.

Covers the data-parallel-only ``ParallelContext``:
  * ``single_gpu()`` purity (no torch.distributed calls) + observable defaults
  * ``from_env`` fail-loud guard when ``dp != world_size``
  * ``local_device`` selection (force_cpu dominance + CUDA indexing)
  * ``ParallelSection`` schema defaults / overrides
  * ``NoopGradSyncStrategy`` single-GPU contract

Numeric assertions are not relevant here (integer rank arithmetic), so exact
integer/list equality is the right comparator rather than ``assert_close``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lighttrain.builtin_plugins.distributed._noop import NoopGradSyncStrategy
from lighttrain.config._schema import ParallelSection
from lighttrain.distributed._context import ParallelContext

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


@pytest.mark.parametrize("local_rank", [0, 1, 7])
def test_local_device_force_cpu_dominates(local_rank: int) -> None:
    """``force_cpu=True`` always returns CPU regardless of CUDA / local_rank.

    Contract: the force_cpu kill-switch must short-circuit any CUDA selection.
    Pre-checked: rank 0/1/7 all hit the same branch.
    """
    ctx = ParallelContext(force_cpu=True, local_rank=local_rank)
    assert ctx.local_device == torch.device("cpu")


def test_dp_mismatch_raises(monkeypatch) -> None:
    """``from_env`` raises ``ValueError`` when ``dp != world_size``.

    Input: dp=4 but world_size=7.
    Contract: misconfigured launchers fail loud, not silently degrade.
    """
    cfg = SimpleNamespace(backend="gloo", dp=4, force_cpu=True)

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **_kw: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 7)
    monkeypatch.setattr("os.environ", {"LOCAL_RANK": "0"})

    with pytest.raises(ValueError, match="world_size"):
        ParallelContext.from_env(cfg)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# single_gpu() observable defaults + degenerate-property contract             #
# --------------------------------------------------------------------------- #


def test_single_gpu_observable_defaults() -> None:
    """``single_gpu()`` yields the documented trivial topology.

    Pin every observable field: all ranks 0, dp_degree 1, force_cpu off, and
    every group handle None.
    """
    ctx = ParallelContext.single_gpu()
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.world_size == 1
    assert ctx.dp_rank == 0
    assert ctx.dp_degree == 1
    assert ctx.force_cpu is False
    assert ctx.device_mesh is None
    assert ctx.dp_group is None


def test_single_gpu_degenerate_properties_all_true() -> None:
    """On single-GPU, the boundary-role properties are all True.

    ``is_main_process`` (rank 0) and ``is_dp_rank0`` (dp_rank 0) hold trivially.
    """
    ctx = ParallelContext.single_gpu()
    assert ctx.is_main_process is True
    assert ctx.is_dp_rank0 is True


def test_single_gpu_repr_encodes_topology() -> None:
    """``repr`` surfaces the rank/dp coordinates as ``x=cur/degree``."""
    r = repr(ParallelContext.single_gpu())
    assert "rank=0/1" in r
    assert "dp=0/1" in r


def test_is_main_process_only_rank0() -> None:
    """``is_main_process`` is True iff global rank == 0, for world_size>1."""
    assert ParallelContext(rank=0, world_size=4).is_main_process is True
    assert ParallelContext(rank=1, world_size=4).is_main_process is False
    assert ParallelContext(rank=3, world_size=4).is_main_process is False


def test_is_dp_rank0_only_dp_rank0() -> None:
    """``is_dp_rank0`` is True iff dp_rank == 0, independent of dp_degree."""
    assert ParallelContext(dp_rank=0, dp_degree=4).is_dp_rank0 is True
    assert ParallelContext(dp_rank=1, dp_degree=4).is_dp_rank0 is False


# --------------------------------------------------------------------------- #
# local_device: CUDA-available branch (force_cpu=False)                       #
# --------------------------------------------------------------------------- #


def test_local_device_no_force_cpu_uses_cuda_or_cpu_rank0() -> None:
    """With ``force_cpu=False`` at local_rank 0: ``cuda:0`` if available else cpu."""
    ctx = ParallelContext(force_cpu=False, local_rank=0)
    if torch.cuda.is_available():
        assert ctx.local_device == torch.device("cuda:0")
    else:
        assert ctx.local_device == torch.device("cpu")


def test_local_device_no_force_cpu_indexes_local_rank() -> None:
    """With ``force_cpu=False``, ``local_device`` indexes the given local_rank."""
    ctx = ParallelContext(force_cpu=False, local_rank=2)
    if torch.cuda.is_available():
        assert ctx.local_device == torch.device("cuda:2")
    else:
        assert ctx.local_device == torch.device("cpu")


# --------------------------------------------------------------------------- #
# ParallelSection schema                                                       #
# --------------------------------------------------------------------------- #


def test_parallel_section_defaults() -> None:
    """A bare ``ParallelSection`` carries the documented single-GPU defaults."""
    ps = ParallelSection()
    assert ps.backend == "nccl"
    assert ps.dp == 1
    assert ps.force_cpu is False
    assert ps.grad_sync.name == "noop"


def test_parallel_section_force_cpu_and_overrides() -> None:
    """Explicit fields (backend/dp/force_cpu) round-trip onto the schema."""
    ps = ParallelSection(backend="gloo", dp=4, force_cpu=True)
    assert ps.force_cpu is True
    assert ps.backend == "gloo"
    assert ps.dp == 4


def test_parallel_section_allows_extra_fields() -> None:
    """Unknown experimental fields are accepted (extra fields allowed)."""
    ps = ParallelSection(experimental_flag=True)  # type: ignore[call-arg]
    assert ps.experimental_flag is True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# NoopGradSyncStrategy single-GPU contract                                     #
# --------------------------------------------------------------------------- #


def _noop_make_model():
    import torch.nn as nn

    return nn.Linear(4, 2)


def test_noop_prepare_moves_model_to_device() -> None:
    """``prepare`` moves the model to ``device``, builds an opt, passes loader through."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    device = torch.device("cpu")
    ctx = ParallelContext.single_gpu()
    wrapped, opt, loader = strategy.prepare(
        model,
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
        loader=None,
        parallel_ctx=ctx,
        device=device,
    )
    assert next(wrapped.parameters()).device == device
    assert opt is not None
    assert loader is None  # passthrough


def test_noop_accumulate_is_nullcontext() -> None:
    """``accumulate`` returns a context manager that enters/exits without raising."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    ctx = strategy.accumulate(model)
    with ctx:
        pass  # must not raise


def test_noop_backward_computes_grad() -> None:
    """``backward`` populates ``.grad`` on every parameter."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    x = torch.randn(2, 4)
    loss = model(x).sum()
    strategy.backward(loss, model)
    assert all(p.grad is not None for p in model.parameters())


def test_noop_clip_grad_norm_returns_nonneg_float() -> None:
    """``clip_grad_norm`` returns a non-negative float total norm."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    x = torch.randn(2, 4)
    model(x).sum().backward()
    ctx = ParallelContext.single_gpu()
    norm = strategy.clip_grad_norm(model, max_norm=1.0, parallel_ctx=ctx)
    assert isinstance(norm, float)
    assert norm >= 0.0


def test_noop_unwrap_model_is_identity() -> None:
    """``unwrap_model`` returns the same object (no wrapper in single-GPU)."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    assert strategy.unwrap_model(model) is model


def test_noop_optimizer_step_updates_params() -> None:
    """``optimizer_step`` applies the gradient update so params change."""
    strategy = NoopGradSyncStrategy()
    model = _noop_make_model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.randn(2, 4)
    model(x).sum().backward()
    before = [p.clone() for p in model.parameters()]
    strategy.optimizer_step(opt, model)
    for p_before, p_after in zip(before, model.parameters(), strict=False):
        assert not torch.equal(p_before, p_after)
