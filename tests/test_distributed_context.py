"""Unit tests for lighttrain.distributed._context.

Covers all logic that can be verified without a real multi-process setup:
  - ParallelContext.single_gpu() defaults
  - local_device property (force_cpu vs CUDA vs CPU fallback)
  - is_main_process / is_dp_rank0 / is_pp_last_stage properties
  - __repr__
  - _compute_ranks mapping
  - ParallelSection schema (force_cpu field)
  - NoopGradSyncStrategy single-GPU contract
"""
from __future__ import annotations

import torch

from lighttrain.builtin_plugins.distributed._noop import NoopGradSyncStrategy
from lighttrain.config._schema import ParallelSection
from lighttrain.distributed._context import ParallelContext, _compute_ranks

# ------------------------------------------------------------------ #
# ParallelContext.single_gpu()                                         #
# ------------------------------------------------------------------ #

class TestSingleGpu:
    def test_defaults(self):
        ctx = ParallelContext.single_gpu()
        assert ctx.rank == 0
        assert ctx.local_rank == 0
        assert ctx.world_size == 1
        assert ctx.dp_rank == ctx.tp_rank == ctx.pp_rank == ctx.ep_rank == 0
        assert ctx.dp_degree == ctx.tp_degree == ctx.pp_degree == ctx.ep_degree == 1
        assert ctx.sp_enabled is False
        assert ctx.force_cpu is False
        assert ctx.device_mesh is None
        assert ctx.dp_group is None
        assert ctx.tp_group is None
        assert ctx.pp_group is None
        assert ctx.ep_group is None

    def test_is_main_process(self):
        assert ParallelContext.single_gpu().is_main_process is True

    def test_is_dp_rank0(self):
        assert ParallelContext.single_gpu().is_dp_rank0 is True

    def test_is_pp_last_stage(self):
        assert ParallelContext.single_gpu().is_pp_last_stage is True  # pp_rank=0, pp_degree=1 → 0==0

    def test_repr(self):
        r = repr(ParallelContext.single_gpu())
        assert "rank=0/1" in r
        assert "dp=0/1" in r
        assert "tp=0/1" in r
        assert "pp=0/1" in r


# ------------------------------------------------------------------ #
# local_device property                                                #
# ------------------------------------------------------------------ #

class TestLocalDevice:
    def test_force_cpu_overrides_cuda(self):
        ctx = ParallelContext(force_cpu=True, local_rank=0)
        assert ctx.local_device == torch.device("cpu")

    def test_force_cpu_overrides_cuda_rank1(self):
        ctx = ParallelContext(force_cpu=True, local_rank=1)
        assert ctx.local_device == torch.device("cpu")

    def test_no_force_cpu_cuda_available(self):
        ctx = ParallelContext(force_cpu=False, local_rank=0)
        if torch.cuda.is_available():
            assert ctx.local_device == torch.device("cuda:0")
        else:
            assert ctx.local_device == torch.device("cpu")

    def test_no_force_cpu_local_rank2(self):
        ctx = ParallelContext(force_cpu=False, local_rank=2)
        if torch.cuda.is_available():
            assert ctx.local_device == torch.device("cuda:2")
        else:
            assert ctx.local_device == torch.device("cpu")


# ------------------------------------------------------------------ #
# is_main_process / is_dp_rank0 / is_pp_last_stage                    #
# ------------------------------------------------------------------ #

class TestProperties:
    def test_is_main_only_rank0(self):
        assert ParallelContext(rank=0, world_size=4).is_main_process is True
        assert ParallelContext(rank=1, world_size=4).is_main_process is False
        assert ParallelContext(rank=3, world_size=4).is_main_process is False

    def test_is_dp_rank0(self):
        assert ParallelContext(dp_rank=0, dp_degree=4).is_dp_rank0 is True
        assert ParallelContext(dp_rank=1, dp_degree=4).is_dp_rank0 is False

    def test_is_pp_last_stage(self):
        assert ParallelContext(pp_rank=0, pp_degree=1).is_pp_last_stage is True
        assert ParallelContext(pp_rank=1, pp_degree=2).is_pp_last_stage is True
        assert ParallelContext(pp_rank=0, pp_degree=2).is_pp_last_stage is False
        assert ParallelContext(pp_rank=2, pp_degree=4).is_pp_last_stage is False
        assert ParallelContext(pp_rank=3, pp_degree=4).is_pp_last_stage is True


# ------------------------------------------------------------------ #
# _compute_ranks                                                       #
# ------------------------------------------------------------------ #

class TestComputeRanks:
    """Layout: rank = dp_rank*(tp*pp) + tp_rank*pp + pp_rank"""

    def test_trivial(self):
        assert _compute_ranks(0, 1, 1, 1) == (0, 0, 0)

    def test_dp_only(self):
        for dp_r in range(4):
            assert _compute_ranks(dp_r, 4, 1, 1) == (dp_r, 0, 0)

    def test_tp_only(self):
        for tp_r in range(4):
            assert _compute_ranks(tp_r, 1, 4, 1) == (0, tp_r, 0)

    def test_pp_only(self):
        for pp_r in range(4):
            assert _compute_ranks(pp_r, 1, 1, 4) == (0, 0, pp_r)

    def test_dp2_tp2_pp2(self):
        """
        dp=2, tp=2, pp=2 → 8 ranks.
        rank = dp_r*4 + tp_r*2 + pp_r
        """
        expected = {}
        for dp_r in range(2):
            for tp_r in range(2):
                for pp_r in range(2):
                    r = dp_r * 4 + tp_r * 2 + pp_r
                    expected[r] = (dp_r, tp_r, pp_r)
        for rank, triple in expected.items():
            assert _compute_ranks(rank, 2, 2, 2) == triple, f"rank={rank}"

    def test_dp4_tp2_pp1(self):
        """dp=4, tp=2, pp=1 → 8 ranks. rank = dp_r*2 + tp_r"""
        for dp_r in range(4):
            for tp_r in range(2):
                r = dp_r * 2 + tp_r
                assert _compute_ranks(r, 4, 2, 1) == (dp_r, tp_r, 0)

    def test_roundtrip_all_ranks(self):
        """All 16 ranks of a dp=4 tp=2 pp=2 mesh recover correctly."""
        dp, tp, pp = 4, 2, 2
        seen = set()
        for rank in range(dp * tp * pp):
            triple = _compute_ranks(rank, dp, tp, pp)
            assert triple not in seen, f"duplicate triple {triple} at rank={rank}"
            seen.add(triple)
            dp_r, tp_r, pp_r = triple
            assert 0 <= dp_r < dp
            assert 0 <= tp_r < tp
            assert 0 <= pp_r < pp
        assert len(seen) == dp * tp * pp


# ------------------------------------------------------------------ #
# ParallelSection schema                                               #
# ------------------------------------------------------------------ #

class TestParallelSection:
    def test_defaults(self):
        ps = ParallelSection()
        assert ps.backend == "nccl"
        assert ps.dp == ps.tp == ps.pp == ps.ep == 1
        assert ps.sp is False
        assert ps.force_cpu is False
        assert ps.grad_sync.name == "noop"
        assert ps.tensor_parallel is None
        assert ps.pipeline is None

    def test_force_cpu_true(self):
        ps = ParallelSection(backend="gloo", dp=4, force_cpu=True)
        assert ps.force_cpu is True
        assert ps.backend == "gloo"
        assert ps.dp == 4

    def test_extra_fields_allowed(self):
        ps = ParallelSection(experimental_flag=True)
        assert ps.experimental_flag is True  # type: ignore[attr-defined]


# ------------------------------------------------------------------ #
# NoopGradSyncStrategy                                                 #
# ------------------------------------------------------------------ #

class TestNoopGradSyncStrategy:
    def _make_model(self):
        import torch.nn as nn
        return nn.Linear(4, 2)

    def test_prepare_moves_model_to_device(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
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

    def test_accumulate_is_nullcontext(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
        ctx = strategy.accumulate(model)
        with ctx:
            pass  # must not raise

    def test_backward_computes_grad(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        strategy.backward(loss, model)
        assert all(p.grad is not None for p in model.parameters())

    def test_clip_grad_norm_returns_float(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
        x = torch.randn(2, 4)
        model(x).sum().backward()
        ctx = ParallelContext.single_gpu()
        norm = strategy.clip_grad_norm(model, max_norm=1.0, parallel_ctx=ctx)
        assert isinstance(norm, float)
        assert norm >= 0.0

    def test_unwrap_model_returns_same(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
        assert strategy.unwrap_model(model) is model

    def test_optimizer_step_updates_params(self):
        strategy = NoopGradSyncStrategy()
        model = self._make_model()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        x = torch.randn(2, 4)
        model(x).sum().backward()
        before = [p.clone() for p in model.parameters()]
        strategy.optimizer_step(opt, model)
        for p_before, p_after in zip(before, model.parameters(), strict=False):
            assert not torch.equal(p_before, p_after)
