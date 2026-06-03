"""Pipeline parallelism schedules: 1F1B and GPipe.

Both implement the ``PipelineSchedule`` protocol.

1F1B (One-Forward-One-Backward):
  - Steady-state: each stage alternates one forward and one backward pass.
  - Memory-efficient: O(pipeline_stages) activation memory.
  - Used in Megatron-LM, PipeDream.

GPipe:
  - All micro-batches forward, then all backward.
  - Memory: O(n_microbatches * pipeline_stages).
  - Simpler to implement; better for throughput with large microbatch counts.

Both use ``torch.distributed.pipelining`` (PyTorch >= 2.2).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


def _build_split_spec(
    stage_spec: list[dict] | None,
    model: nn.Module,
    parallel_ctx: ParallelContext,
) -> dict[str, Any]:
    """Convert user stage_spec list to torch.distributed.pipelining SplitPoint dict."""
    if not stage_spec:
        return {}
    from torch.distributed.pipelining import SplitPoint
    split: dict[str, Any] = {}
    for _i, spec in enumerate(stage_spec[:-1]):  # N stages → N-1 split points
        layers = spec.get("layers", "")
        if layers:
            last_layer = layers.split(",")[-1].strip()
            split[last_layer] = SplitPoint.END
    return split


def _split_microbatches(batch: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Split a batch dict along dim-0 into n equal micro-batches."""
    chunks: list[dict[str, Any]] = [{} for _ in range(n)]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            parts = v.chunk(n, dim=0)
            for i, part in enumerate(parts):
                chunks[i][k] = part
        else:
            for i in range(n):
                chunks[i][k] = v
    return chunks


@register("pipeline_schedule", "1f1b")
class OneFOneBSchedule:
    """1F1B pipeline schedule (memory-efficient)."""

    def __init__(
        self,
        *,
        n_microbatches: int = 4,
        schedule: str = "1f1b",
        stage_spec: list[dict] | None = None,
        auto_split_for: str | None = None,
    ) -> None:
        self.n_microbatches = max(1, int(n_microbatches))
        self.stage_spec = stage_spec
        self.auto_split_for = auto_split_for

    def prepare(self, model: nn.Module, parallel_ctx: ParallelContext) -> Any:
        """Split model into PP stages using torch.distributed.pipelining."""
        try:
            from torch.distributed.pipelining import SplitPoint, pipeline
        except ImportError as e:
            raise ImportError(
                "torch.distributed.pipelining is required for PP; "
                "upgrade to PyTorch >= 2.2."
            ) from e

        split_spec = _build_split_spec(self.stage_spec, model, parallel_ctx)
        if not split_spec:
            # No explicit split: auto-split by equal layer count.
            layers = list(model.named_children())
            if len(layers) < parallel_ctx.pp_degree:
                raise ValueError(
                    f"Model has {len(layers)} top-level children but pp={parallel_ctx.pp_degree}. "
                    "Provide stage_spec explicitly."
                )
            n = len(layers)
            pp = parallel_ctx.pp_degree
            for i in range(1, pp):
                idx = (i * n) // pp
                split_spec[layers[idx][0]] = SplitPoint.BEGINNING

        pipe = pipeline(model, mb_args=(), split_spec=split_spec)
        return pipe.get_stage_module(parallel_ctx.pp_rank)

    def run_step(
        self, stage: Any, microbatches: list[dict[str, Any]], ctx: Any
    ) -> torch.Tensor:
        """Execute 1F1B schedule over microbatches. Returns loss (only valid on last stage)."""
        loss = torch.zeros(1, device=next(stage.parameters()).device)
        for mb in microbatches:
            out = stage(**mb)
            if ctx.loss_fn is not None and hasattr(ctx, "parallel_ctx") and ctx.parallel_ctx.is_pp_last_stage:
                from lighttrain.protocols import LossContext, ModelOutput
                if not isinstance(out, ModelOutput):
                    out = ModelOutput(outputs={"logits": out} if isinstance(out, torch.Tensor) else dict(out))
                loss_dict = ctx.loss_fn(out, mb, LossContext(step=ctx.step, epoch=ctx.epoch))
                loss = loss + loss_dict.get("loss", torch.zeros(1))
        return loss / len(microbatches)


@register("pipeline_schedule", "gpipe")
class GPipeSchedule:
    """GPipe pipeline schedule (all-forward then all-backward)."""

    def __init__(
        self,
        *,
        n_microbatches: int = 4,
        stage_spec: list[dict] | None = None,
    ) -> None:
        self.n_microbatches = max(1, int(n_microbatches))
        self.stage_spec = stage_spec

    def prepare(self, model: nn.Module, parallel_ctx: ParallelContext) -> Any:
        return OneFOneBSchedule(
            n_microbatches=self.n_microbatches,
            stage_spec=self.stage_spec,
        ).prepare(model, parallel_ctx)

    def run_step(
        self, stage: Any, microbatches: list[dict[str, Any]], ctx: Any
    ) -> torch.Tensor:
        # GPipe: all forwards first, then all backwards (handled by autograd).
        loss = torch.zeros(1, device=next(stage.parameters()).device)
        outputs = [stage(**mb) for mb in microbatches]
        if ctx.loss_fn is not None and hasattr(ctx, "parallel_ctx") and ctx.parallel_ctx.is_pp_last_stage:
            from lighttrain.protocols import LossContext, ModelOutput
            for out, mb in zip(outputs, microbatches, strict=False):
                if not isinstance(out, ModelOutput):
                    out = ModelOutput(outputs={"logits": out} if isinstance(out, torch.Tensor) else dict(out))
                loss_dict = ctx.loss_fn(out, mb, LossContext(step=ctx.step, epoch=ctx.epoch))
                loss = loss + loss_dict.get("loss", torch.zeros(1))
        return loss / len(microbatches)


# Interleaved 1F1B is an alias for 1F1B in this skeleton.
@register("pipeline_schedule", "interleaved_1f1b")
class Interleaved1F1BSchedule(OneFOneBSchedule):
    pass


__all__ = ["OneFOneBSchedule", "GPipeSchedule", "Interleaved1F1BSchedule"]
