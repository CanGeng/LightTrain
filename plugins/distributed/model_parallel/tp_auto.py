"""TensorParallelStrategy — automatic TP surgery via torch.distributed.tensor.parallel.

Two paths:
A) auto_plan_for: use a built-in plan for known architectures (llama, gpt2, mistral)
B) plan: user-supplied list of {module_path: colwise|rowwise} specs
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


# Built-in parallelize_plan presets for common architectures.
# Keys are normalised lowercase architecture names; values are dicts mapping
# submodule dotted paths to parallel styles ("colwise" / "rowwise" / "sequence_parallel").
_BUILTIN_PLANS: dict[str, dict[str, str]] = {
    "llama": {
        # Attention projections
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        # MLP
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    },
    "gpt2": {
        "transformer.h.*.attn.c_attn": "colwise",
        "transformer.h.*.attn.c_proj": "rowwise",
        "transformer.h.*.mlp.c_fc": "colwise",
        "transformer.h.*.mlp.c_proj": "rowwise",
    },
    "mistral": {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    },
}


def _expand_plan(raw_plan: dict[str, str], model: nn.Module) -> dict[str, Any]:
    """Expand wildcard patterns (*.0.*) to concrete submodule names."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    style_map = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
    }

    expanded: dict[str, Any] = {}
    for pattern, style in raw_plan.items():
        if "*" not in pattern:
            expanded[pattern] = style_map.get(style, ColwiseParallel())
            continue
        # Expand one * at a time against actual named modules.
        parts = pattern.split(".")
        _expand_recursive(model, parts, 0, "", style_map.get(style, ColwiseParallel()), expanded)
    return expanded


def _expand_recursive(
    current_module: nn.Module,
    parts: list[str],
    depth: int,
    prefix: str,
    style: Any,
    out: dict[str, Any],
) -> None:
    if depth == len(parts):
        out[prefix.lstrip(".")] = style
        return
    part = parts[depth]
    if part == "*":
        for name, _ in current_module.named_children():
            child = getattr(current_module, name, None)
            if child is not None:
                _expand_recursive(child, parts, depth + 1, f"{prefix}.{name}", style, out)
    else:
        child = getattr(current_module, part, None)
        if child is not None:
            _expand_recursive(child, parts, depth + 1, f"{prefix}.{part}", style, out)


@register("model_parallel_strategy", "tensor_parallel")
class TensorParallelStrategy:
    """Automatic tensor-parallelism surgery using parallelize_module."""

    def __init__(
        self,
        *,
        auto_plan_for: str | None = None,
        plan: list[dict[str, str]] | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        self.auto_plan_for = auto_plan_for
        self.plan = plan
        self.sequence_parallel = sequence_parallel

    def apply(self, model: nn.Module, parallel_ctx: ParallelContext) -> nn.Module:
        from torch.distributed.tensor.parallel import parallelize_module

        if parallel_ctx.tp_degree <= 1:
            return model

        if parallel_ctx.device_mesh is None:
            raise RuntimeError(
                "TensorParallelStrategy requires a DeviceMesh "
                "(parallel_ctx.device_mesh must be set). "
                "Use ParallelContext.from_env() to initialise it."
            )

        tp_mesh = parallel_ctx.device_mesh["tp"]

        if self.auto_plan_for is not None:
            raw = _BUILTIN_PLANS.get(self.auto_plan_for.lower())
            if raw is None:
                raise ValueError(
                    f"No built-in TP plan for {self.auto_plan_for!r}. "
                    f"Available: {sorted(_BUILTIN_PLANS)}. "
                    "Use 'plan' to provide an explicit list instead."
                )
            parallelize_plan = _expand_plan(raw, model)
        elif self.plan is not None:
            raw_merged: dict[str, str] = {}
            for entry in self.plan:
                raw_merged.update(entry)
            parallelize_plan = _expand_plan(raw_merged, model)
        else:
            raise ValueError(
                "TensorParallelStrategy requires either 'auto_plan_for' "
                "(a known arch name) or 'plan' (explicit module list)."
            )

        return parallelize_module(model, tp_mesh, parallelize_plan)

    def is_stateless(self) -> bool:
        return True  # TP only reshards weights; no rank-specific routing state


__all__ = ["TensorParallelStrategy"]
