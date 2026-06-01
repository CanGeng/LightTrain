"""ExpertParallelStrategy — route MoE expert layers to assigned EP ranks.

EP is stateful: routing decisions and expert weights are rank-specific.
The strategy installs all-to-all communication hooks on the router so that
tokens are dispatched to the correct expert rank.

Requires the model to expose MoE router modules at a known path (configurable
via ``router_module_pattern``).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


@register("model_parallel_strategy", "expert_parallel")
class ExpertParallelStrategy:
    """Expert parallelism: route MoE tokens across EP process group."""

    def __init__(
        self,
        *,
        router_module_pattern: str = "*.router",
        top_k: int = 2,
    ) -> None:
        self.router_module_pattern = router_module_pattern
        self.top_k = top_k

    def apply(self, model: nn.Module, parallel_ctx: ParallelContext) -> nn.Module:
        if parallel_ctx.ep_degree <= 1 or parallel_ctx.ep_group is None:
            return model

        # Find all modules matching the router pattern.
        routers = _find_modules(model, self.router_module_pattern)
        for name, router in routers:
            _install_ep_hooks(router, parallel_ctx, self.top_k)

        return model

    def is_stateless(self) -> bool:
        # EP routing state is rank-specific — can't naively share state dicts.
        return False


def _find_modules(model: nn.Module, pattern: str) -> list[tuple[str, nn.Module]]:
    """Find named modules matching a glob pattern (supports leading *)."""
    suffix = pattern.lstrip("*.")
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if name.endswith(suffix)
    ]


def _install_ep_hooks(router: nn.Module, parallel_ctx: ParallelContext, top_k: int) -> None:
    """Install all-to-all dispatch/combine hooks on a router module.

    This is a skeleton — a real implementation would replace the dispatch
    step inside the router's forward with all-to-all communication.
    """
    ep_group = parallel_ctx.ep_group
    ep_rank = parallel_ctx.ep_rank
    ep_size = parallel_ctx.ep_degree

    # Attach metadata so the router can inspect its EP config at forward time.
    router._ep_group = ep_group      # type: ignore[attr-defined]
    router._ep_rank = ep_rank        # type: ignore[attr-defined]
    router._ep_size = ep_size        # type: ignore[attr-defined]
    router._ep_top_k = top_k         # type: ignore[attr-defined]


__all__ = ["ExpertParallelStrategy"]
