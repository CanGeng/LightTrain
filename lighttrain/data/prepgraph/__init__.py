"""PrepGraph DAG.

Single-process data preparation DAG with content-addressed caching, atomic
writes, layer-parallel execution, and per-shard resume. The public surface is::

    from lighttrain.data.prepgraph import PrepGraph, PrepRunner, PrepNode

Concrete node implementations register through ``@register("prep_node", ...)``
and live under ``lighttrain.builtin_plugins.data.prepgraph.nodes``.
"""

from ._banner import PlanEntry, format_plan, print_plan
from ._fp import (
    SCHEMA_VERSION,
    canonical_config,
    code_version_for,
    compose_fingerprint,
)
from .dag import PrepGraph
from .node import NodeEstimate, NodeResult, PrepNode, RunContext
from .runner import PrepRunner

__all__ = [
    "NodeEstimate",
    "NodeResult",
    "PlanEntry",
    "PrepGraph",
    "PrepNode",
    "PrepRunner",
    "RunContext",
    "SCHEMA_VERSION",
    "canonical_config",
    "code_version_for",
    "compose_fingerprint",
    "format_plan",
    "print_plan",
]
