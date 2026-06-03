"""Index PrepNode — stub (RAG index, not yet implemented)."""

from __future__ import annotations

from lighttrain.registry import register
from lighttrain.prepgraph.node import NodeResult, PrepNode, RunContext


@register("prep_node", "index")
class IndexNode(PrepNode):
    """RAG index node (not yet implemented)."""

    kind = "index"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:
        raise NotImplementedError(
            "IndexNode: RAG index support is not yet implemented."
        )


__all__ = ["IndexNode"]
