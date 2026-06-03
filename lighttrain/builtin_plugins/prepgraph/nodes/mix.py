"""Mix PrepNode — interleave rows from N upstream nodes.

All mixing happens on iterators of in-memory rows. Strategies and
``max_samples_*`` are forwarded to :func:`mix_rows`.
"""

from __future__ import annotations

from typing import Any

from lighttrain.registry import register
from lighttrain.data.mixing._mixed import mix_rows
from lighttrain.prepgraph.node import NodeResult, PrepNode, RunContext


@register("prep_node", "mix")
class MixNode(PrepNode):
    """Mix rows from upstream nodes by ``inputs`` order.

    Config keys:

    * ``strategy``: ``"weighted" | "round_robin" | "exhaust_then_resample"``.
    * ``weights``: list[float] aligned with ``inputs``.
    * ``temperature``: float (1.0 = identity).
    * ``max_samples_per_source``, ``max_samples_total``: optional ints.
    * ``seed``: int.
    """

    kind = "mix"
    schema_kind = "mixed_rows"

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"MixNode {self.name!r}: requires upstream inputs.")
        sources = []
        for u in self.inputs:
            up = ctx.upstream[u]
            sources.append(list(up.rows or []))

        merged = list(
            mix_rows(
                sources,
                strategy=str(self.config.get("strategy", "weighted")),
                weights=self.config.get("weights"),
                temperature=float(self.config.get("temperature", 1.0)),
                max_samples_per_source=self.config.get("max_samples_per_source"),
                max_samples_total=self.config.get("max_samples_total"),
                seed=int(self.config.get("seed", 0)),
            )
        )
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=merged,
            extras={
                "row_count": len(merged),
                "per_source_counts": [len(s) for s in sources],
            },
        )


__all__ = ["MixNode"]
