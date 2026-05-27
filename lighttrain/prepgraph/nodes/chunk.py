"""Chunk PrepNode — split rows whose ``input_ids`` exceed ``max_len``.

Optional ``overlap`` keeps a tail of context across chunks (useful for
long-document SFT).
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping

from ...registry import register
from ..node import NodeResult, PrepNode, RunContext


def _chunk_one(
    row: Mapping[str, Any], *, max_len: int, overlap: int
) -> Iterator[dict[str, Any]]:
    ids = list(row.get("input_ids", []))
    labels = list(row.get("labels", ids))
    attn = list(row.get("attention_mask", [1] * len(ids)))
    if len(ids) <= max_len:
        out = dict(row)
        out["input_ids"] = ids
        out["labels"] = labels
        out["attention_mask"] = attn
        yield out
        return
    step = max(1, max_len - overlap)
    for start in range(0, len(ids), step):
        end = start + max_len
        if start >= len(ids):
            break
        out = dict(row)
        out["input_ids"] = ids[start:end]
        out["labels"] = labels[start:end]
        out["attention_mask"] = attn[start:end]
        yield out
        if end >= len(ids):
            break


@register("prep_node", "chunk")
class ChunkNode(PrepNode):
    """Split long sequences into overlapping windows.

    Config keys:

    * ``max_len``: required int.
    * ``overlap``: int (default 0).
    """

    kind = "chunk"
    schema_kind = "chunked_rows"

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"ChunkNode {self.name!r}: requires upstream input.")
        max_len = int(self.config.get("max_len", 0))
        if max_len <= 0:
            raise ValueError(f"ChunkNode {self.name!r}: `max_len` must be > 0.")
        overlap = int(self.config.get("overlap", 0))
        upstream = ctx.upstream[self.inputs[0]]
        rows = upstream.rows or []
        out: list[dict[str, Any]] = []
        for row in rows:
            out.extend(_chunk_one(row, max_len=max_len, overlap=overlap))
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=out,
            extras={"row_count": len(out)},
        )


__all__ = ["ChunkNode"]
