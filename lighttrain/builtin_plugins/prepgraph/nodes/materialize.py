"""Materialize PrepNode — persist upstream rows as on-disk shards.

Two layouts are supported:

* ``layout: "rows"`` (default) — JSONL or Parquet shards via
  :class:`ShardWriter`. Yields a ``RowsDataset`` reader.
* ``layout: "memmap"`` — fixed-shape numpy memmap (only valid for
  packed/fixed-length rows). Yields a :class:`MemmapDataset` reader.

The resulting cache directory is committed atomically by ``PrepRunner``;
this node only writes shards into the staging dir and emits the
``MANIFEST_COMPLETE`` payload via the runner.
"""

from __future__ import annotations

from typing import Any

from lighttrain.data.cache._rows import _RowsDataset
from lighttrain.data.cache._shards import ShardWriter
from lighttrain.data.cache._memmap import MemmapDataset, write_memmap
from lighttrain.registry import register
from lighttrain.prepgraph.node import NodeResult, PrepNode, RunContext


@register("prep_node", "materialize")
class MaterializeNode(PrepNode):
    """Persist upstream rows to disk; expose a map-style dataset reader.

    Config keys:

    * ``layout``: ``"rows" | "memmap"`` (default ``"rows"``).
    * ``fmt``: ``"jsonl" | "parquet"`` for rows layout (default ``"jsonl"``).
    * ``shard_size``: int (default 50_000) for rows layout.
    * ``seq_len``: int — required for memmap layout.
    * ``dtype``: str — numpy dtype for memmap (default ``"int32"``).
    """

    kind = "materialize"
    schema_kind = "materialized"

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"MaterializeNode {self.name!r}: requires upstream input.")
        upstream = ctx.upstream[self.inputs[0]]
        rows = list(upstream.rows or [])

        layout = str(self.config.get("layout", "rows"))
        out_dir = ctx.store_root  # staging — runner commits to final
        store: Any
        extras: dict[str, Any] = {"row_count": len(rows), "layout": layout}

        if layout == "rows":
            fmt = str(self.config.get("fmt", "jsonl"))
            shard_size = int(self.config.get("shard_size", 50_000))
            writer = ShardWriter(out_dir=out_dir, shard_size=shard_size, fmt=fmt)
            writer.write_many(rows)
            manifest = writer.finalize()
            extras["fmt"] = manifest["fmt"]
            extras["shards"] = len(manifest["shards"])
            store = _RowsDataset(out_dir)
        elif layout == "memmap":
            seq_len = int(self.config.get("seq_len", 0))
            if seq_len <= 0:
                raise ValueError(
                    f"MaterializeNode {self.name!r}: memmap layout needs `seq_len`."
                )
            dtype = str(self.config.get("dtype", "int64"))
            fields = tuple(
                self.config.get(
                    "fields", ("input_ids", "position_ids", "document_ids")
                )
            )
            dtypes = {f: dtype for f in fields}
            extras["seq_len"] = seq_len
            extras["dtype"] = dtype
            extras["fields"] = list(fields)
            write_memmap(
                out_dir, rows=rows, seq_len=seq_len, fields=fields, dtypes=dtypes
            )
            store = MemmapDataset(out_dir)
        else:
            raise ValueError(
                f"MaterializeNode {self.name!r}: unknown layout {layout!r}."
            )

        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=rows,  # downstream may still consume rows in-memory
            store=store,
            extras=extras,
        )


__all__ = ["MaterializeNode"]
