"""Load PrepNode — read raw rows from JSONL / Parquet / HuggingFace datasets.

Yields rows lazily; nothing is written to disk by this node. Downstream
nodes consume the iterator. The fingerprint depends on the source URI and
``raw_data_version``, so rotating the upstream data automatically
invalidates downstream caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ...registry import register
from ..node import NodeEstimate, NodeResult, PrepNode, RunContext


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(str(path))
    for r in table.to_pylist():
        yield r


def _iter_dir(path: Path) -> Iterator[dict[str, Any]]:
    files = sorted(path.iterdir())
    for f in files:
        if f.suffix == ".jsonl":
            yield from _iter_jsonl(f)
        elif f.suffix == ".parquet":
            yield from _iter_parquet(f)


def _iter_hf(name: str, split: str, subset: str | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
    for ex in ds:
        yield dict(ex)


@register("prep_node", "load")
class LoadNode(PrepNode):
    """Read raw rows from a fixed source.

    Config keys:

    * ``source``: required. ``"jsonl:<path>"``, ``"parquet:<path>"``,
      ``"dir:<path>"`` (auto-detect by extension), or ``"hf:<name>[:subset]"``.
    * ``split``: HF split name (default ``"train"``).
    * ``raw_data_version``: arbitrary string baked into the fingerprint so
      bumping it invalidates downstream caches without changing the URI.
    * ``limit``: optional int — emit at most this many rows.
    """

    kind = "load"
    schema_kind = "rows"

    def estimate(self, ctx: RunContext) -> NodeEstimate:
        return NodeEstimate(note=f"load source={self.config.get('source')!r}")

    def _iter(self) -> Iterator[dict[str, Any]]:
        source = str(self.config.get("source") or "")
        if not source:
            raise ValueError(f"LoadNode {self.name!r}: missing `source` in config.")
        scheme, _, payload = source.partition(":")
        if not payload and scheme:
            # Allow plain paths.
            payload, scheme = scheme, "auto"
        if scheme in ("jsonl",):
            return _iter_jsonl(Path(payload))
        if scheme in ("parquet",):
            return _iter_parquet(Path(payload))
        if scheme in ("dir",):
            return _iter_dir(Path(payload))
        if scheme in ("hf",):
            name, _, subset = payload.partition(":")
            return _iter_hf(
                name=name,
                split=str(self.config.get("split", "train")),
                subset=subset or None,
            )
        if scheme in ("auto",):
            p = Path(payload)
            if p.is_dir():
                return _iter_dir(p)
            if p.suffix == ".jsonl":
                return _iter_jsonl(p)
            if p.suffix == ".parquet":
                return _iter_parquet(p)
        raise ValueError(f"LoadNode {self.name!r}: unknown source scheme {scheme!r}.")

    def run(self, ctx: RunContext) -> NodeResult:
        limit = self.config.get("limit")
        rows = list(self._iter())
        if limit is not None:
            rows = rows[: int(limit)]
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=rows,
            extras={"row_count": len(rows)},
        )


__all__ = ["LoadNode"]
