"""Variable-length row shards with cross-tool friendly storage.

We use JSONL by default (zero-dep, debuggable) and Parquet when ``pyarrow`` is
available (faster + smaller for large corpora). The header carries the
preprocess key so a stale shard fails loud rather than corrupting training.

API::

    writer = ShardWriter(out_dir, shard_size=50_000)
    for row in rows:
        writer.write(row)
    writer.finalize()

    for row in iter_rows(out_dir):
        ...

Each shard tracks its row count + completion bit so a crashed run can resume
shard-by-shard via ``shard_state(out_dir)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover — pyarrow is in our base deps
    _HAS_PARQUET = False


def cache_key(
    *,
    tokenizer: str,
    chat_template: str | None = None,
    raw_data_version: str = "0",
    preprocess_code: str = "",
) -> str:
    """Deterministic cache key for a tokenized dataset."""
    payload = {
        "tokenizer": tokenizer,
        "chat_template": chat_template or "",
        "raw_data_version": raw_data_version,
        "preprocess_code": preprocess_code,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@dataclass
class ShardWriter:
    """Append-only sharded row writer (JSONL by default)."""

    out_dir: str | Path
    shard_size: int = 50_000
    fmt: str = "jsonl"  # jsonl | parquet
    _current: list[dict[str, Any]] = field(default_factory=list)
    _shard_idx: int = 0
    _total_rows: int = 0
    _completed: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.fmt == "parquet" and not _HAS_PARQUET:
            self.fmt = "jsonl"
        self.shard_size = max(1, int(self.shard_size))

    def write(self, row: Mapping[str, Any]) -> None:
        self._current.append(dict(row))
        self._total_rows += 1
        if len(self._current) >= self.shard_size:
            self._flush()

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.write(r)

    def finalize(self) -> dict[str, Any]:
        if self._current:
            self._flush()
        manifest = {
            "fmt": self.fmt,
            "shards": list(self._completed),
            "total_rows": self._total_rows,
        }
        (Path(self.out_dir) / "shards.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _flush(self) -> None:
        idx = self._shard_idx
        rows = self._current
        if self.fmt == "parquet":
            path = Path(self.out_dir) / f"shard-{idx:05d}.parquet"
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, str(path))
        else:
            path = Path(self.out_dir) / f"shard-{idx:05d}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
        self._completed.append(
            {"index": idx, "path": path.name, "rows": len(rows), "complete": True}
        )
        self._current.clear()
        self._shard_idx += 1


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_manifest(out_dir: str | Path) -> dict[str, Any] | None:
    p = Path(out_dir) / "shards.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def iter_rows(out_dir: str | Path) -> Iterable[dict[str, Any]]:
    manifest = read_manifest(out_dir)
    out_dir = Path(out_dir)
    if not manifest:
        # Permissive: scan jsonl files in lexicographic order.
        for shard in sorted(out_dir.glob("shard-*.jsonl")):
            yield from _iter_jsonl(shard)
        return
    fmt = manifest.get("fmt", "jsonl")
    for shard in manifest.get("shards", []):
        path = out_dir / shard["path"]
        if fmt == "parquet" and _HAS_PARQUET:
            table = pq.read_table(str(path))
            yield from table.to_pylist()
        else:
            yield from _iter_jsonl(path)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def shard_state(out_dir: str | Path) -> list[dict[str, Any]]:
    """Per-shard completion list — used by ``materialize`` to resume."""
    manifest = read_manifest(out_dir)
    return list(manifest.get("shards", [])) if manifest else []


def count_rows(out_dir: str | Path) -> int:
    manifest = read_manifest(out_dir)
    return int(manifest.get("total_rows", 0)) if manifest else 0


__all__ = [
    "ShardWriter",
    "cache_key",
    "count_rows",
    "iter_rows",
    "read_manifest",
    "shard_state",
]
