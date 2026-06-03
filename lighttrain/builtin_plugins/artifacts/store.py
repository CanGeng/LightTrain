"""Artifact stores.

Three on-disk backends:

  * ``safetensors-shards`` (default) — variable-length tensors keyed by
    ``sample_id``. Each shard packs N samples; the manifest maps
    ``sample_id -> shard_idx``. Optimized for the typical "logits per token"
    use case where each sample has a different length.
  * ``memmap-fixed`` — single ``data.bin`` + ``header.json``. Fixed shape per
    tensor; layout suits packed pretraining caches.
  * ``parquet-rows`` — pure-pandas / pyarrow friendly. Optional dep; raises
    a clear ``ImportError`` when unavailable.

Every store carries an :class:`ArtifactHeader`. ``open_artifact_store`` verifies
the on-disk header against a user-supplied expectation; mismatches raise
:class:`StaleArtifactError` unless ``allow_stale=True``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from lighttrain.artifacts.base import (
    ArtifactHeader,
    ArtifactIncompleteError,
    ArtifactStoreBase,
    StaleArtifactError,
)
from lighttrain.protocols import ArtifactStoreProtocol
from lighttrain.registry import register

try:
    from safetensors.torch import load_file as _st_load_file
    from safetensors.torch import save_file as _st_save_file
    _HAS_ST = True
except ImportError:  # pragma: no cover — safetensors is a hard dep
    _HAS_ST = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:  # pragma: no cover — optional
    _HAS_PARQUET = False




# ---------------------------------------------------------------- safetensors-shards


@register("artifact_store", "safetensors-shards")
class SafetensorsShardStore(ArtifactStoreBase):
    """Default backend.

    Each ``sample_id``'s tensors are saved as keys ``"<sample_id>/<tensor_name>"``
    inside a shard file. ``shard_size`` samples per shard. ``manifest.json``
    maps ``sample_id -> shard_idx``; ``MANIFEST_COMPLETE.json`` is the
    presence-marker written last.
    """

    backend = "safetensors-shards"

    def __init__(
        self,
        root: str | Path,
        *,
        shard_size: int = 1000,
        header: ArtifactHeader | None = None,
    ) -> None:
        if not _HAS_ST:  # pragma: no cover
            raise ImportError("safetensors-shards backend requires `safetensors`.")
        super().__init__(root, header=header)
        self.shard_size = int(shard_size)
        self._idx_path = self.root / "manifest.json"
        self._index: dict[str, int] = {}
        self._pending: dict[str, dict[str, torch.Tensor]] = {}
        if self._idx_path.exists():
            try:
                self._index = json.loads(self._idx_path.read_text(encoding="utf-8")).get(
                    "sample_to_shard", {}
                )
            except json.JSONDecodeError:
                self._index = {}

    def put(self, sample_id: str, tensors: Mapping[str, torch.Tensor]) -> None:
        """Persist ``tensors`` for ``sample_id``.

        Idempotent: if ``sample_id`` was already put — whether already
        flushed to a shard (``_index``) or still buffered (``_pending``) —
        the call is a no-op and the first write wins. This makes the
        store resume-safe both across process restarts AND within a
        single in-flight session before the first flush.
        """
        if self._finalized:
            raise RuntimeError(f"store at {self.root} already finalized — cannot put more samples")
        if sample_id in self._index:
            return  # first write already flushed to a shard
        if sample_id in self._pending:
            return  # first write still buffered — second put must not overwrite
        self._pending[sample_id] = {
            f"{sample_id}/{k}": v.detach().contiguous().cpu() for k, v in tensors.items()
        }
        if len(self._pending) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._pending:
            return
        shard_idx = self._next_shard_idx()
        merged: dict[str, torch.Tensor] = {}
        for sid_tensors in self._pending.values():
            merged.update(sid_tensors)
        shard_path = self.root / f"shard_{shard_idx:05d}.safetensors"
        tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
        _st_save_file(merged, str(tmp))
        os.replace(tmp, shard_path)
        complete = self.root / f"shard_{shard_idx:05d}.complete"
        complete.write_text("ok", encoding="utf-8")
        for sid in self._pending:
            self._index[sid] = shard_idx
        self._persist_index()
        self._pending.clear()

    def _next_shard_idx(self) -> int:
        existing = sorted(self.root.glob("shard_*.safetensors"))
        if not existing:
            return 0
        last = existing[-1].stem.split("_")[-1]
        return int(last) + 1

    def _persist_index(self) -> None:
        body = {"sample_to_shard": self._index, "shard_size": self.shard_size}
        tmp = self._idx_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, self._idx_path)

    def contains(self, sample_id: str) -> bool:
        return sample_id in self._index or sample_id in self._pending

    def iter_keys(self) -> Iterator[str]:
        yield from self._index.keys()

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        if sample_id in self._pending:
            return {k.split("/", 1)[1]: v for k, v in self._pending[sample_id].items()}
        if sample_id not in self._index:
            raise KeyError(sample_id)
        shard_idx = self._index[sample_id]
        shard_path = self.root / f"shard_{shard_idx:05d}.safetensors"
        loaded = _st_load_file(str(shard_path))
        out: dict[str, torch.Tensor] = {}
        prefix = f"{sample_id}/"
        for k, v in loaded.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
        return out

    def finalize(self) -> Path:
        self._flush_shard()
        self._persist_index()
        self._write_header()
        manifest = self._write_manifest({"count": len(self._index)})
        self._finalized = True
        return manifest


# ---------------------------------------------------------------- memmap-fixed


@register("artifact_store", "memmap-fixed")
class MemmapFixedStore(ArtifactStoreBase):
    """Fixed shape per sample. Single ``data.bin`` per tensor name.

    Use when every sample's tensor has identical shape (e.g. fixed seq_len
    packed logits topk). Layout: one ``<tensor>.bin`` + ``<tensor>.shape.json``
    per tensor name; ``manifest.json`` maps ``sample_id -> row_idx``.
    """

    backend = "memmap-fixed"

    def __init__(self, root: str | Path, *, header: ArtifactHeader | None = None) -> None:
        super().__init__(root, header=header)
        self._index: dict[str, int] = {}
        self._next_row = 0
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self._handles: dict[str, Any] = {}
        idx_path = self.root / "manifest.json"
        if idx_path.exists():
            try:
                payload = json.loads(idx_path.read_text(encoding="utf-8"))
                self._index = payload.get("sample_to_row", {})
                self._next_row = max(self._index.values()) + 1 if self._index else 0
                self._shapes = {k: tuple(v) for k, v in payload.get("shapes", {}).items()}
            except json.JSONDecodeError:
                pass

    def put(self, sample_id: str, tensors: Mapping[str, torch.Tensor]) -> None:
        if self._finalized:
            raise RuntimeError(
                f"store at {self.root} already finalized — cannot put more samples"
            )
        if sample_id in self._index:
            return
        row = self._next_row
        for name, t in tensors.items():
            t = t.detach().contiguous().cpu()
            shape = tuple(t.shape)
            if name not in self._shapes:
                self._shapes[name] = shape
                self._dtypes[name] = t.dtype
            elif self._shapes[name] != shape:
                raise ValueError(
                    f"memmap-fixed expects identical shapes per tensor; "
                    f"{name}: prev {self._shapes[name]}, got {shape}"
                )
            self._append_row(name, t)
        self._index[sample_id] = row
        self._next_row += 1

    def _append_row(self, name: str, tensor: torch.Tensor) -> None:
        bin_path = self.root / f"{name}.bin"
        meta_path = self.root / f"{name}.shape.json"
        with open(bin_path, "ab") as f:
            f.write(tensor.numpy().tobytes())
        meta_path.write_text(
            json.dumps({"shape": list(self._shapes[name]), "dtype": str(self._dtypes[name])}),
            encoding="utf-8",
        )

    def contains(self, sample_id: str) -> bool:
        return sample_id in self._index

    def iter_keys(self) -> Iterator[str]:
        yield from self._index.keys()

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        if sample_id not in self._index:
            raise KeyError(sample_id)
        row = self._index[sample_id]
        out: dict[str, torch.Tensor] = {}
        for name, shape in self._shapes.items():
            bin_path = self.root / f"{name}.bin"
            meta = json.loads((self.root / f"{name}.shape.json").read_text(encoding="utf-8"))
            dtype = _dtype_from_str(meta["dtype"])
            row_size = 1
            for d in shape:
                row_size *= d
            elem_bytes = torch.tensor([], dtype=dtype).element_size()
            offset = row * row_size * elem_bytes
            with open(bin_path, "rb") as f:
                f.seek(offset)
                buf = f.read(row_size * elem_bytes)
            t = torch.frombuffer(buf, dtype=dtype).reshape(shape).clone()
            out[name] = t
        return out

    def finalize(self) -> Path:
        idx_path = self.root / "manifest.json"
        idx_path.write_text(
            json.dumps({
                "sample_to_row": self._index,
                "shapes": {k: list(v) for k, v in self._shapes.items()},
            }),
            encoding="utf-8",
        )
        self._write_header()
        manifest = self._write_manifest({"count": len(self._index)})
        self._finalized = True
        return manifest


def _dtype_from_str(s: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32, "torch.float": torch.float32,
        "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64, "torch.long": torch.int64,
        "torch.int32": torch.int32, "torch.int16": torch.int16,
        "torch.uint8": torch.uint8, "torch.bool": torch.bool,
    }
    return mapping.get(s, torch.float32)


# ---------------------------------------------------------------- parquet-rows


@register("artifact_store", "parquet-rows")
class ParquetRowStore(ArtifactStoreBase):
    """Row-oriented parquet. Each sample = one row; each tensor field = column
    with serialized payload. Requires ``pyarrow``.
    """

    backend = "parquet-rows"

    def __init__(self, root: str | Path, *, header: ArtifactHeader | None = None) -> None:
        if not _HAS_PARQUET:
            raise ImportError(
                "parquet-rows backend requires `pyarrow`. "
                "Install with `pip install pyarrow` or use safetensors-shards."
            )
        super().__init__(root, header=header)
        self._rows: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        # Reload a finalized store from disk so ``open_artifact_store`` returns
        # a populated (read-back) store across processes — mirrors
        # ``MemmapFixedStore.__init__``. ``self._finalized`` is set by the base
        # class from ``MANIFEST_COMPLETE.json``.
        if self._finalized:
            self._reload_from_disk()

    def _finalized_count(self) -> int:
        """Sample count recorded in ``MANIFEST_COMPLETE.json`` (0 if unreadable)."""
        complete = self.root / "MANIFEST_COMPLETE.json"
        if not complete.exists():
            return 0
        try:
            return int(json.loads(complete.read_text(encoding="utf-8")).get("count", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0

    def _reload_from_disk(self) -> None:
        """Rebuild ``_rows`` / ``_index`` from a finalized store on disk.

        ``manifest.json`` (``sample_to_row``) is the index source of truth.
        Unlike the safetensors/memmap reload paths, a corrupt or missing
        manifest here is fail-loud, not a silent empty store.
        """
        idx_path = self.root / "manifest.json"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"finalized parquet store at {self.root} is missing manifest.json"
            )
        payload = json.loads(idx_path.read_text(encoding="utf-8"))
        # ``sample_to_row`` must be present and a dict — a manifest that lost it
        # (e.g. truncated to ``{}``) is corrupt, NOT an empty store, and must
        # fail loud rather than silently return zero samples.
        sample_to_row = payload.get("sample_to_row")
        if not isinstance(sample_to_row, dict):
            raise ValueError(
                f"finalized parquet store at {self.root}: manifest.json is missing "
                "a valid 'sample_to_row' index (corrupt manifest)"
            )
        rows_path = self.root / "rows.parquet"
        if not sample_to_row:
            # A genuinely empty store writes no rows.parquet and records
            # count==0. An empty index alongside on-disk rows / a positive count
            # is a corrupt or mismatched manifest.
            count = self._finalized_count()
            if rows_path.exists() or count:
                raise ValueError(
                    f"finalized parquet store at {self.root}: empty sample_to_row "
                    f"but rows.parquet present={rows_path.exists()} / count={count} "
                    "(corrupt manifest)"
                )
            return
        if not rows_path.exists():
            raise FileNotFoundError(
                f"finalized parquet store at {self.root} declares "
                f"{len(sample_to_row)} sample(s) but rows.parquet is missing"
            )
        table_rows = pq.read_table(str(rows_path)).to_pylist()
        if len(table_rows) != len(sample_to_row):
            raise ValueError(
                f"parquet store at {self.root} inconsistent: rows.parquet has "
                f"{len(table_rows)} row(s) but manifest lists {len(sample_to_row)}"
            )
        rebuilt: list[dict[str, Any]] = []
        for i, raw in enumerate(table_rows):
            sid = raw.get("sample_id")
            if sample_to_row.get(sid) != i:
                raise ValueError(
                    f"parquet store at {self.root}: row {i} sample_id {sid!r} "
                    "does not match manifest sample_to_row"
                )
            # Field set = columns that are not sample_id / a __shape/__dtype
            # sidecar. The finalize union-schema gives absent fields a None
            # triple; keep a field only when its whole triple is present, and
            # fail loud on a partially-None (corrupt) triple.
            clean: dict[str, Any] = {"sample_id": sid}
            for col, val in raw.items():
                if col == "sample_id" or col.endswith("__shape") or col.endswith("__dtype"):
                    continue
                shape_v = raw.get(f"{col}__shape")
                dtype_v = raw.get(f"{col}__dtype")
                present = [val is not None, shape_v is not None, dtype_v is not None]
                if not any(present):
                    continue  # field absent for this row (union-schema None columns)
                if not all(present):
                    raise ValueError(
                        f"parquet store at {self.root}: corrupt row {sid!r} — field "
                        f"{col!r} has a partially-None payload/shape/dtype triple"
                    )
                clean[col] = val
                clean[f"{col}__shape"] = shape_v
                clean[f"{col}__dtype"] = dtype_v
            rebuilt.append(clean)
        self._rows = rebuilt
        self._index = dict(sample_to_row)

    def put(self, sample_id: str, tensors: Mapping[str, torch.Tensor]) -> None:
        if self._finalized:
            raise RuntimeError(
                f"store at {self.root} already finalized — cannot put more samples"
            )
        if sample_id in self._index:
            return
        row: dict[str, Any] = {"sample_id": sample_id}
        for k, v in tensors.items():
            t = v.detach().contiguous().cpu()
            row[k] = t.numpy().tobytes()
            row[f"{k}__shape"] = list(t.shape)
            row[f"{k}__dtype"] = str(t.dtype)
        self._index[sample_id] = len(self._rows)
        self._rows.append(row)

    def contains(self, sample_id: str) -> bool:
        return sample_id in self._index

    def iter_keys(self) -> Iterator[str]:
        yield from self._index.keys()

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        if sample_id not in self._index:
            raise KeyError(sample_id)
        row = self._rows[self._index[sample_id]]
        out: dict[str, torch.Tensor] = {}
        for k, v in row.items():
            if k == "sample_id" or k.endswith("__shape") or k.endswith("__dtype"):
                continue
            shape = tuple(row[f"{k}__shape"])
            dtype = _dtype_from_str(row[f"{k}__dtype"])
            out[k] = torch.frombuffer(v, dtype=dtype).reshape(shape).clone()
        return out

    def finalize(self) -> Path:
        if self._rows:
            schema_keys = sorted({k for r in self._rows for k in r.keys()})
            arrays = {k: pa.array([r.get(k) for r in self._rows]) for k in schema_keys}
            table = pa.table(arrays)
            tmp = self.root / "rows.parquet.tmp"
            pq.write_table(table, str(tmp))
            os.replace(tmp, self.root / "rows.parquet")
        idx_path = self.root / "manifest.json"
        idx_path.write_text(
            json.dumps({"sample_to_row": self._index}), encoding="utf-8"
        )
        self._write_header()
        manifest = self._write_manifest({"count": len(self._index)})
        self._finalized = True
        return manifest


# ---------------------------------------------------------------- helpers


def open_artifact_store(
    root: str | Path,
    *,
    expected_header: ArtifactHeader | Mapping[str, Any] | None = None,
    allow_stale: bool = False,
    backend: str | None = None,
) -> ArtifactStoreProtocol:
    """Open a finalized artifact store for reading.

    Header mismatches raise :class:`StaleArtifactError` unless ``allow_stale``
    is set. ``backend`` is auto-detected from ``MANIFEST_COMPLETE.json`` when
    omitted.
    """
    root = Path(root)
    manifest_path = root / "MANIFEST_COMPLETE.json"
    if not manifest_path.exists():
        raise ArtifactIncompleteError(
            f"no MANIFEST_COMPLETE.json at {root}; producer did not finalize"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend = backend or manifest.get("backend", "safetensors-shards")
    on_disk_header = ArtifactHeader.from_dict(manifest.get("header") or {})

    if expected_header is not None:
        if isinstance(expected_header, Mapping):
            expected_header = ArtifactHeader.from_dict(expected_header)
        diffs = on_disk_header.disagreements(expected_header)
        if diffs and not allow_stale:
            raise StaleArtifactError(
                f"artifact header mismatch at {root}: fields differ {diffs}. "
                f"Pass --allow-stale-artifact / allow_stale=True to override."
            )

    if backend == "safetensors-shards":
        store: ArtifactStoreBase = SafetensorsShardStore(root, header=on_disk_header)
    elif backend == "memmap-fixed":
        store = MemmapFixedStore(root, header=on_disk_header)
    elif backend == "parquet-rows":
        store = ParquetRowStore(root, header=on_disk_header)
    else:
        raise ValueError(f"unknown artifact_store backend {backend!r}")
    store._finalized = True
    return store


__all__ = [
    "ArtifactHeader",
    "ArtifactIncompleteError",
    "ArtifactStoreProtocol",
    "MemmapFixedStore",
    "ParquetRowStore",
    "SafetensorsShardStore",
    "StaleArtifactError",
    "open_artifact_store",
]
