"""Fixed-shape memmap cache for packed sequences.

Optimal storage for ``packed`` data where every row has identical
``seq_len`` — three int64 columns (input_ids, position_ids, document_ids) plus
an int8 attention_mask. The header is a JSON sidecar; the .bin is contiguous
``np.memmap`` so DataLoader workers can mmap zero-copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HEADER_NAME = "header.json"
DATA_NAME = "data.bin"


@dataclass
class MemmapHeader:
    seq_len: int
    n_rows: int
    fields: list[str]
    dtypes: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "n_rows": self.n_rows,
            "fields": list(self.fields),
            "dtypes": dict(self.dtypes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemmapHeader":
        return cls(
            seq_len=int(d["seq_len"]),
            n_rows=int(d["n_rows"]),
            fields=list(d["fields"]),
            dtypes=dict(d["dtypes"]),
        )


def _row_bytes(header: MemmapHeader) -> int:
    total = 0
    for f in header.fields:
        total += header.seq_len * np.dtype(header.dtypes[f]).itemsize
    return total


def write_memmap(
    out_dir: str | Path,
    rows: Iterable[dict],
    *,
    seq_len: int,
    fields: tuple[str, ...] = ("input_ids", "position_ids", "document_ids"),
    dtypes: dict[str, str] | None = None,
) -> MemmapHeader:
    """Write a fixed-shape memmap. Each row must have all ``fields``.

    Returns the finalized header. Atomicity: data + header both go through
    a temp file rename. The PrepGraph runner adds its own MANIFEST_COMPLETE
    on top of this.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtypes = dtypes or {f: "int64" for f in fields}
    rows_list = list(rows)
    n = len(rows_list)
    header = MemmapHeader(
        seq_len=int(seq_len), n_rows=n, fields=list(fields), dtypes=dtypes
    )
    bin_tmp = out_dir / (DATA_NAME + ".tmp")
    arrays = {
        f: np.zeros((n, seq_len), dtype=np.dtype(dtypes[f])) for f in fields
    }
    for i, row in enumerate(rows_list):
        for f in fields:
            v = row.get(f)
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.dtype(dtypes[f]))
            arr = arr[:seq_len]
            arrays[f][i, : len(arr)] = arr
    with bin_tmp.open("wb") as fh:
        for f in fields:
            fh.write(arrays[f].tobytes(order="C"))
    bin_path = out_dir / DATA_NAME
    if bin_path.exists():
        bin_path.unlink()
    bin_tmp.rename(bin_path)

    header_tmp = out_dir / (HEADER_NAME + ".tmp")
    header_tmp.write_text(json.dumps(header.to_dict(), indent=2), encoding="utf-8")
    header_path = out_dir / HEADER_NAME
    if header_path.exists():
        header_path.unlink()
    header_tmp.rename(header_path)
    return header


def read_header(out_dir: str | Path) -> MemmapHeader | None:
    path = Path(out_dir) / HEADER_NAME
    if not path.exists():
        return None
    return MemmapHeader.from_dict(json.loads(path.read_text(encoding="utf-8")))


class MemmapDataset:
    """Map-style dataset over a memmap directory written by ``write_memmap``."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        header = read_header(self.out_dir)
        if header is None:
            raise FileNotFoundError(f"No memmap header at {self.out_dir}")
        self.header = header
        bin_path = self.out_dir / DATA_NAME
        if not bin_path.exists():
            raise FileNotFoundError(f"No memmap data at {bin_path}")
        self._views: dict[str, np.memmap] = {}
        offset = 0
        n, T = header.n_rows, header.seq_len
        for f in header.fields:
            dtype = np.dtype(header.dtypes[f])
            count = n * T
            view = np.memmap(
                str(bin_path),
                dtype=dtype,
                mode="r",
                offset=offset,
                shape=(n, T),
            )
            self._views[f] = view
            offset += count * dtype.itemsize

    def __len__(self) -> int:
        return self.header.n_rows

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        out: dict = {}
        for f, view in self._views.items():
            out[f] = view[idx].tolist()
        # Build a default attention_mask: 1 where input_ids != 0.
        if "attention_mask" not in out and "input_ids" in out:
            ids = self._views["input_ids"][idx]
            mask = (ids != 0).astype(np.int64)
            out["attention_mask"] = mask.tolist()
        # labels mirror input_ids by default (causal LM); -100 on padding.
        if "labels" not in out and "input_ids" in out:
            ids = self._views["input_ids"][idx]
            labels = ids.astype(np.int64).copy()
            labels[ids == 0] = -100
            out["labels"] = labels.tolist()
        return out


__all__ = ["MemmapDataset", "MemmapHeader", "read_header", "write_memmap"]
