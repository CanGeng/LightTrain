"""Sequence packing — EOS-glue rows up to ``seq_len``.

Returns rows with three int64 vectors:

* ``input_ids``    — packed tokens, padded with ``pad_id`` to ``seq_len``
* ``position_ids`` — position within the original document, restarted at 0 per doc
* ``document_ids`` — sequential id within the packed sample (0..k-1 for k docs)

This lets attention mask the boundary between glued documents at training
time (block-level mask) without having to track real boundaries elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class SequencePacker:
    seq_len: int
    eos_id: int
    pad_id: int = 0
    keep_short: bool = True

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.eos_id = int(self.eos_id)
        self.pad_id = int(self.pad_id)

    def pack(self, rows: Iterable[dict]) -> Iterator[dict]:
        """Greedy pack — flush whenever the buffer would exceed ``seq_len``."""
        buf_ids: list[int] = []
        buf_pos: list[int] = []
        buf_doc: list[int] = []
        doc_idx = 0

        def emit() -> dict:
            nonlocal buf_ids, buf_pos, buf_doc, doc_idx
            ids = list(buf_ids[: self.seq_len])
            pos = list(buf_pos[: self.seq_len])
            doc = list(buf_doc[: self.seq_len])
            pad = self.seq_len - len(ids)
            if pad > 0:
                ids += [self.pad_id] * pad
                pos += [0] * pad
                doc += [-1] * pad
            buf_ids = []
            buf_pos = []
            buf_doc = []
            doc_idx = 0
            return {
                "input_ids": ids,
                "position_ids": pos,
                "document_ids": doc,
            }

        for row in rows:
            ids = list(row.get("input_ids") or [])
            if not ids:
                continue
            # Append EOS if the row didn't already end in one.
            if ids[-1] != self.eos_id:
                ids.append(self.eos_id)
            if len(buf_ids) + len(ids) > self.seq_len:
                if buf_ids:
                    yield emit()
                # If a single doc is bigger than seq_len, truncate it head-first.
                if len(ids) > self.seq_len:
                    ids = ids[: self.seq_len]
            buf_ids.extend(ids)
            buf_pos.extend(range(len(ids)))
            buf_doc.extend([doc_idx] * len(ids))
            doc_idx += 1
            if len(buf_ids) >= self.seq_len:
                yield emit()

        if buf_ids and self.keep_short:
            yield emit()


__all__ = ["SequencePacker"]
