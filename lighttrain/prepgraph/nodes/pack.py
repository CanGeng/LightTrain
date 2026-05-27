"""Pack PrepNode — EOS-glue tokenized rows to fixed ``seq_len``.

Wraps :class:`lighttrain.data.packing.SequencePacker`. Output rows have
``input_ids``, ``position_ids``, ``document_ids``, plus ``labels`` mirrored
from the source rows (with mask values preserved per document).
"""

from __future__ import annotations

from typing import Any

from ...registry import register
from ...data.packing._packer import SequencePacker
from ..node import NodeResult, PrepNode, RunContext


@register("prep_node", "pack")
class PackNode(PrepNode):
    """EOS-glue rows up to ``seq_len``.

    Config keys:

    * ``seq_len``: required int.
    * ``eos_id``: required int.
    * ``pad_id``: int (default 0).
    * ``label_ignore``: int (default ``-100``) — used for labels-mode masking
      on pad positions.
    * ``keep_short``: bool (default True) — emit final partial buffer.
    """

    kind = "pack"
    schema_kind = "packed_rows"

    def _pack_with_labels(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seq_len = int(self.config.get("seq_len", 0))
        eos_id = int(self.config.get("eos_id", 0))
        pad_id = int(self.config.get("pad_id", 0))
        label_ignore = int(self.config.get("label_ignore", -100))
        keep_short = bool(self.config.get("keep_short", True))

        # Materialize labels alongside ids by running an identical packer over
        # paired streams. Cheap because rows are already in memory.
        packer = SequencePacker(
            seq_len=seq_len, eos_id=eos_id, pad_id=pad_id, keep_short=keep_short
        )
        # Build a parallel rows-with-labels stream.
        parallel = []
        for r in rows:
            ids = list(r.get("input_ids") or [])
            labels = list(r.get("labels") or list(ids))
            if len(labels) != len(ids):
                labels = list(ids)
            parallel.append({"input_ids": ids, "labels": labels})

        out_rows: list[dict[str, Any]] = []
        # Run the packer once for ids and reproduce labels by mirroring buffer
        # logic inside the same loop.
        buf_ids: list[int] = []
        buf_lab: list[int] = []
        buf_pos: list[int] = []
        buf_doc: list[int] = []
        doc_idx = 0

        def emit() -> dict[str, Any]:
            nonlocal buf_ids, buf_lab, buf_pos, buf_doc, doc_idx
            ids = list(buf_ids[:seq_len])
            labs = list(buf_lab[:seq_len])
            pos = list(buf_pos[:seq_len])
            doc = list(buf_doc[:seq_len])
            pad = seq_len - len(ids)
            if pad > 0:
                ids += [pad_id] * pad
                labs += [label_ignore] * pad
                pos += [0] * pad
                doc += [-1] * pad
            buf_ids.clear()
            buf_lab.clear()
            buf_pos.clear()
            buf_doc.clear()
            doc_idx = 0
            return {
                "input_ids": ids,
                "labels": labs,
                "position_ids": pos,
                "document_ids": doc,
                "attention_mask": [1 if d >= 0 else 0 for d in doc],
                "modality": "text",
            }

        for r in parallel:
            ids = list(r["input_ids"])
            labs = list(r["labels"])
            if not ids:
                continue
            if ids[-1] != eos_id:
                ids.append(eos_id)
                labs.append(eos_id)
            if len(buf_ids) + len(ids) > seq_len:
                if buf_ids:
                    out_rows.append(emit())
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                    labs = labs[:seq_len]
            buf_ids.extend(ids)
            buf_lab.extend(labs)
            buf_pos.extend(range(len(ids)))
            buf_doc.extend([doc_idx] * len(ids))
            doc_idx += 1
            if len(buf_ids) >= seq_len:
                out_rows.append(emit())

        if buf_ids and keep_short:
            out_rows.append(emit())
        return out_rows

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"PackNode {self.name!r}: requires upstream input.")
        if int(self.config.get("seq_len", 0)) <= 0:
            raise ValueError(f"PackNode {self.name!r}: `seq_len` must be > 0.")
        upstream = ctx.upstream[self.inputs[0]]
        rows = list(upstream.rows or [])
        out_rows = self._pack_with_labels(rows)
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=out_rows,
            extras={"row_count": len(out_rows)},
        )


__all__ = ["PackNode"]
