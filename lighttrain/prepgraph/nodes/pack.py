"""Pack PrepNode — combine tokenized rows into fixed-``seq_len`` sequences.

Three packing strategies, selected by the ``strategy`` config key:

* ``concat_chunk`` (**default**) — the classic *padding-free* baseline: glue every
  document (+EOS) into one stream and slice at fixed ``seq_len`` boundaries. Zero
  padding (except a final partial chunk), but documents straddling a boundary are
  truncated. This is what "packing" usually means and what most baselines report.
* ``next_fit`` — greedy next-fit with padding: append a document to the current
  buffer and, when the next would overflow, **flush the short buffer padded to
  ``seq_len``** and start fresh; only documents longer than ``seq_len`` are cut.
  Low-memory streaming behavior; wastes padding. (This was the historical default.)
* ``best_fit`` — Best-Fit-Decreasing bin packing from *Fewer Truncations Improve
  Language Modeling* (Ding et al., ICML 2024): place each whole document into the
  bin with the smallest residual capacity that still fits; documents are never
  split unless longer than ``seq_len``. Lowest truncation, but combinatorial and
  only toy-scale-validated here — **opt-in**.

All three emit the built-in ``packed_rows`` schema (``input_ids``, ``labels``,
``position_ids``, ``document_ids``, ``attention_mask``) and surface standardized
metrics on ``NodeResult.extras`` (``truncation_rate``, ``token_utilization``,
``n_truncated_docs``, ``n_sequences``, …), visible via ``prep-status --extras``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import register
from ..node import NodeResult, PrepNode, RunContext

_DEFAULT_STRATEGY = "concat_chunk"
_STRATEGIES = ("concat_chunk", "next_fit", "best_fit")


# ---------------------------------------------------------------------------
# shared helpers (ported from the best-fit packing reproduction experiment,
# whose pure-Python BFD reference is bit-exact parity-tested at 234/234 bins)
# ---------------------------------------------------------------------------
def _units_from_rows(
    rows: list[Mapping[str, Any]], *, eos_id: int
) -> list[tuple[list[int], list[int]]]:
    """Turn tokenized rows into ``(ids+eos, labels+eos)`` units in corpus order.

    Empty rows are skipped. EOS is appended unless the row already ends in it.
    """
    units: list[tuple[list[int], list[int]]] = []
    for r in rows:
        ids = list(r.get("input_ids") or [])
        if not ids:
            continue
        labels = list(r.get("labels") or list(ids))
        if len(labels) != len(ids):
            labels = list(ids)
        if ids[-1] != eos_id:
            ids = ids + [eos_id]
            labels = labels + [eos_id]
        units.append((ids, labels))
    return units


def _emit_bin(
    items: list[tuple[list[int], list[int]]],
    *,
    seq_len: int,
    pad_id: int,
    label_ignore: int,
) -> dict[str, Any]:
    """Render one packed sequence from a list of ``(ids, labels)`` items."""
    input_ids: list[int] = []
    labels: list[int] = []
    position_ids: list[int] = []
    document_ids: list[int] = []
    for doc_id, (ids, labs) in enumerate(items):
        input_ids.extend(ids)
        labels.extend(labs)
        position_ids.extend(range(len(ids)))
        document_ids.extend([doc_id] * len(ids))
    pad = seq_len - len(input_ids)
    if pad > 0:
        input_ids += [pad_id] * pad
        labels += [label_ignore] * pad
        position_ids += [0] * pad
        document_ids += [-1] * pad
    return {
        "input_ids": input_ids[:seq_len],
        "labels": labels[:seq_len],
        "position_ids": position_ids[:seq_len],
        "document_ids": document_ids[:seq_len],
        "attention_mask": [1 if d >= 0 else 0 for d in document_ids[:seq_len]],
        "modality": "text",
    }


def best_fit_decreasing(
    units: list[tuple[list[int], list[int]]], *, seq_len: int
) -> tuple[list[list[tuple[list[int], list[int]]]], int]:
    """Pure BFD. Returns ``(bins, n_truncated_docs)``.

    ``bins`` is a list (creation order) of lists of ``(ids, labels)`` items in
    placement order. ``n_truncated_docs`` counts documents cut because their unit
    exceeded ``seq_len``. Items are sorted longest-first (ties by original index);
    each is placed in the existing bin with the smallest residual capacity that
    still fits, else a new bin is opened.
    """
    items: list[tuple[int, int, list[int], list[int]]] = []  # (len, idx, ids, labs)
    n_truncated = 0
    item_idx = 0
    for ids, labs in units:
        if len(ids) <= seq_len:
            items.append((len(ids), item_idx, ids, labs))
            item_idx += 1
        else:
            n_truncated += 1
            for s in range(0, len(ids), seq_len):
                items.append(
                    (len(ids[s : s + seq_len]), item_idx, ids[s : s + seq_len], labs[s : s + seq_len])
                )
                item_idx += 1

    items.sort(key=lambda t: (-t[0], t[1]))

    bins_remaining: list[int] = []
    bins_items: list[list[tuple[list[int], list[int]]]] = []
    for length, _idx, ids, labs in items:
        best_b = -1
        best_rem = seq_len + 1
        for b, rem in enumerate(bins_remaining):
            if rem >= length and rem < best_rem:
                best_rem = rem
                best_b = b
        if best_b < 0:
            bins_remaining.append(seq_len - length)
            bins_items.append([(ids, labs)])
        else:
            bins_remaining[best_b] -= length
            bins_items[best_b].append((ids, labs))
    return bins_items, n_truncated


def _pack_stats(
    *, n_documents: int, n_truncated: int, rows: list[dict[str, Any]], seq_len: int
) -> dict[str, Any]:
    """Standardized packing metrics, surfaced on ``NodeResult.extras``."""
    real = sum(sum(1 for d in r["document_ids"] if d >= 0) for r in rows)
    total = len(rows) * seq_len
    pad = total - real
    return {
        "row_count": len(rows),
        "n_documents": n_documents,
        "n_sequences": len(rows),
        "n_truncated_docs": n_truncated,
        "truncation_rate": (n_truncated / n_documents) if n_documents else 0.0,
        "real_tokens": real,
        "pad_tokens": pad,
        "token_utilization": (real / total) if total else 0.0,
        "seq_len": seq_len,
    }


@register("prep_node", "pack")
class PackNode(PrepNode):
    """Pack tokenized rows into fixed-``seq_len`` sequences.

    Config keys:

    * ``seq_len``: required int.
    * ``strategy``: ``concat_chunk`` (default) | ``next_fit`` | ``best_fit``.
    * ``eos_id``: int (default 0).
    * ``pad_id``: int (default 0).
    * ``label_ignore``: int (default ``-100``) — mask value on pad positions.
    * ``keep_short``: bool (default True) — emit the final partial buffer/chunk
      (``concat_chunk`` and ``next_fit`` only).
    """

    kind = "pack"
    schema_kind = "packed_rows"

    # -- strategies: each returns (out_rows, n_documents, n_truncated_docs) -----

    def _pack_concat_chunk(
        self, units, *, seq_len, pad_id, label_ignore, keep_short
    ) -> tuple[list[dict[str, Any]], int, int]:
        # Concatenate every (ids+eos) into one stream, slice at seq_len boundaries.
        stream_ids: list[int] = []
        stream_labels: list[int] = []
        stream_doc: list[int] = []
        spans: list[tuple[int, int]] = []
        for doc_idx, (ids, labs) in enumerate(units):
            start = len(stream_ids)
            stream_ids.extend(ids)
            stream_labels.extend(labs)
            stream_doc.extend([doc_idx] * len(ids))
            spans.append((start, len(stream_ids)))

        # A document is truncated iff its span crosses a chunk boundary.
        n_trunc = sum(
            1 for (s, e) in spans if e > s and (s // seq_len) != ((e - 1) // seq_len)
        )

        out_rows: list[dict[str, Any]] = []
        total = len(stream_ids)
        for c in range(0, total, seq_len):
            chunk_ids = stream_ids[c : c + seq_len]
            chunk_labs = stream_labels[c : c + seq_len]
            chunk_doc = stream_doc[c : c + seq_len]
            if len(chunk_ids) < seq_len and not keep_short and c + seq_len > total:
                break
            n = len(chunk_ids)
            pad = seq_len - n
            out_rows.append(
                {
                    "input_ids": chunk_ids + [pad_id] * pad,
                    "labels": chunk_labs + [label_ignore] * pad,
                    "position_ids": list(range(n)) + [0] * pad,
                    "document_ids": chunk_doc + [-1] * pad,
                    "attention_mask": [1] * n + [0] * pad,
                    "modality": "text",
                }
            )
        return out_rows, len(units), n_trunc

    def _pack_best_fit(
        self, units, *, seq_len, pad_id, label_ignore
    ) -> tuple[list[dict[str, Any]], int, int]:
        bins, n_trunc = best_fit_decreasing(units, seq_len=seq_len)
        out_rows = [
            _emit_bin(b, seq_len=seq_len, pad_id=pad_id, label_ignore=label_ignore)
            for b in bins
        ]
        return out_rows, len(units), n_trunc

    def _pack_next_fit(
        self, units, *, seq_len, pad_id, label_ignore, keep_short
    ) -> tuple[list[dict[str, Any]], int, int]:
        # Historical greedy-pad-flush behavior, preserved bit-for-bit.
        out_rows: list[dict[str, Any]] = []
        buf_ids: list[int] = []
        buf_lab: list[int] = []
        buf_pos: list[int] = []
        buf_doc: list[int] = []
        doc_idx = 0
        n_trunc = 0

        def emit() -> dict[str, Any]:
            nonlocal doc_idx
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

        for ids, labs in units:
            ids = list(ids)
            labs = list(labs)
            if len(buf_ids) + len(ids) > seq_len:
                if buf_ids:
                    out_rows.append(emit())
                if len(ids) > seq_len:
                    n_trunc += 1
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
        return out_rows, len(units), n_trunc

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"PackNode {self.name!r}: requires upstream input.")
        seq_len = int(self.config.get("seq_len", 0))
        if seq_len <= 0:
            raise ValueError(f"PackNode {self.name!r}: `seq_len` must be > 0.")
        strategy = str(self.config.get("strategy", _DEFAULT_STRATEGY))
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"PackNode {self.name!r}: unknown strategy {strategy!r}; "
                f"choose one of {list(_STRATEGIES)}."
            )
        eos_id = int(self.config.get("eos_id", 0))
        pad_id = int(self.config.get("pad_id", 0))
        label_ignore = int(self.config.get("label_ignore", -100))
        keep_short = bool(self.config.get("keep_short", True))

        rows = list(ctx.upstream[self.inputs[0]].rows or [])
        units = _units_from_rows(rows, eos_id=eos_id)

        if strategy == "concat_chunk":
            out_rows, n_docs, n_trunc = self._pack_concat_chunk(
                units, seq_len=seq_len, pad_id=pad_id,
                label_ignore=label_ignore, keep_short=keep_short,
            )
        elif strategy == "best_fit":
            out_rows, n_docs, n_trunc = self._pack_best_fit(
                units, seq_len=seq_len, pad_id=pad_id, label_ignore=label_ignore,
            )
        else:  # next_fit
            out_rows, n_docs, n_trunc = self._pack_next_fit(
                units, seq_len=seq_len, pad_id=pad_id,
                label_ignore=label_ignore, keep_short=keep_short,
            )

        extras = _pack_stats(
            n_documents=n_docs, n_truncated=n_trunc, rows=out_rows, seq_len=seq_len
        )
        extras["strategy"] = strategy
        if ctx.log is not None:
            ctx.log.print(
                f"[pack:{strategy}] {n_docs} docs -> {len(out_rows)} seqs | "
                f"truncation_rate={extras['truncation_rate']:.4f} | "
                f"token_utilization={extras['token_utilization']:.4f}"
            )
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=out_rows,
            extras=extras,
        )


__all__ = ["PackNode", "best_fit_decreasing"]
