"""Validate PrepNode — sanity-check rows, emit a tiny report.

Computes:

* length histogram (over ``input_ids``)
* OOV ratio (vs configurable ``vocab_size``)
* label-mask coverage (fraction of non-``label_ignore`` labels)
* row count

The report is dumped as ``report.json`` in the staging dir; rows pass
through unchanged so downstream nodes can consume them.
"""

from __future__ import annotations

import json
from typing import Any

from lighttrain.data.prepgraph.node import NodeResult, PrepNode, RunContext
from lighttrain.registry import register


def _histogram(values: list[int], bins: list[int]) -> list[int]:
    counts = [0] * (len(bins) + 1)
    for v in values:
        placed = False
        for i, b in enumerate(bins):
            if v <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


@register("prep_node", "validate")
class ValidateNode(PrepNode):
    """Validate tokenized rows; emit a JSON report.

    Config keys:

    * ``vocab_size``: optional int — when set, OOV = id >= vocab_size.
    * ``label_ignore``: int (default ``-100``).
    * ``hist_bins``: list[int] (default ``[64, 128, 256, 512, 1024, 2048]``).
    * ``min_rows``: int (default 1) — fail if upstream produced fewer rows.
    * ``max_oov_ratio``: float (default 1.0) — soft limit.
    """

    kind = "validate"
    schema_kind = "validate_report"

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"ValidateNode {self.name!r}: requires upstream input.")
        upstream = ctx.upstream[self.inputs[0]]
        rows = list(upstream.rows or [])

        vocab_size = self.config.get("vocab_size")
        label_ignore = int(self.config.get("label_ignore", -100))
        bins = list(self.config.get("hist_bins", [64, 128, 256, 512, 1024, 2048]))
        min_rows = int(self.config.get("min_rows", 1))
        max_oov_ratio = float(self.config.get("max_oov_ratio", 1.0))

        lengths: list[int] = []
        oov_total = 0
        token_total = 0
        label_kept = 0
        label_total = 0
        for row in rows:
            ids = row.get("input_ids") or []
            lengths.append(len(ids))
            token_total += len(ids)
            if vocab_size is not None:
                oov_total += sum(1 for x in ids if int(x) >= int(vocab_size))
            labels = row.get("labels") or []
            for x in labels:
                label_total += 1
                if int(x) != label_ignore:
                    label_kept += 1

        report: dict[str, Any] = {
            "rows": len(rows),
            "tokens": token_total,
            "length_histogram": _histogram(lengths, bins),
            "length_bins": bins,
            "label_keep_ratio": label_kept / max(1, label_total),
        }
        if vocab_size is not None:
            ratio = oov_total / max(1, token_total)
            report["oov_total"] = oov_total
            report["oov_ratio"] = ratio
            if ratio > max_oov_ratio:
                raise RuntimeError(
                    f"ValidateNode {self.name!r}: OOV ratio {ratio:.3f} > limit "
                    f"{max_oov_ratio:.3f}."
                )
        if len(rows) < min_rows:
            raise RuntimeError(
                f"ValidateNode {self.name!r}: only {len(rows)} rows, need >= {min_rows}."
            )

        # Persist the report under the staging dir.
        (ctx.store_root / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=rows,
            extras={"report": report},
        )


__all__ = ["ValidateNode"]
