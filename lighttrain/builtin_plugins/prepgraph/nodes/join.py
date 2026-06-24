"""JoinNode — PrepGraph artifact join.

Joins upstream rows against one or more ArtifactStores keyed by
``sample.id``. The behaviour mirrors :class:`ArtifactJoinedDataset` but
runs at prep time (not at training time), so the joined tensors live on
disk in the materialized output of this node and the downstream training
loop reads them with zero per-batch store lookups.

Why duplicate the join semantics?

* ``ArtifactJoinedDataset`` is the training-time fallback path (lazy
  per-batch join). PrepGraph-time ``join`` is the eager path: the artifact
  tensors are merged into rows and committed alongside the source dataset.
* Same ``missing`` policy (``require | drop | fill_zero``) so users learn
  one mental model.

Config schema::

    - name: train_with_teacher
      kind: join
      inputs: [tokenized]
      missing: require               # or drop / fill_zero
      sample_id_key: id
      allow_stale_artifact: false
      stores:
        - store: artifacts/teacher_v1   # path to artifact root
          namespace: teacher             # → aux.teacher.<key>
          missing: require              # per-store override (optional)
          expected_header:              # optional StaleArtifact check
            model_id: tiny_lm_teacher_v1
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from lighttrain.artifacts import ArtifactHeader
from lighttrain.builtin_plugins.artifacts.store import open_artifact_store
from lighttrain.data.core._schema import derive_sample_id
from lighttrain.prepgraph.node import NodeResult, PrepNode, RunContext
from lighttrain.registry import register

_log = logging.getLogger(__name__)

_MISSING_REQUIRE = "require"
_MISSING_DROP = "drop"
_MISSING_FILL = "fill_zero"


@register("prep_node", "join")
class JoinNode(PrepNode):
    """Join upstream rows with ArtifactStore tensors by ``sample_id``."""

    kind = "join"
    schema_kind = "rows"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        stores = self.config.get("stores") or []
        if not stores:
            raise ValueError(
                f"JoinNode {self.name!r}: at least one entry in `stores:` is required"
            )
        self._store_specs: list[dict[str, Any]] = [dict(s) for s in stores]
        self.default_missing = str(self.config.get("missing", _MISSING_REQUIRE))
        self.sample_id_key = str(self.config.get("sample_id_key", "id"))
        self.allow_stale_artifact = bool(
            self.config.get("allow_stale_artifact", False)
        )

    def run(self, ctx: RunContext) -> NodeResult:
        if len(self.inputs) != 1:
            raise ValueError(
                f"JoinNode {self.name!r} expects exactly 1 input, got {self.inputs!r}"
            )
        upstream_name = self.inputs[0]
        upstream = ctx.upstream.get(upstream_name)
        if upstream is None or upstream.rows is None:
            raise RuntimeError(
                f"JoinNode {self.name!r}: upstream {upstream_name!r} has no rows"
            )

        stores: list[tuple[dict[str, Any], Any]] = []
        for spec in self._store_specs:
            cfg = dict(spec)
            root = cfg.pop("store", cfg.pop("path", None))
            if not root:
                raise ValueError(
                    "each entry in JoinNode.stores requires `store` (path)"
                )
            allow = bool(cfg.pop("allow_stale_artifact", self.allow_stale_artifact))
            expected_raw = cfg.pop("expected_header", None)
            expected = (
                ArtifactHeader.from_dict(dict(expected_raw))
                if isinstance(expected_raw, Mapping)
                else None
            )
            store = open_artifact_store(
                root, expected_header=expected, allow_stale=allow
            )
            cfg.setdefault("namespace", _default_namespace(Path(str(root)).name))
            cfg.setdefault("missing", self.default_missing)
            stores.append((cfg, store))

        out_rows: list[dict[str, Any]] = []
        for row in upstream.rows:
            joined = _join_one(row, stores, self.sample_id_key)
            if joined is not None:
                out_rows.append(joined)

        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=out_rows,
            extras={"row_count": len(out_rows)},
        )


def _join_one(
    row: Mapping[str, Any],
    stores: Iterable[tuple[dict[str, Any], Any]],
    sample_id_key: str,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = dict(row)
    sid = str(merged.get(sample_id_key) or derive_sample_id(merged))
    merged.setdefault(sample_id_key, sid)
    for cfg, store in stores:
        namespace = cfg["namespace"]
        missing = cfg["missing"]
        if not store.contains(sid):
            if missing == _MISSING_DROP:
                return None
            if missing == _MISSING_FILL:
                for k, shape_str in store.header.field_schema.items():
                    shape = _parse_shape(shape_str)
                    merged[f"aux.{namespace}.{k}"] = (
                        torch.zeros(shape).tolist()
                    )
                continue
            raise KeyError(
                f"JoinNode: sample {sid!r} not present in store at {store.root}. "
                f"Set missing='drop' to skip or 'fill_zero' to substitute zeros."
            )
        tensors = store.get(sid)
        for k, v in tensors.items():
            # Persist as plain list so downstream JSONL serializers handle it;
            # downstream materialize / packers may re-tensorize as needed.
            merged[f"aux.{namespace}.{k}"] = (
                v.tolist() if hasattr(v, "tolist") else v
            )
    return merged


def _default_namespace(name: str) -> str:
    return name.split("_")[0] or "aux"


def _parse_shape(shape_str: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(shape_str)
        if isinstance(parsed, tuple):
            return parsed
        if isinstance(parsed, list):
            return tuple(int(x) for x in parsed)
        if isinstance(parsed, int):
            return (parsed,)
    except Exception:  # noqa: BLE001
        _log.warning("join: shape literal %r could not be parsed; treating as empty shape", shape_str, exc_info=True)
    return ()


__all__ = ["JoinNode"]
