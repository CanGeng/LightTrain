"""Retention policy + GC.

Multi-policy union: a node survives if ANY policy retains it. Candidates are
first marked ``deprecated = 1`` with ``deprecated_ts = now()``; only on a
second GC pass after ``ttl_deprecated_hours`` (default 24h) are their
``payload_path`` directories deleted.

``keep_best_by_metric`` is wired against ``evaluated_by`` edges' payload.
When no evaluation edges have been written yet it returns the empty set.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .store import LineageStore


@dataclass
class RetentionPolicy:
    """Multi-strategy retention.

    Policies are evaluated **as a union** — a node is kept if any policy
    retains it. Empty / ``None`` fields are ignored.
    """

    keep_last: int | None = None
    keep_best_by_metric: dict[str, Any] | None = None  # {metric, mode, k}
    keep_tagged: bool = True
    keep_pinned: bool = True
    ttl_days: float | None = None
    ttl_deprecated_hours: float = 24.0


@dataclass
class GCReport:
    deprecated: list[int] = field(default_factory=list)
    deleted: list[int] = field(default_factory=list)
    paths_deleted: list[str] = field(default_factory=list)


def gc_artifacts(
    store: LineageStore,
    *,
    policy: RetentionPolicy | None = None,
    kind: str = "artifact",
    name: str | None = None,
    now: float | None = None,
    dry_run: bool = False,
    delete_paths: bool = True,
) -> GCReport:
    """Mark / sweep one kind of node.

    Pass ``name`` to scope to a single artifact name's versions; otherwise GC
    runs across all names of that kind. Paths under each node's ``payload_path``
    are removed during the second-pass sweep — set ``delete_paths=False`` for
    metadata-only retention.
    """
    policy = policy or RetentionPolicy(keep_last=3, keep_tagged=True, keep_pinned=True)
    now = now or time.time()
    report = GCReport()

    nodes = [n for n in store.iter_nodes(kind=kind) if name is None or n["name"] == name]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        by_name.setdefault(n["name"] or "", []).append(n)

    for grp in by_name.values():
        grp.sort(key=lambda x: x["ts"], reverse=True)  # newest first

        keep_ids: set[int] = set()
        if policy.keep_last:
            for n in grp[: policy.keep_last]:
                keep_ids.add(n["id"])
        if policy.keep_tagged:
            for n in grp:
                if n["tags"]:
                    keep_ids.add(n["id"])
        if policy.keep_pinned:
            for n in grp:
                if n.get("pinned"):
                    keep_ids.add(n["id"])
        if policy.keep_best_by_metric:
            keep_ids |= _keep_best_by_metric(store, grp, policy.keep_best_by_metric)

        for n in grp:
            nid = int(n["id"])
            if nid in keep_ids:
                continue
            # TTL — even otherwise-retained nodes drop after ttl_days.
            if policy.ttl_days and (now - n["ts"]) > policy.ttl_days * 86400.0:
                pass  # let the eviction proceed
            elif not policy.keep_last and not policy.ttl_days:
                continue

            if not n.get("deprecated"):
                if not dry_run:
                    store.invalidate(nid)
                report.deprecated.append(nid)
                continue
            # Already deprecated → check grace period
            dts = n.get("deprecated_ts") or now
            if now - dts < policy.ttl_deprecated_hours * 3600.0:
                continue
            if not dry_run:
                if delete_paths:
                    p = n.get("payload_path")
                    if p and Path(p).exists():
                        shutil.rmtree(p, ignore_errors=True)
                        report.paths_deleted.append(str(p))
                store.delete_node(nid)
            report.deleted.append(nid)

    return report


def _keep_best_by_metric(
    store: LineageStore, grp: Iterable[dict[str, Any]], spec: dict[str, Any]
) -> set[int]:
    """Scan ``evaluated_by`` edge payloads for the spec'd metric.

    Returns the top-K node ids in the desired direction.
    Returns an empty set when no evaluation edges carry metric payloads.
    """
    metric = spec.get("metric")
    mode = spec.get("mode", "max")
    k = int(spec.get("k", 1))
    scored: list[tuple[float, int]] = []
    for n in grp:
        for edge in store.edges_to(int(n["id"]), kind="evaluated_by"):
            try:
                import json

                payload = json.loads(edge.get("payload") or "{}")
            except Exception:  # noqa: BLE001
                payload = {}
            v = payload.get(metric)
            if isinstance(v, (int, float)):
                scored.append((float(v), int(n["id"])))
    if not scored:
        return set()
    scored.sort(reverse=(mode == "max"))
    return {nid for _, nid in scored[:k]}


def prune_orphans(store: LineageStore, *, dry_run: bool = False) -> list[int]:
    """Drop nodes whose ``payload_path`` no longer exists on disk."""
    removed: list[int] = []
    for n in store.iter_nodes():
        p = n.get("payload_path")
        if not p or not Path(p).exists():
            if p and not Path(p).exists():
                removed.append(int(n["id"]))
                if not dry_run:
                    store.delete_node(int(n["id"]))
    return removed


__all__ = ["GCReport", "RetentionPolicy", "gc_artifacts", "prune_orphans"]
