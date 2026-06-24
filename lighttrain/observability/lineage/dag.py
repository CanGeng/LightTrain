"""Lineage DAG utilities — cycle detection + graph export.

Cycle detection walks ``derived_from`` + ``produced_by`` edges
backwards up to ``K`` hops (default 4). A hit means a node was derived from an
artifact produced by the same run — the classic self-feeding loop.

Graph export emits Mermaid (default) or Graphviz DOT. Both are pure-text
formats so the CLI can pipe them or write them next to the run dir.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from .store import LineageStore

_log = logging.getLogger(__name__)


@dataclass
class CycleHit:
    """A self-feeding loop discovered during cycle detection."""

    node_id: int
    via_run_id: str
    depth: int

    def __str__(self) -> str:  # pragma: no cover — debug only
        return f"cycle: node#{self.node_id} via run={self.via_run_id} depth={self.depth}"


def cycle_check(
    store: LineageStore,
    start_node: int,
    *,
    current_run_id: str,
    k: int = 4,
) -> list[CycleHit]:
    """Walk back through ``derived_from``+``produced_by`` edges K hops.

    A hit occurs when an ancestor's ``run_id`` matches ``current_run_id`` —
    i.e. the current run is consuming an artifact that an earlier step in the
    same run produced. Up-to-K BFS, so the cost is bounded.
    """
    hits: list[CycleHit] = []
    visited: set[int] = set()
    frontier: list[tuple[int, int]] = [(start_node, 0)]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for node_id, depth in frontier:
            if node_id in visited or depth > k:
                continue
            visited.add(node_id)
            node = store.get_node(node_id)
            if node and node.get("run_id") == current_run_id and node_id != start_node:
                hits.append(CycleHit(node_id=node_id, via_run_id=current_run_id, depth=depth))
            for kind in ("derived_from", "produced_by"):
                for src in store.parents(node_id, edge_kind=kind):
                    nxt.append((src, depth + 1))
        frontier = nxt
    return hits


def apply_cycle_policy(
    hits: list[CycleHit],
    *,
    self_feeding: Literal["allowed", "warn", "forbid"] = "warn",
    require_external_signal: bool = False,
    external_signal_present: bool = False,
    logger: Any = None,
) -> None:
    """Translate cycle hits into ``info`` / ``warning`` / ``raise`` per policy.

    Three policy levels: ``info`` / ``warn`` / ``raise`` — wire via ``self_feeding``.
    """
    if not hits:
        return
    msg = f"lineage: self-feeding cycle detected ({len(hits)} hit(s))"
    if require_external_signal and not external_signal_present:
        msg += " (no external judge/reward signal on the loop)"
        # forced upgrade to warn if not already forbid
        if self_feeding == "allowed":
            self_feeding = "warn"
    if self_feeding == "allowed":
        if logger is not None and hasattr(logger, "log_text"):
            try:
                logger.log_text(msg, 0)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "lineage.dag: logger.log_text failed while emitting "
                    "self-feeding cycle notice; suppressing",
                    exc_info=True,
                )
        return
    if self_feeding == "warn":
        warnings.warn(msg, stacklevel=2)
        return
    raise RuntimeError(msg)


# ---------------------------------------------------------------- graph export


_MERMAID_KIND_SHAPE = {
    "artifact": ("[(", ")]"),
    "checkpoint": ("[/", "/]"),
    "config": ("{", "}"),
    "run": ("((", "))"),
    "frozen_step": ("[[", "]]"),
}


def to_mermaid(store: LineageStore, root_id: int, *, depth: int = 5) -> str:
    """BFS outward up to ``depth`` hops in both directions; emit Mermaid graph."""
    visited_nodes: set[int] = set()
    visited_edges: set[tuple[int, int, str]] = set()
    frontier: list[tuple[int, int]] = [(root_id, 0)]
    lines: list[str] = ["graph TD"]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for node_id, d in frontier:
            if node_id in visited_nodes or d > depth:
                continue
            visited_nodes.add(node_id)
            node = store.get_node(node_id)
            if not node:
                continue
            shape_l, shape_r = _MERMAID_KIND_SHAPE.get(node["kind"], ("[", "]"))
            label = _label_for(node)
            lines.append(f'    n{node_id}{shape_l}"{label}"{shape_r}')
            for edge in store.edges_from(node_id):
                key = (edge["src"], edge["dst"], edge["kind"])
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                lines.append(
                    f"    n{edge['src']} -->|{edge['kind']}| n{edge['dst']}"
                )
                nxt.append((int(edge["dst"]), d + 1))
            for edge in store.edges_to(node_id):
                key = (edge["src"], edge["dst"], edge["kind"])
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                lines.append(
                    f"    n{edge['src']} -->|{edge['kind']}| n{edge['dst']}"
                )
                nxt.append((int(edge["src"]), d + 1))
        frontier = nxt
    return "\n".join(lines)


def to_dot(store: LineageStore, root_id: int, *, depth: int = 5) -> str:
    visited_nodes: set[int] = set()
    visited_edges: set[tuple[int, int, str]] = set()
    frontier: list[tuple[int, int]] = [(root_id, 0)]
    lines: list[str] = ["digraph lineage {", "    rankdir=LR;"]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for node_id, d in frontier:
            if node_id in visited_nodes or d > depth:
                continue
            visited_nodes.add(node_id)
            node = store.get_node(node_id)
            if not node:
                continue
            label = _label_for(node)
            shape = {
                "artifact": "box",
                "checkpoint": "folder",
                "config": "note",
                "run": "ellipse",
                "frozen_step": "diamond",
            }.get(node["kind"], "box")
            lines.append(f'    n{node_id} [label="{label}", shape={shape}];')
            for edge in store.edges_from(node_id):
                key = (edge["src"], edge["dst"], edge["kind"])
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                lines.append(
                    f'    n{edge["src"]} -> n{edge["dst"]} [label="{edge["kind"]}"];'
                )
                nxt.append((int(edge["dst"]), d + 1))
            for edge in store.edges_to(node_id):
                key = (edge["src"], edge["dst"], edge["kind"])
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                lines.append(
                    f'    n{edge["src"]} -> n{edge["dst"]} [label="{edge["kind"]}"];'
                )
                nxt.append((int(edge["src"]), d + 1))
        frontier = nxt
    lines.append("}")
    return "\n".join(lines)


def _label_for(node: dict[str, Any]) -> str:
    kind = node["kind"]
    name = node.get("name") or "?"
    version = node.get("version") or ""
    return f"{kind}:{name}:{version}" if version else f"{kind}:{name}"


__all__ = ["CycleHit", "apply_cycle_policy", "cycle_check", "to_dot", "to_mermaid"]
