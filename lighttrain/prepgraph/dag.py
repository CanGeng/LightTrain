"""DAG construction + topological grouping for PrepGraph.

Reads the ``prep_graph:`` block from a recipe and builds a graph of resolved
``PrepNode`` instances. Entry point: ``PrepGraph.from_config(spec)``.

The DAG returns ``layers``: a list of node lists where every node within a
layer can run in parallel (its inputs are all in earlier layers). The runner
walks layers serially while running nodes within a layer concurrently.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config._resolver import resolve as _resolve
from .node import PrepNode


@dataclass
class PrepGraph:
    nodes: dict[str, PrepNode] = field(default_factory=dict)
    terminals: list[str] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> PrepGraph:
        if not isinstance(spec, Mapping):
            raise ValueError("prep_graph spec must be a mapping.")
        raw_nodes = spec.get("nodes")
        if not raw_nodes:
            raise ValueError("prep_graph: requires a non-empty `nodes:` list.")
        if isinstance(raw_nodes, Mapping):
            raw_nodes = list(raw_nodes.values())

        nodes: dict[str, PrepNode] = {}
        for entry in raw_nodes:
            if not isinstance(entry, Mapping):
                raise ValueError(f"prep_graph node entry must be a mapping, got {type(entry).__name__}.")
            entry = dict(entry)
            name = entry.pop("name", None)
            kind = entry.pop("kind", None)
            inputs = entry.pop("inputs", None) or []
            if not name or not kind:
                raise ValueError("Each prep_graph node needs `name` and `kind`.")
            if name in nodes:
                raise ValueError(f"Duplicate node name in prep_graph: {name!r}.")
            target = entry.pop("_target_", None)
            # Remaining keys = node config.
            node_config = dict(entry)
            spec_for_resolver: dict[str, Any] = {
                "params": {
                    "name": name,
                    "inputs": list(inputs),
                    "config": node_config,
                },
            }
            if target:
                spec_for_resolver["_target_"] = target
            else:
                spec_for_resolver["name"] = kind
            node = _resolve(spec_for_resolver, category="prep_node")
            if not isinstance(node, PrepNode):
                raise TypeError(
                    f"prep_node {name!r} (kind={kind!r}) must subclass PrepNode, got {type(node).__name__}."
                )
            # Sanity: kind on the instance must match the YAML.
            if node.kind and node.kind != kind:
                raise ValueError(
                    f"Node {name!r}: declared kind={kind!r} does not match class kind={node.kind!r}."
                )
            nodes[name] = node

        terminals = list(spec.get("terminals") or _auto_terminals(nodes))
        for t in terminals:
            if t not in nodes:
                raise ValueError(f"prep_graph terminal {t!r} not in nodes.")

        graph = cls(nodes=nodes, terminals=terminals)
        graph.layers = graph._topo_layers()
        return graph

    # ---- topology ---------------------------------------------------------

    def _topo_layers(self) -> list[list[str]]:
        # Validate inputs exist + collect indegree.
        indeg: dict[str, int] = {n: 0 for n in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for name, node in self.nodes.items():
            for u in node.inputs:
                if u not in self.nodes:
                    raise ValueError(
                        f"Node {name!r} references unknown input {u!r}."
                    )
                children[u].append(name)
                indeg[name] += 1

        # Kahn's algorithm in layered form.
        layers: list[list[str]] = []
        ready = deque(sorted(n for n, d in indeg.items() if d == 0))
        seen = 0
        while ready:
            layer = list(ready)
            ready.clear()
            for n in layer:
                for c in sorted(children[n]):
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        ready.append(c)
                seen += 1
            layers.append(layer)
        if seen != len(self.nodes):
            cycle = [n for n, d in indeg.items() if d > 0]
            raise ValueError(f"prep_graph has a cycle involving: {sorted(cycle)}")
        return layers

    # ---- queries ----------------------------------------------------------

    def topo_order(self) -> list[str]:
        return [n for layer in self.layers for n in layer]

    def parents_of(self, name: str) -> Sequence[str]:
        return list(self.nodes[name].inputs)


def _auto_terminals(nodes: Mapping[str, PrepNode]) -> list[str]:
    """Return nodes with no children (sinks). Used when ``terminals:`` omitted."""
    referenced: set[str] = set()
    for n in nodes.values():
        for u in n.inputs:
            referenced.add(u)
    return sorted(n for n in nodes if n not in referenced)


__all__ = ["PrepGraph"]
