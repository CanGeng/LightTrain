"""Coverage-gap tests for ``lighttrain.data.prepgraph.dag``.

Pins every branch that the existing ``test_dag_and_runner.py`` leaves uncovered:

* Line 34  – ``from_config`` rejects a non-Mapping spec.
* Line 37  – ``from_config`` rejects a spec with no/empty ``nodes:`` key.
* Line 39  – ``from_config`` accepts ``nodes:`` as a Mapping (dict of dicts).
* Line 44  – node entry that is not a Mapping raises ``ValueError``.
* Line 50  – node entry missing ``name`` or ``kind`` raises ``ValueError``.
* Line 69  – resolved object not a ``PrepNode`` subclass raises ``TypeError``.
* Line 74  – instance ``kind`` != declared ``kind`` raises ``ValueError``.
* Line 82  – declared terminal not present in nodes raises ``ValueError``.
* Line 128 – ``parents_of`` returns inputs of the named node.
* Lines 133-137 – ``_auto_terminals`` returns only sink nodes (no children).

General edge-case sweep:
* ``topo_order`` flattens layers in topological order.
* Single-node graph (no inputs, no terminals key) auto-detects its terminal.
* Diamond DAG (A→B, A→C, B→D, C→D) layers and ``parents_of``.
* ``_auto_terminals`` on a linear chain picks only the tail.
* ``_auto_terminals`` on a graph where every node has children returns ``[]``.
"""

from __future__ import annotations

import pytest

from lighttrain.data.prepgraph.dag import PrepGraph, _auto_terminals
from lighttrain.data.prepgraph.node import NodeResult, PrepNode, RunContext

# ---------------------------------------------------------------------------
# Stub nodes — all resolved via ``_target_`` so the global registry is untouched
# ---------------------------------------------------------------------------


class _DummyNode(PrepNode):
    """Minimal no-op node used for structural DAG tests."""

    kind = "dummy"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover
        return NodeResult(fingerprint="x", rows=[])


class _AltKindNode(PrepNode):
    """Node whose class-level ``kind`` differs from what the YAML declares.

    Used to trigger the kind-mismatch guard (dag.py line 74).
    """

    kind = "alt_kind"   # intentionally different from "dummy"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover
        return NodeResult(fingerprint="x", rows=[])


class _NotANode:
    """Does NOT subclass PrepNode — triggers the isinstance guard (line 69)."""

    def __init__(self, *, name: str, inputs: list, config: dict) -> None:
        self.name = name


# Dotted _target_ strings used in specs below
_DUMMY_TARGET = f"{__name__}._DummyNode"
_ALT_KIND_TARGET = f"{__name__}._AltKindNode"
_NOT_A_NODE_TARGET = f"{__name__}._NotANode"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_entry(name: str, inputs: list[str] | None = None) -> dict:
    return {
        "name": name,
        "kind": "dummy",
        "_target_": _DUMMY_TARGET,
        "inputs": list(inputs or []),
    }


def _make_graph(*node_names_with_inputs: tuple[str, list[str]]) -> PrepGraph:
    """Build a PrepGraph from (name, inputs) pairs; terminals auto-detected."""
    entries = [_node_entry(n, ins) for n, ins in node_names_with_inputs]
    return PrepGraph.from_config({"nodes": entries})


# ===========================================================================
# Line 34 – spec must be a Mapping
# ===========================================================================


def test_invariant_from_config_rejects_non_mapping_spec() -> None:
    """``from_config`` raises ``ValueError`` when the spec is not a Mapping
    (dag.py line 34).
    """
    with pytest.raises(ValueError, match="must be a mapping"):
        PrepGraph.from_config("not a dict")  # type: ignore[arg-type]


def test_invariant_from_config_rejects_list_spec() -> None:
    """Lists are not Mappings; ``from_config`` should reject them (line 34)."""
    with pytest.raises(ValueError, match="must be a mapping"):
        PrepGraph.from_config([{"nodes": []}])  # type: ignore[arg-type]


# ===========================================================================
# Line 37 – spec with missing or empty nodes: key
# ===========================================================================


def test_invariant_from_config_rejects_missing_nodes_key() -> None:
    """``from_config`` raises when ``nodes:`` key is absent (line 37)."""
    with pytest.raises(ValueError, match="non-empty"):
        PrepGraph.from_config({})


def test_invariant_from_config_rejects_empty_nodes_list() -> None:
    """``from_config`` raises when ``nodes:`` is an empty list (line 37)."""
    with pytest.raises(ValueError, match="non-empty"):
        PrepGraph.from_config({"nodes": []})


def test_invariant_from_config_rejects_none_nodes() -> None:
    """``from_config`` raises when ``nodes:`` is explicitly None (line 37)."""
    with pytest.raises(ValueError, match="non-empty"):
        PrepGraph.from_config({"nodes": None})


# ===========================================================================
# Line 39 – nodes: may be a Mapping (dict of dicts) instead of a list
# ===========================================================================


def test_invariant_from_config_accepts_mapping_nodes() -> None:
    """When ``nodes:`` is a Mapping, ``from_config`` converts values to a list
    (dag.py line 39) and builds the graph correctly.
    """
    spec = {
        "nodes": {
            "first": _node_entry("A"),
            "second": _node_entry("B", ["A"]),
        },
        "terminals": ["B"],
    }
    graph = PrepGraph.from_config(spec)
    assert set(graph.nodes) == {"A", "B"}
    assert graph.terminals == ["B"]


def test_pin_current_behavior_mapping_nodes_all_values_used() -> None:
    """Pin: when nodes is a Mapping, ALL dict values are used as entries.

    NOTE: The dict keys are ignored; what matters is the ``name`` field
    inside each value dict. This is the current documented behaviour (line 39).
    """
    spec = {
        "nodes": {
            "ignored_key_1": _node_entry("X"),
            "ignored_key_2": _node_entry("Y", ["X"]),
        },
    }
    graph = PrepGraph.from_config(spec)
    assert "X" in graph.nodes
    assert "Y" in graph.nodes


# ===========================================================================
# Line 44 – non-Mapping node entry raises ValueError
# ===========================================================================


def test_invariant_from_config_rejects_non_mapping_node_entry() -> None:
    """A bare string node entry is not a Mapping; ``from_config`` raises
    ``ValueError`` (dag.py line 44).
    """
    spec = {"nodes": ["not_a_dict"]}
    with pytest.raises(ValueError, match="must be a mapping"):
        PrepGraph.from_config(spec)


def test_invariant_from_config_rejects_integer_node_entry() -> None:
    """An integer node entry triggers the same guard (line 44)."""
    spec = {"nodes": [42]}
    with pytest.raises(ValueError, match="must be a mapping"):
        PrepGraph.from_config(spec)


# ===========================================================================
# Line 50 – node entry missing name or kind
# ===========================================================================


def test_invariant_from_config_rejects_entry_without_name() -> None:
    """A node entry with no ``name`` field raises ``ValueError`` (line 50)."""
    spec = {"nodes": [{"kind": "dummy", "_target_": _DUMMY_TARGET}]}
    with pytest.raises(ValueError, match="needs `name` and `kind`"):
        PrepGraph.from_config(spec)


def test_invariant_from_config_rejects_entry_without_kind() -> None:
    """A node entry with no ``kind`` field raises ``ValueError`` (line 50)."""
    spec = {"nodes": [{"name": "A", "_target_": _DUMMY_TARGET}]}
    with pytest.raises(ValueError, match="needs `name` and `kind`"):
        PrepGraph.from_config(spec)


def test_invariant_from_config_rejects_entry_with_empty_name() -> None:
    """An empty ``name`` string is falsy and triggers the same guard (line 50)."""
    spec = {
        "nodes": [{"name": "", "kind": "dummy", "_target_": _DUMMY_TARGET}]
    }
    with pytest.raises(ValueError, match="needs `name` and `kind`"):
        PrepGraph.from_config(spec)


def test_invariant_from_config_rejects_entry_with_empty_kind() -> None:
    """An empty ``kind`` string is falsy and triggers the same guard (line 50)."""
    spec = {
        "nodes": [{"name": "A", "kind": "", "_target_": _DUMMY_TARGET}]
    }
    with pytest.raises(ValueError, match="needs `name` and `kind`"):
        PrepGraph.from_config(spec)


# ===========================================================================
# Line 69 – resolved object not a PrepNode raises TypeError
# ===========================================================================


def test_invariant_from_config_rejects_non_prepnode_target() -> None:
    """When ``_target_`` resolves to a class that does not subclass PrepNode,
    ``from_config`` raises ``TypeError`` (dag.py line 69).
    """
    spec = {
        "nodes": [
            {
                "name": "A",
                "kind": "dummy",
                "_target_": _NOT_A_NODE_TARGET,
            }
        ],
        "terminals": ["A"],
    }
    with pytest.raises(TypeError, match="must subclass PrepNode"):
        PrepGraph.from_config(spec)


# ===========================================================================
# Line 74 – instance kind != declared kind raises ValueError
# ===========================================================================


def test_invariant_from_config_rejects_kind_mismatch() -> None:
    """When a PrepNode subclass has class-level ``kind`` that differs from the
    YAML ``kind:`` field, ``from_config`` raises ``ValueError`` (line 74).
    """
    spec = {
        "nodes": [
            {
                "name": "A",
                "kind": "dummy",          # declared kind
                "_target_": _ALT_KIND_TARGET,  # class kind = "alt_kind"
            }
        ],
        "terminals": ["A"],
    }
    with pytest.raises(ValueError, match="does not match class kind"):
        PrepGraph.from_config(spec)


# ===========================================================================
# Line 82 – terminal not in nodes raises ValueError
# ===========================================================================


def test_invariant_from_config_rejects_unknown_terminal() -> None:
    """Declaring a terminal that does not exist in nodes raises ``ValueError``
    (dag.py line 82).
    """
    spec = {
        "nodes": [_node_entry("A")],
        "terminals": ["MISSING"],
    }
    with pytest.raises(ValueError, match="not in nodes"):
        PrepGraph.from_config(spec)


def test_invariant_from_config_rejects_multiple_unknown_terminals() -> None:
    """Multiple unknown terminals — even the first one triggers line 82."""
    spec = {
        "nodes": [_node_entry("A")],
        "terminals": ["A", "GHOST"],
    }
    with pytest.raises(ValueError, match="not in nodes"):
        PrepGraph.from_config(spec)


# ===========================================================================
# Line 128 – parents_of returns inputs list
# ===========================================================================


def test_invariant_parents_of_returns_empty_for_root_node() -> None:
    """``parents_of`` on a node with no inputs returns an empty list (line 128)."""
    graph = _make_graph(("A", []))
    assert graph.parents_of("A") == []


def test_invariant_parents_of_returns_inputs_for_internal_node() -> None:
    """``parents_of`` returns the declared inputs of a node (line 128)."""
    graph = _make_graph(("A", []), ("B", ["A"]))
    assert graph.parents_of("B") == ["A"]


def test_invariant_parents_of_diamond_node_has_two_parents() -> None:
    """A diamond merge node has two parents (line 128)."""
    graph = _make_graph(
        ("A", []),
        ("B", ["A"]),
        ("C", ["A"]),
        ("D", ["B", "C"]),
    )
    result = graph.parents_of("D")
    assert set(result) == {"B", "C"}
    assert len(result) == 2


# ===========================================================================
# Lines 133-137 – _auto_terminals function
# ===========================================================================


def test_invariant_auto_terminals_single_node() -> None:
    """A single node with no inputs is its own terminal (lines 133-137)."""
    graph = _make_graph(("Solo", []))
    assert "Solo" in graph.terminals


def test_invariant_auto_terminals_picks_sinks_in_chain() -> None:
    """In a linear chain A→B→C, only C is a sink (lines 133-137)."""
    graph = _make_graph(("A", []), ("B", ["A"]), ("C", ["B"]))
    assert graph.terminals == ["C"]
    assert "A" not in graph.terminals
    assert "B" not in graph.terminals


def test_invariant_auto_terminals_diamond_picks_merge_node() -> None:
    """In a diamond A→B,C→D, only D is a sink (lines 133-137)."""
    graph = _make_graph(
        ("A", []),
        ("B", ["A"]),
        ("C", ["A"]),
        ("D", ["B", "C"]),
    )
    assert graph.terminals == ["D"]


def test_invariant_auto_terminals_two_disconnected_sinks() -> None:
    """Two independent chains each produce a sink; both appear in terminals."""
    spec = {
        "nodes": [
            _node_entry("R1"),
            _node_entry("T1", ["R1"]),
            _node_entry("R2"),
            _node_entry("T2", ["R2"]),
        ],
    }
    graph = PrepGraph.from_config(spec)
    assert sorted(graph.terminals) == ["T1", "T2"]


def test_invariant_auto_terminals_direct_function() -> None:
    """Call ``_auto_terminals`` directly on a synthetic nodes mapping."""

    class _FakeNode:
        def __init__(self, inputs: list[str]) -> None:
            self.inputs = inputs

    nodes = {
        "A": _FakeNode([]),
        "B": _FakeNode(["A"]),
        "C": _FakeNode(["A"]),
        "D": _FakeNode(["B", "C"]),
    }
    result = _auto_terminals(nodes)  # type: ignore[arg-type]
    assert result == ["D"]


def test_invariant_auto_terminals_no_sinks_returns_empty() -> None:
    """If every node is referenced by another node as an input,
    ``_auto_terminals`` returns an empty list.

    NOTE: this is a degenerate case (cycle or self-reference); the
    function itself does not validate — it simply returns [] when all
    nodes appear in the referenced set (lines 133-137).
    """

    class _FakeNode:
        def __init__(self, inputs: list[str]) -> None:
            self.inputs = inputs

    # A and B reference each other (cycle); both are referenced → no sinks.
    nodes = {
        "A": _FakeNode(["B"]),
        "B": _FakeNode(["A"]),
    }
    result = _auto_terminals(nodes)  # type: ignore[arg-type]
    assert result == []


# ===========================================================================
# topo_order: flatten layers
# ===========================================================================


def test_invariant_topo_order_flattens_layers() -> None:
    """``topo_order()`` returns a flat list consistent with the layer order."""
    graph = _make_graph(
        ("A", []),
        ("B", ["A"]),
        ("C", ["A"]),
        ("D", ["B", "C"]),
    )
    order = graph.topo_order()
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
    assert set(order) == {"A", "B", "C", "D"}


def test_invariant_topo_order_single_node() -> None:
    """Single-node graph has a topo_order of length 1."""
    graph = _make_graph(("Solo", []))
    assert graph.topo_order() == ["Solo"]


# ===========================================================================
# General correctness: graph structure after construction
# ===========================================================================


def test_invariant_nodes_dict_populated_after_construction() -> None:
    """``graph.nodes`` is a dict keyed by node name after ``from_config``."""
    spec = {
        "nodes": [
            _node_entry("X"),
            _node_entry("Y", ["X"]),
        ],
        "terminals": ["Y"],
    }
    graph = PrepGraph.from_config(spec)
    assert "X" in graph.nodes
    assert "Y" in graph.nodes
    assert isinstance(graph.nodes["X"], PrepNode)
    assert isinstance(graph.nodes["Y"], PrepNode)


def test_invariant_explicit_terminals_stored() -> None:
    """When ``terminals:`` is given, ``graph.terminals`` matches it exactly."""
    spec = {
        "nodes": [
            _node_entry("A"),
            _node_entry("B", ["A"]),
            _node_entry("C", ["A"]),
        ],
        "terminals": ["B", "C"],
    }
    graph = PrepGraph.from_config(spec)
    assert sorted(graph.terminals) == ["B", "C"]


def test_invariant_layers_cover_all_nodes() -> None:
    """Every node in ``graph.nodes`` appears in exactly one layer."""
    spec = {
        "nodes": [
            _node_entry("A"),
            _node_entry("B", ["A"]),
            _node_entry("C", ["B"]),
        ],
    }
    graph = PrepGraph.from_config(spec)
    flat = [n for layer in graph.layers for n in layer]
    assert sorted(flat) == sorted(graph.nodes.keys())


def test_invariant_extra_config_keys_preserved_in_node() -> None:
    """Remaining keys after popping name/kind/inputs become the node's config.

    This exercises the ``node_config = dict(entry)`` path that passes custom
    fields into the PrepNode constructor.
    """
    entry = {
        "name": "A",
        "kind": "dummy",
        "_target_": _DUMMY_TARGET,
        "extra_param": 42,
    }
    graph = PrepGraph.from_config({"nodes": [entry]})
    assert "extra_param" in graph.nodes["A"].config
    assert graph.nodes["A"].config["extra_param"] == 42
