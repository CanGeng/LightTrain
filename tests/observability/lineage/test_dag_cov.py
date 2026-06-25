"""Edge-case coverage tests for ``lighttrain.observability.lineage.dag``.

Drives uncovered lines to covered:
  * Lines 89-92: ``apply_cycle_policy`` ``allowed`` branch with a functioning
    logger (89-90) and a logger whose ``log_text`` raises (91-92 except path).
  * Line 130: ``to_mermaid`` — root node id not in store → ``if not node: continue``.
  * Lines 137-138: ``to_mermaid`` — duplicate edges_from key skipped.
  * Lines 147-148, 151: ``to_mermaid`` — edges_to block (inbound edge from parent
    discovered while visiting root) emits line + queues predecessor.
  * Lines 165, 169: ``to_dot`` — depth-truncation (d > depth) and missing-node guard.
  * Lines 180-187: ``to_dot`` — edges_from block: dedup skip + emit + enqueue.
  * Lines 189-196: ``to_dot`` — edges_to block: inbound edge emit + enqueue.
  * Line 137 (mermaid dedup) already partially tested in test_dag.py for edges_from;
    here we also exercise the edges_to dedup path (line 145-146).

Style mirrors tests/eval/test_suite.py and tests/trainers/test_base_seams.py.
"""
from __future__ import annotations

import warnings

import pytest

from lighttrain.observability.lineage.dag import (
    CycleHit,
    apply_cycle_policy,
    to_dot,
    to_mermaid,
)

RUN_A = "run-cov-A"
RUN_B = "run-cov-B"


# --------------------------------------------------------------------------- #
# Stubs                                                                       #
# --------------------------------------------------------------------------- #


class _CapturingLogger:
    """Stub logger that records calls to ``log_text``.

    Used to exercise the ``allowed`` + logger branch (lines 88-90).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def log_text(self, msg: str, step: int) -> None:
        self.calls.append((msg, step))


class _ExplodingLogger:
    """Stub logger whose ``log_text`` raises on every call.

    Used to exercise the except-suppression path (lines 91-92).
    """

    def log_text(self, msg: str, step: int) -> None:
        raise ValueError("simulated log_text failure")


# --------------------------------------------------------------------------- #
# apply_cycle_policy — allowed + logger paths (lines 88-97)                  #
# --------------------------------------------------------------------------- #


def test_apply_cycle_policy_allowed_with_logger_calls_log_text() -> None:
    """``allowed`` policy with a working ``logger.log_text`` must call it exactly once
    with the cycle message at step 0.

    Pins lines 88-90 of dag.py: the function checks ``logger is not None`` and
    ``hasattr(logger, "log_text")``, then calls ``logger.log_text(msg, 0)``.
    No Python warning must be emitted.
    """
    logger = _CapturingLogger()
    hit = CycleHit(node_id=1, via_run_id=RUN_A, depth=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([hit], self_feeding="allowed", logger=logger)

    assert len(logger.calls) == 1, "log_text should be called exactly once"
    msg, step = logger.calls[0]
    assert "self-feeding" in msg
    assert step == 0
    assert caught == [], "allowed policy must not emit a Python warning"


def test_apply_cycle_policy_allowed_with_exploding_logger_does_not_raise() -> None:
    """``allowed`` + ``logger.log_text`` raises → exception suppressed, no Python
    warning emitted, no re-raise.

    Pins lines 91-96 of dag.py: the ``except Exception`` block must swallow the
    error and then the function returns normally (``return`` on line 97 is the
    ``allowed`` branch exit).
    """
    logger = _ExplodingLogger()
    hit = CycleHit(node_id=2, via_run_id=RUN_A, depth=1)

    # Must not raise despite the logger exploding.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([hit], self_feeding="allowed", logger=logger)

    # The policy is "allowed" — swallowing the logger error must not escalate
    # into a warning either.
    assert caught == []


def test_apply_cycle_policy_allowed_no_logger_silent() -> None:
    """``allowed`` policy with no logger and no hits → silent no-op.

    Ensures the ``if logger is not None`` guard does not cause any side-effect
    when no logger is supplied.
    """
    hit = CycleHit(node_id=3, via_run_id=RUN_A, depth=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([hit], self_feeding="allowed")  # no logger arg
    assert caught == []


def test_apply_cycle_policy_allowed_logger_without_log_text_is_ignored() -> None:
    """A logger object that does NOT have ``log_text`` must be silently ignored.

    Pins the ``hasattr(logger, "log_text")`` guard on line 88: only objects
    that actually expose ``log_text`` are called; others pass through silently.
    """
    class _NoLogTextLogger:
        pass

    hit = CycleHit(node_id=4, via_run_id=RUN_A, depth=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([hit], self_feeding="allowed", logger=_NoLogTextLogger())
    assert caught == []


# --------------------------------------------------------------------------- #
# to_mermaid — missing root node (line 130)                                   #
# --------------------------------------------------------------------------- #


def test_to_mermaid_missing_root_node_returns_header_only(lineage_store_factory) -> None:
    """``to_mermaid`` with a root_id that has no node row in the store emits
    only the ``"graph TD"`` header — the ``if not node: continue`` guard
    on line 130 short-circuits without crashing.
    """
    store = lineage_store_factory()
    out = to_mermaid(store, root_id=99999, depth=2)
    assert out == "graph TD", f"Expected only header line, got: {out!r}"


# --------------------------------------------------------------------------- #
# to_mermaid — edges_from deduplication (line 137)                            #
# --------------------------------------------------------------------------- #


def test_to_mermaid_edges_from_dedup_via_revisit(lineage_store_factory) -> None:
    """When the BFS visits a node that was already visited (because it was
    discovered via two different paths), the ``if node_id in visited_nodes``
    guard short-circuits. But the inner edge-dedup (line 137: ``if key in
    visited_edges``) fires when the *same edge* is encountered from two
    different node visits.

    Setup: triangle A→B, A→C, B→C. Start from A.
    - A is processed first; edges_from(A) = [A→B, A→C] → both emitted.
    - B is queued (d=1), processed; edges_from(B) = [B→C] → C queued (d=2).
    - C is queued (d=1 from A), processed first (BFS ordering); edges_from(C)=[].
      edges_to(C) = [B→C, A→C].
      The A→C edge key was already added when we processed A's edges_from.
      So the ``if key in visited_edges: continue`` path fires for A→C here.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="tri-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="tri-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="tri-C", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(a, c, kind="derived_from")
    store.add_edge(b, c, kind="derived_from")

    out = to_mermaid(store, root_id=a, depth=5)
    # Each edge should appear exactly once.
    assert out.count(f"n{a} -->|derived_from| n{b}") == 1
    assert out.count(f"n{a} -->|derived_from| n{c}") == 1
    assert out.count(f"n{b} -->|derived_from| n{c}") == 1


# --------------------------------------------------------------------------- #
# to_mermaid — edges_to block (lines 147-148, 151)                            #
# --------------------------------------------------------------------------- #


def test_to_mermaid_edges_to_inbound_edge_emitted_and_predecessor_queued(
    lineage_store_factory,
) -> None:
    """When ``to_mermaid`` processes the root node and ``store.edges_to(root)``
    returns an inbound edge (src→root), that edge should appear in the output
    AND the src node should be queued for subsequent processing.

    This specifically exercises lines 147-148, 151: ``visited_edges.add(key)``,
    the ``lines.append(...)`` for the inbound edge, and ``nxt.append(src, d+1)``.

    Setup: parent → root (derived_from). Start BFS from root.
    edges_from(root) = []. edges_to(root) = [parent→root]. So the inbound
    edge is encountered in the edges_to loop, lines 143-151.
    """
    store = lineage_store_factory()
    parent = store.upsert_node(kind="run", name="mermaid-parent", run_id=RUN_A)
    root = store.upsert_node(kind="artifact", name="mermaid-root", run_id=RUN_A)
    store.add_edge(parent, root, kind="produced_by")

    out = to_mermaid(store, root_id=root, depth=3)
    lines = out.splitlines()
    assert lines[0] == "graph TD"
    # The inbound edge must appear exactly once (edges_to path).
    assert out.count(f"n{parent} -->|produced_by| n{root}") == 1
    # The parent node label should also appear (it was queued and visited).
    assert any(f"n{parent}" in ln for ln in lines)


def test_to_mermaid_edges_to_dedup_skips_already_visited_key(
    lineage_store_factory,
) -> None:
    """The edges_to dedup path (lines 145-146) must skip an inbound-edge key
    that was already added when the same edge was seen via a previous node's
    edges_from pass.

    Setup: A → B → C. BFS from A.
    - A processed: edges_from(A)=[A→B] → key (A,B,derived_from) added.
      B queued at d=1.
    - B processed: edges_from(B)=[B→C] → key (B,C) added. C queued at d=2.
      edges_to(B)=[A→B] → key (A,B,derived_from) already in visited_edges →
      lines 145-146 ``if key in visited_edges: continue`` fires.
    Each edge must appear exactly once.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="chain-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="chain-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="chain-C", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(b, c, kind="derived_from")

    out = to_mermaid(store, root_id=a, depth=5)
    assert out.count(f"n{a} -->|derived_from| n{b}") == 1
    assert out.count(f"n{b} -->|derived_from| n{c}") == 1


# --------------------------------------------------------------------------- #
# to_mermaid — depth truncation                                               #
# --------------------------------------------------------------------------- #


def test_to_mermaid_depth_truncation_excludes_deep_nodes(lineage_store_factory) -> None:
    """Nodes at distance > ``depth`` must not have their label line emitted.

    Build a chain A→B→C→D (lengths 1,2,3 from A). depth=2 → D's node declaration
    must not appear (though D's ID may appear as a target in an edge line emitted
    when C is processed at depth=2; the node label for D is what must be absent).

    Pin: ``to_mermaid`` emits the edge from C→D when processing C (d=2 = depth),
    but does NOT process D because ``d > depth`` fires when D is dequeued at d=3.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="depth-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="depth-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="depth-C", run_id=RUN_A)
    d = store.upsert_node(kind="artifact", name="depth-D", run_id=RUN_A)
    for src, dst in [(a, b), (b, c), (c, d)]:
        store.add_edge(src, dst, kind="derived_from")

    out = to_mermaid(store, root_id=a, depth=2)
    # The node label for D must not appear (D never gets processed).
    # Node label lines look like: n4[("artifact:depth-D")] — contains the name.
    assert '"artifact:depth-D"' not in out, (
        "Node D's label must not be emitted when depth=2"
    )


# --------------------------------------------------------------------------- #
# to_mermaid — all node kinds use correct shape                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind, shape_fragment",
    [
        ("artifact", "[("),
        ("checkpoint", "[/"),
        ("config", "{"),
        ("run", "(("),
        ("frozen_step", "[["),
    ],
)
def test_to_mermaid_node_kind_shape(
    kind: str, shape_fragment: str, lineage_store_factory
) -> None:
    """Each recognised node kind emits the correct Mermaid shape delimiter."""
    store = lineage_store_factory()
    n = store.upsert_node(kind=kind, name=f"shp-{kind}", run_id=RUN_A)
    out = to_mermaid(store, root_id=n, depth=0)
    assert shape_fragment in out, (
        f"kind={kind!r}: expected shape fragment {shape_fragment!r} in {out!r}"
    )


# --------------------------------------------------------------------------- #
# to_dot — missing root node (line 169)                                       #
# --------------------------------------------------------------------------- #


def test_to_dot_missing_root_node_returns_valid_empty_digraph(
    lineage_store_factory,
) -> None:
    """``to_dot`` with an unknown root_id must return a valid (but empty)
    digraph string — ``if not node: continue`` on line 169 short-circuits.
    """
    store = lineage_store_factory()
    out = to_dot(store, root_id=99999, depth=2)
    assert out.startswith("digraph lineage {")
    assert out.rstrip().endswith("}")
    # Only the header lines and the closing brace; no node lines.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert lines == ["digraph lineage {", "rankdir=LR;", "}"]


# --------------------------------------------------------------------------- #
# to_dot — depth truncation (line 165)                                        #
# --------------------------------------------------------------------------- #


def test_to_dot_depth_truncation_excludes_deep_nodes(lineage_store_factory) -> None:
    """Nodes beyond ``depth`` hops do not have their node-declaration line emitted.

    Exercises the ``d > depth`` branch (line 164) in ``to_dot``.
    Chain: A→B→C→D; depth=1 → C and D's node declarations must not appear.

    Pin: ``to_dot`` emits the B→C edge arrow when processing B (d=1 = depth), but
    does not process C because ``d > depth`` when C is dequeued at d=2. Thus C's
    label appears in an edge arrow line but NOT in a node-declaration line.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="dot-depth-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="dot-depth-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="dot-depth-C", run_id=RUN_A)
    d = store.upsert_node(kind="artifact", name="dot-depth-D", run_id=RUN_A)
    for src, dst in [(a, b), (b, c), (c, d)]:
        store.add_edge(src, dst, kind="derived_from")

    out = to_dot(store, root_id=a, depth=1)
    # Node declaration lines contain the name in their label string.
    assert '"artifact:dot-depth-C"' not in out, (
        "Node C's label must not be emitted when depth=1"
    )
    assert '"artifact:dot-depth-D"' not in out, (
        "Node D's label must not be emitted when depth=1"
    )


# --------------------------------------------------------------------------- #
# to_dot — edges_from block (lines 180-187)                                   #
# --------------------------------------------------------------------------- #


def test_to_dot_edges_from_emits_arrow_and_enqueues_dst(lineage_store_factory) -> None:
    """``to_dot`` must emit a DOT arrow ``n{src} -> n{dst}`` for each outgoing
    edge from the root, and queue the destination.

    Pins lines 184-186 (the arrow line) and 187 (nxt.append).
    """
    store = lineage_store_factory()
    src_n = store.upsert_node(kind="artifact", name="dot-src", run_id=RUN_A)
    dst_n = store.upsert_node(kind="artifact", name="dot-dst", run_id=RUN_A)
    store.add_edge(src_n, dst_n, kind="derived_from")

    out = to_dot(store, root_id=src_n, depth=3)
    assert f'n{src_n} -> n{dst_n} [label="derived_from"];' in out
    # dst node should appear too (was enqueued and processed).
    assert f"n{dst_n}" in out


def test_to_dot_edges_from_dedup_skips_duplicate_key(lineage_store_factory) -> None:
    """When the same (src,dst,kind) edge key is encountered a second time in the
    edges_from loop, it must be skipped (lines 181-182: ``if key in
    visited_edges: continue``).

    A diamond graph causes C's edge to appear once from B's edges_from and
    once from A's edges_from. The second encounter must be skipped.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="dot-tri-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="dot-tri-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="dot-tri-C", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(a, c, kind="derived_from")
    store.add_edge(b, c, kind="derived_from")

    out = to_dot(store, root_id=a, depth=5)
    assert out.count(f'n{b} -> n{c} [label="derived_from"];') == 1
    assert out.count(f'n{a} -> n{c} [label="derived_from"];') == 1


# --------------------------------------------------------------------------- #
# to_dot — edges_to block (lines 188-196)                                     #
# --------------------------------------------------------------------------- #


def test_to_dot_edges_to_inbound_edge_emitted_and_predecessor_queued(
    lineage_store_factory,
) -> None:
    """When ``to_dot`` visits the root and ``store.edges_to(root)`` returns
    inbound edges, those edges must be emitted AND the src nodes queued.

    Exercises lines 189-196 of dag.py: the edges_to loop inside to_dot.

    Setup: parent→root (produced_by). BFS from root: edges_from(root)=[],
    edges_to(root)=[parent→root] → lines 193-195 emit arrow + line 196 queues parent.
    """
    store = lineage_store_factory()
    parent = store.upsert_node(kind="run", name="dot-parent", run_id=RUN_A)
    root = store.upsert_node(kind="artifact", name="dot-root", run_id=RUN_A)
    store.add_edge(parent, root, kind="produced_by")

    out = to_dot(store, root_id=root, depth=3)
    assert f'n{parent} -> n{root} [label="produced_by"];' in out
    # parent node should also appear (enqueued from edges_to and then processed).
    assert f"n{parent} [label=" in out


def test_to_dot_edges_to_dedup_skips_already_visited_key(lineage_store_factory) -> None:
    """The edges_to dedup path in to_dot (lines 190-191) must skip an inbound
    edge whose key was already registered during edges_from processing.

    Setup: A→B→C chain, BFS from A.
    When visiting B, edges_from(B)=[B→C] → key (B,C,derived_from) added.
    Then edges_to(B)=[A→B] → key (A,B,derived_from) already added via A's
    edges_from pass → ``if key in visited_edges: continue`` fires.
    Each arrow must appear exactly once.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="dot-chain-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="dot-chain-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="dot-chain-C", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(b, c, kind="derived_from")

    out = to_dot(store, root_id=a, depth=5)
    assert out.count(f'n{a} -> n{b} [label="derived_from"];') == 1
    assert out.count(f'n{b} -> n{c} [label="derived_from"];') == 1


# --------------------------------------------------------------------------- #
# to_dot — shape mapping for all node kinds                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind, expected_shape",
    [
        ("artifact", "box"),
        ("checkpoint", "folder"),
        ("config", "note"),
        ("run", "ellipse"),
        ("frozen_step", "diamond"),
    ],
)
def test_to_dot_node_kind_shape(
    kind: str, expected_shape: str, lineage_store_factory
) -> None:
    """Each recognised node kind maps to the correct DOT shape attribute."""
    store = lineage_store_factory()
    n = store.upsert_node(kind=kind, name=f"dot-shp-{kind}", run_id=RUN_A)
    out = to_dot(store, root_id=n, depth=0)
    assert f"shape={expected_shape}" in out, (
        f"kind={kind!r}: expected shape={expected_shape!r} in DOT output"
    )


# --------------------------------------------------------------------------- #
# to_dot — version label (_label_for with/without version)                    #
# --------------------------------------------------------------------------- #


def test_to_dot_label_includes_version_when_present(lineage_store_factory) -> None:
    """``_label_for`` emits ``kind:name:version`` when version is set."""
    store = lineage_store_factory()
    n = store.upsert_node(
        kind="artifact", name="versioned", version="v42", run_id=RUN_A
    )
    out = to_dot(store, root_id=n, depth=0)
    assert '"artifact:versioned:v42"' in out


def test_to_dot_label_omits_version_when_absent(lineage_store_factory) -> None:
    """``_label_for`` emits ``kind:name`` (no trailing colon) when version is None."""
    store = lineage_store_factory()
    n = store.upsert_node(kind="artifact", name="noversion", run_id=RUN_A)
    out = to_dot(store, root_id=n, depth=0)
    assert '"artifact:noversion"' in out
    assert "artifact:noversion:" not in out


# --------------------------------------------------------------------------- #
# to_dot and to_mermaid — visited_nodes prevents revisiting (BFS invariant)  #
# --------------------------------------------------------------------------- #


def test_invariant_to_dot_visited_nodes_prevents_double_processing(
    lineage_store_factory,
) -> None:
    """A node reachable via two paths must appear exactly once in the DOT output.

    Invariant: ``visited_nodes`` in ``to_dot`` (line 164: ``if node_id in
    visited_nodes``) ensures each node is emitted at most once even in a diamond.

    Diamond: A→B, A→C, B→D, C→D. BFS from A. D is reachable via B and via C.
    D's *node label declaration* (contains its unique name) must appear exactly once.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="dia-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="dia-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="dia-C", run_id=RUN_A)
    d = store.upsert_node(kind="artifact", name="dia-D", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(a, c, kind="derived_from")
    store.add_edge(b, d, kind="derived_from")
    store.add_edge(c, d, kind="derived_from")

    out = to_dot(store, root_id=a, depth=5)
    # D's node declaration contains its unique label string; count that.
    assert out.count('"artifact:dia-D"') == 1, (
        "D's node label must appear exactly once in the DOT output"
    )


def test_invariant_to_mermaid_visited_nodes_prevents_double_processing(
    lineage_store_factory,
) -> None:
    """Same diamond invariant for ``to_mermaid``.

    D is reachable via A→B→D and A→C→D; its unique name in the label must
    appear exactly once in the Mermaid output.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="mdia-A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="mdia-B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="mdia-C", run_id=RUN_A)
    d = store.upsert_node(kind="artifact", name="mdia-D", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(a, c, kind="derived_from")
    store.add_edge(b, d, kind="derived_from")
    store.add_edge(c, d, kind="derived_from")

    out = to_mermaid(store, root_id=a, depth=5)
    # D's node declaration contains its unique label string.
    assert out.count('"artifact:mdia-D"') == 1, (
        "D's node label must appear exactly once in the Mermaid output"
    )


# --------------------------------------------------------------------------- #
# Integration: mixed edge kinds in a single graph                             #
# --------------------------------------------------------------------------- #


def test_to_mermaid_multiple_edge_kinds_all_emitted(lineage_store_factory) -> None:
    """A node with both ``derived_from`` and ``produced_by`` outgoing edges must
    have both edges emitted in Mermaid output.
    """
    store = lineage_store_factory()
    root = store.upsert_node(kind="run", name="multi-root", run_id=RUN_A)
    art = store.upsert_node(kind="artifact", name="multi-art", run_id=RUN_A)
    ckpt = store.upsert_node(kind="checkpoint", name="multi-ckpt", run_id=RUN_A)
    store.add_edge(root, art, kind="produced_by")
    store.add_edge(root, ckpt, kind="produced_by")

    out = to_mermaid(store, root_id=root, depth=2)
    assert f"n{root} -->|produced_by| n{art}" in out
    assert f"n{root} -->|produced_by| n{ckpt}" in out


def test_to_dot_multiple_edge_kinds_all_emitted(lineage_store_factory) -> None:
    """A node with multiple outgoing edges in ``to_dot`` must emit all arrows."""
    store = lineage_store_factory()
    root = store.upsert_node(kind="run", name="dot-multi-root", run_id=RUN_A)
    art = store.upsert_node(kind="artifact", name="dot-multi-art", run_id=RUN_A)
    ckpt = store.upsert_node(kind="checkpoint", name="dot-multi-ckpt", run_id=RUN_A)
    store.add_edge(root, art, kind="produced_by")
    store.add_edge(root, ckpt, kind="produced_by")

    out = to_dot(store, root_id=root, depth=2)
    assert f'n{root} -> n{art} [label="produced_by"];' in out
    assert f'n{root} -> n{ckpt} [label="produced_by"];' in out
