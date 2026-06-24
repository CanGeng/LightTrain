"""Adversarial tests for ``lighttrain.observability.lineage.dag``.

Targets that the legacy ``tests/test_lineage_cycle.py`` misses:
  * Exact depth of detected cycles (not just "non-empty hits")
  * K-hop boundary semantics (``depth > k`` truncates correctly)
  * The exclusion-of-start-node invariant
  * Resilience to a missing parent row during traversal
  * The three-level policy (``allowed`` / ``warn`` / ``forbid``) + the
    ``require_external_signal`` upgrade pin
  * Mermaid / DOT graph export format pins
"""
from __future__ import annotations

import warnings

import pytest

from lighttrain.observability.lineage import LineageStore
from lighttrain.observability.lineage.dag import (
    CycleHit,
    apply_cycle_policy,
    cycle_check,
    to_dot,
    to_mermaid,
)

RUN_A = "run-A"
RUN_B = "run-B"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _add_chain(store: LineageStore, run_id: str, depth: int) -> list[int]:
    """Insert ``depth+1`` artifacts forming a chain ``n_0 ← n_1 ← ... ← n_depth``
    via ``derived_from`` edges (parent → child semantics: edge src is the
    parent the child was derived from).

    Returns the node ids in order ``[n_0, n_1, ..., n_depth]``. ``n_depth``
    is the start node for cycle_check; its ancestors are at distance 1..depth.
    """
    ids: list[int] = []
    for i in range(depth + 1):
        nid = store.upsert_node(
            kind="artifact",
            name=f"a{i}",
            version=run_id,
            run_id=run_id,
        )
        ids.append(nid)
    # Edge: src=parent, dst=child, kind=derived_from
    for parent, child in zip(ids[:-1], ids[1:], strict=False):
        store.add_edge(parent, child, kind="derived_from")
    return ids


# --------------------------------------------------------------------------- #
# cycle_check: depth + boundary + start exclusion                             #
# --------------------------------------------------------------------------- #


def test_cycle_check_at_depth_one(lineage_store_factory) -> None:
    """A direct parent in the same run is detected as a hit at depth=1.

    Input: chain ``parent → child`` with the same ``run_id`` as ``current_run_id``.
    Analytical: cycle_check walks back via ``parents`` of child → arrives at
    parent at depth=1; ``parent.run_id == current_run_id`` so it's a hit.
    The depth field of the hit must be 1.
    """
    store = lineage_store_factory()
    ids = _add_chain(store, RUN_A, depth=1)
    hits = cycle_check(store, start_node=ids[-1], current_run_id=RUN_A, k=4)
    assert len(hits) == 1
    assert hits[0].depth == 1
    assert hits[0].node_id == ids[0]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_invariant_cycle_check_depth_bounded_by_k(
    k: int, lineage_store_factory
) -> None:
    """Ancestor at depth==k is detected; ancestor at depth==k+1 is NOT.

    Invariant: ``cycle_check`` is a bounded BFS. The condition ``depth > k``
    (line 51) means nodes at depth==k are processed (and can become hits),
    but their ancestors (depth==k+1) are skipped.

    Input: chain of length k+1 (depths 1..k+1 from the start) all in the
    same run. Hits collected: depths in ``range(1, k+1)``.
    """
    store = lineage_store_factory()
    chain_depth = k + 1
    ids = _add_chain(store, RUN_A, depth=chain_depth)
    hits = cycle_check(store, start_node=ids[-1], current_run_id=RUN_A, k=k)
    seen_depths = sorted(h.depth for h in hits)
    assert seen_depths == list(range(1, k + 1)), (
        f"k={k}: expected depths 1..{k}, got {seen_depths}"
    )


def test_invariant_cycle_check_excludes_start_node(lineage_store_factory) -> None:
    """The start node is never reported as its own hit, even though its
    ``run_id`` trivially matches ``current_run_id``.

    Invariant: cycle_check filters ``node_id != start_node`` (line 55).
    Removing that check would produce a spurious self-hit at depth 0.

    Input: a single isolated node ``n`` in run RUN_A. cycle_check(n, RUN_A) →
    empty (no parents AND start node excluded).
    """
    store = lineage_store_factory()
    n = store.upsert_node(kind="artifact", name="loner", run_id=RUN_A)
    hits = cycle_check(store, start_node=n, current_run_id=RUN_A, k=4)
    assert hits == []


def test_cycle_check_different_run_no_hit(lineage_store_factory) -> None:
    """An ancestor whose ``run_id`` differs is not a self-feeding hit.

    Input: chain parent (RUN_B) → child (RUN_A). cycle_check from child with
    current_run_id=RUN_A: ancestor's run_id is RUN_B, no hit.
    """
    store = lineage_store_factory()
    parent = store.upsert_node(kind="artifact", name="p", version="vA",
                               run_id=RUN_B)
    child = store.upsert_node(kind="artifact", name="c", version="vA",
                              run_id=RUN_A)
    store.add_edge(parent, child, kind="derived_from")
    hits = cycle_check(store, start_node=child, current_run_id=RUN_A, k=4)
    assert hits == []


def test_invariant_cycle_check_broken_parent_no_crash(lineage_store_factory) -> None:
    """A traversal that hits a missing-parent edge does not crash.

    Invariant: ``store.get_node(node_id)`` returning ``None`` is handled
    gracefully — ``if node and ...`` falls through, frontier continues.

    Input: chain A → B → C with all in RUN_A. We DELETE B by raw SQL,
    leaving an orphan edge B → C still pointing at the now-missing B.
    cycle_check from C must not raise.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="C", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(b, c, kind="derived_from")

    # Temporarily disable FK enforcement so we can remove B without cascade
    # deleting the edges (we want the orphan-edge state to test traversal).
    store.conn.executescript("PRAGMA foreign_keys=OFF;")
    store.conn.execute("DELETE FROM nodes WHERE id = ?", (b,))
    store.conn.executescript("PRAGMA foreign_keys=ON;")

    # Must not raise — traversal must gracefully skip the missing B.
    hits = cycle_check(store, start_node=c, current_run_id=RUN_A, k=4)
    # A is still reachable (B → C edge exists, A → B edge exists, but B is
    # gone). With B's node row absent, ``store.parents(c, ...)`` returns [b]
    # but ``get_node(b)`` is None — frontier extends to A only if the code
    # still queues b's parents. Looking at dag.py:57-59: the parent loop runs
    # regardless of node validity, so A IS reachable from C through orphan-B.
    a_hit = [h for h in hits if h.node_id == a]
    assert len(a_hit) == 1
    assert a_hit[0].depth == 2  # C → B(missing) → A


def test_cycle_check_traverses_both_edge_kinds(lineage_store_factory) -> None:
    """``derived_from`` and ``produced_by`` ancestors are both reachable.

    Input: a node with one ancestor via each edge kind, both in RUN_A.
    Analytical: cycle_check walks both kinds; both ancestors → 2 hits.
    """
    store = lineage_store_factory()
    p1 = store.upsert_node(kind="artifact", name="p1", run_id=RUN_A)
    p2 = store.upsert_node(kind="run", name="p2", run_id=RUN_A)
    start = store.upsert_node(kind="artifact", name="start", run_id=RUN_A)
    store.add_edge(p1, start, kind="derived_from")
    store.add_edge(p2, start, kind="produced_by")
    hits = cycle_check(store, start_node=start, current_run_id=RUN_A, k=4)
    hit_ids = sorted(h.node_id for h in hits)
    assert hit_ids == sorted([p1, p2])


def test_cycle_check_visited_dedup(lineage_store_factory) -> None:
    """Diamond ancestor (two paths converging on the same node) is visited once.

    Input: A → B → start AND A → C → start, all in RUN_A. A is reachable
    via two distinct parent paths but the visited set short-circuits.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="C", run_id=RUN_A)
    start = store.upsert_node(kind="artifact", name="start", run_id=RUN_A)
    for parent, child in ((a, b), (a, c), (b, start), (c, start)):
        store.add_edge(parent, child, kind="derived_from")

    hits = cycle_check(store, start_node=start, current_run_id=RUN_A, k=4)
    a_hits = [h for h in hits if h.node_id == a]
    assert len(a_hits) == 1


# --------------------------------------------------------------------------- #
# apply_cycle_policy: three levels + external-signal upgrade                  #
# --------------------------------------------------------------------------- #


def _mk_hit() -> CycleHit:
    return CycleHit(node_id=1, via_run_id=RUN_A, depth=1)


def test_apply_cycle_policy_allowed_silent() -> None:
    """``allowed`` policy: no warning, no raise.

    Invariant: silent is silent. No ``warnings.warn`` triggered.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([_mk_hit()], self_feeding="allowed")
    assert caught == []


def test_apply_cycle_policy_warn_emits_warning_with_count() -> None:
    """``warn`` policy emits a UserWarning containing the exact hit count.

    Pin: the message format ``"({n} hit(s))"`` lets downstream parsers
    extract the count without regexing the full string.
    """
    hits = [_mk_hit(), _mk_hit()]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy(hits, self_feeding="warn")
    msgs = [str(w.message) for w in caught]
    assert any("(2 hit(s))" in m for m in msgs), msgs


def test_apply_cycle_policy_forbid_raises_runtime_error() -> None:
    """``forbid`` policy raises ``RuntimeError``."""
    with pytest.raises(RuntimeError, match="self-feeding"):
        apply_cycle_policy([_mk_hit()], self_feeding="forbid")


def test_apply_cycle_policy_empty_hits_noop() -> None:
    """Empty hits = no-op regardless of policy.

    Invariant: never warn/raise when there's nothing to report.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy([], self_feeding="warn")
        apply_cycle_policy([], self_feeding="forbid")
        apply_cycle_policy([], self_feeding="allowed")
    assert caught == []


def test_invariant_apply_cycle_policy_warn_emits_warning_even_with_logger() -> None:
    """Under ``self_feeding='warn'``, ``apply_cycle_policy`` ALWAYS emits a
    Python ``warnings.warn``, even when a logger is also passed.

    Adversarial PR-reviewer pass: a lazy newcomer might "optimize" the warn
    branch to route through ``logger.log_text`` when a logger is provided,
    suppressing the Python warning. That would silently disable cycle
    detection signals for any caller that passes a logger. This test pins
    the current contract that a Python warning is always raised under
    ``warn`` policy.

    Input: a fake logger object (with ``log_text`` method) is passed in.
    The Python warning machinery must still emit a UserWarning.
    """
    captured_log_calls: list[tuple[str, int]] = []

    class _FakeLogger:
        def log_text(self, msg: str, step: int) -> None:
            captured_log_calls.append((msg, step))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy(
            [_mk_hit()],
            self_feeding="warn",
            logger=_FakeLogger(),
        )
    msgs = [str(w.message) for w in caught]
    assert any("self-feeding" in m for m in msgs), (
        f"warn policy must always emit a Python warning, even with a logger; "
        f"got warnings={msgs}"
    )


def test_pin_lineage_external_signal_upgrades_allowed_to_warn() -> None:
    """``allowed`` + ``require_external_signal=True`` + signal absent → upgraded
    to ``warn``. With signal present → stays silent.

    If this behavior is intentionally changed, update this test AND bump
    SCHEMA_VERSION (or document the breaking change). Today the upgrade
    rule lives at dag.py:79-83 and is part of the user-visible policy
    semantics.
    """
    # Signal absent: expect a warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy(
            [_mk_hit()],
            self_feeding="allowed",
            require_external_signal=True,
            external_signal_present=False,
        )
    assert caught, "upgrade allowed→warn did not fire when signal absent"
    assert any("no external judge/reward signal" in str(w.message) for w in caught)

    # Signal present: silent.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        apply_cycle_policy(
            [_mk_hit()],
            self_feeding="allowed",
            require_external_signal=True,
            external_signal_present=True,
        )
    assert caught2 == []


# --------------------------------------------------------------------------- #
# Graph export: Mermaid + DOT format pins                                     #
# --------------------------------------------------------------------------- #


def test_to_mermaid_emits_valid_header_and_label(lineage_store_factory) -> None:
    """Output begins with ``"graph TD"`` and contains a node label of the form
    ``"kind:name"``.

    Pin: the Mermaid format is a wire contract (downstream tools render it);
    the first line and label shape must stay stable.
    """
    store = lineage_store_factory()
    n = store.upsert_node(kind="artifact", name="alpha", run_id=RUN_A)
    out = to_mermaid(store, root_id=n, depth=2)
    lines = out.splitlines()
    assert lines[0] == "graph TD"
    assert any('"artifact:alpha"' in ln for ln in lines)


def test_to_dot_emits_valid_digraph(lineage_store_factory) -> None:
    """Output begins with ``"digraph lineage {"`` and ends with ``"}"``."""
    store = lineage_store_factory()
    n = store.upsert_node(kind="artifact", name="alpha", run_id=RUN_A)
    out = to_dot(store, root_id=n, depth=2)
    assert out.startswith("digraph lineage {")
    assert out.rstrip().endswith("}")


def test_to_mermaid_dedups_edges(lineage_store_factory) -> None:
    """A diamond DAG emits each edge label exactly once in the Mermaid output.

    Invariant: the ``visited_edges`` set in ``to_mermaid`` (line 112) prevents
    duplicate edges in the rendered graph.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A", run_id=RUN_A)
    b = store.upsert_node(kind="artifact", name="B", run_id=RUN_A)
    c = store.upsert_node(kind="artifact", name="C", run_id=RUN_A)
    d = store.upsert_node(kind="artifact", name="D", run_id=RUN_A)
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(a, c, kind="derived_from")
    store.add_edge(b, d, kind="derived_from")
    store.add_edge(c, d, kind="derived_from")

    out = to_mermaid(store, root_id=a, depth=5)
    # Each unique (src, dst, kind) edge should appear once.
    assert out.count(f"n{a} -->|derived_from| n{b}") == 1
    assert out.count(f"n{a} -->|derived_from| n{c}") == 1
    assert out.count(f"n{b} -->|derived_from| n{d}") == 1
    assert out.count(f"n{c} -->|derived_from| n{d}") == 1
