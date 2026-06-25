"""Edge-case unit tests for ``lighttrain.observability.lineage.store``.

Companion to ``test_store_and_retention.py`` / ``test_dag.py`` (which already
pin edges, FK cascade, GC, resolve_ref happy-path). This file drives the
remaining uncovered branches of ``store.py`` toward 100%:

  * ``upsert_node`` rejects unknown node kinds (line 121)
  * ``get_node`` tolerates a corrupt (non-JSON) ``payload`` blob (180-181)
  * ``update_node_payload`` type-guard, missing-node early return, corrupt-JSON
    swallow, and merge vs. replace semantics (200, 207, 215-216)
  * ``children`` edge-kind filtering (304-305)
  * ``ancestors_until`` bounded BFS over derived_from + produced_by,
    cycle-safety, and kind filtering (310-327)
  * ``unpin`` clears the pinned flag (355)
  * ``by_tag`` with and without a kind filter (366-367)
  * ``resolve_ref`` ``#<bad-int>`` and >3-part refs return ``None`` (380-381, 389)
  * ``transaction`` COMMIT on the success path (408)
  * ``close`` swallows + logs a double-close exception (413-414)

Style mirrors tests/eval/test_suite.py and tests/trainers/test_base_seams.py.
"""
from __future__ import annotations

import json
import logging

import pytest

from lighttrain.observability.lineage import LineageStore

# --------------------------------------------------------------------------- #
# Helpers / stubs                                                             #
# --------------------------------------------------------------------------- #


class _BadMapping:
    """A non-Mapping object that nonetheless has dict-ish attributes.

    Used to prove ``update_node_payload``'s ``isinstance(payload, Mapping)``
    guard rejects anything that is not an actual mapping, not merely anything
    lacking ``keys``.
    """

    def keys(self):  # pragma: no cover - never reached; guard fires first
        return []


def _write_raw_payload(store: LineageStore, node_id: int, raw: str) -> None:
    """Stuff a raw (possibly non-JSON) string straight into ``nodes.payload``.

    The public ``upsert_node`` / ``update_node_payload`` paths always
    ``json.dumps`` so they can never write malformed JSON; we go around them to
    exercise the defensive ``except json.JSONDecodeError`` branches.
    """
    store.conn.execute(
        "UPDATE nodes SET payload = ? WHERE id = ?", (raw, int(node_id))
    )


# --------------------------------------------------------------------------- #
# upsert_node: kind validation                                               #
# --------------------------------------------------------------------------- #


def test_upsert_node_rejects_unknown_kind(lineage_store_factory) -> None:
    """Unknown node kinds raise ``ValueError`` naming the offending kind.

    Contract (line 121): validation is eager and the message echoes both the
    bad kind and the allowed set so callers can self-correct.
    """
    store = lineage_store_factory()
    with pytest.raises(ValueError, match="unknown lineage node kind 'bogus'"):
        store.upsert_node(kind="bogus", name="x")


@pytest.mark.parametrize(
    "kind", ["artifact", "checkpoint", "config", "run", "frozen_step"]
)
def test_invariant_all_documented_node_kinds_accepted(
    kind: str, lineage_store_factory
) -> None:
    """Every kind listed in the module docstring / ``_NODE_KINDS`` is accepted."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind=kind, name=f"n_{kind}")
    assert store.get_node(nid)["kind"] == kind


# --------------------------------------------------------------------------- #
# get_node: corrupt payload tolerance                                        #
# --------------------------------------------------------------------------- #


def test_get_node_returns_none_for_missing_id(lineage_store_factory) -> None:
    """``get_node`` returns ``None`` for an id that was never inserted."""
    store = lineage_store_factory()
    assert store.get_node(99999) is None


def test_get_node_roundtrips_payload_and_tags(lineage_store_factory) -> None:
    """A well-formed JSON payload deserializes back to a dict; tags default ``[]``."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind="artifact", name="A", payload={"loss": 0.25})
    node = store.get_node(nid)
    assert node["payload"] == {"loss": 0.25}
    assert node["tags"] == []


def test_get_node_tolerates_corrupt_payload_json(lineage_store_factory) -> None:
    """A non-JSON ``payload`` blob is left as the raw string, not raised on.

    Pins the defensive ``except json.JSONDecodeError: pass`` (lines 180-181):
    a hand-edited / partially-written row must not break ``get_node`` reads.
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="artifact", name="A")
    _write_raw_payload(store, nid, "{not valid json")
    node = store.get_node(nid)
    # Truthy-but-unparseable payload survives as the original string.
    assert node["payload"] == "{not valid json"


# --------------------------------------------------------------------------- #
# update_node_payload: guard, merge, replace, missing-node, corrupt-json      #
# --------------------------------------------------------------------------- #


def test_update_node_payload_rejects_non_mapping(lineage_store_factory) -> None:
    """A non-Mapping ``payload`` raises ``TypeError`` (line 200)."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r")
    with pytest.raises(TypeError, match="payload must be a mapping"):
        store.update_node_payload(nid, _BadMapping())


def test_update_node_payload_merge_appends_without_clobber(
    lineage_store_factory,
) -> None:
    """``merge=True`` shallow-merges new fields onto the existing payload.

    Mirrors the documented ``on_train_end`` use-case: adding ``ended_ts``
    must not drop a previously-written ``started_ts``.
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r", payload={"started_ts": 1.0})
    store.update_node_payload(nid, {"ended_ts": 2.0}, merge=True)
    assert store.get_node(nid)["payload"] == {"started_ts": 1.0, "ended_ts": 2.0}


def test_update_node_payload_merge_overwrites_overlapping_keys(
    lineage_store_factory,
) -> None:
    """Overlapping keys: the new value wins (``merged.update(new)``)."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r", payload={"k": "old", "keep": 1})
    store.update_node_payload(nid, {"k": "new"}, merge=True)
    assert store.get_node(nid)["payload"] == {"k": "new", "keep": 1}


def test_update_node_payload_replace_discards_old(lineage_store_factory) -> None:
    """``merge=False`` replaces the payload wholesale (no read of the old)."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r", payload={"started_ts": 1.0})
    store.update_node_payload(nid, {"only": True}, merge=False)
    assert store.get_node(nid)["payload"] == {"only": True}


def test_update_node_payload_merge_on_empty_payload(lineage_store_factory) -> None:
    """Merging onto a node with no prior payload just writes the new dict.

    Exercises the ``if cur:`` false branch (cur is ``None``) so the merge falls
    straight through to the write without touching the corrupt-JSON path.
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r")  # payload is NULL
    store.update_node_payload(nid, {"fresh": 1}, merge=True)
    assert store.get_node(nid)["payload"] == {"fresh": 1}


def test_update_node_payload_missing_node_is_noop(lineage_store_factory) -> None:
    """``merge=True`` on a non-existent id returns early without inserting.

    Pins the ``if row is None: return`` guard (lines 206-207): the store must
    not resurrect a deleted node via a stray payload update.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    before = sorted(n["id"] for n in store.iter_nodes())
    # No exception, and no new row.
    store.update_node_payload(a + 12345, {"x": 1}, merge=True)
    after = sorted(n["id"] for n in store.iter_nodes())
    assert before == after


def test_update_node_payload_merge_swallows_corrupt_existing_json(
    lineage_store_factory,
) -> None:
    """When the *existing* payload is unparseable, merge silently overwrites it.

    Pins lines 215-216: ``json.loads(cur)`` raises ``JSONDecodeError`` →
    swallowed → ``new`` (the incoming dict) is written as-is, so the corrupt
    blob is replaced rather than merged.
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r")
    _write_raw_payload(store, nid, "<<<corrupt>>>")
    store.update_node_payload(nid, {"recovered": True}, merge=True)
    assert store.get_node(nid)["payload"] == {"recovered": True}


def test_update_node_payload_merge_replaces_non_dict_json(
    lineage_store_factory,
) -> None:
    """If the existing payload is valid JSON but NOT a dict (e.g. a list), the
    ``isinstance(merged, dict)`` guard skips the merge and the new dict wins.

    Covers the false branch of ``if isinstance(merged, dict)`` (line 212).
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="run", name="r")
    _write_raw_payload(store, nid, json.dumps([1, 2, 3]))  # valid JSON, a list
    store.update_node_payload(nid, {"now": "dict"}, merge=True)
    assert store.get_node(nid)["payload"] == {"now": "dict"}


# --------------------------------------------------------------------------- #
# children: edge-kind filtering                                              #
# --------------------------------------------------------------------------- #


def test_children_filter_by_edge_kind(lineage_store_factory) -> None:
    """``children`` returns dsts of outgoing edges, filtered by edge kind.

    Covers lines 304-305 (the method body). One source fans out to two
    children via different edge kinds; the filter isolates each.
    """
    store = lineage_store_factory()
    src = store.upsert_node(kind="run", name="src")
    c_prod = store.upsert_node(kind="artifact", name="produced")
    c_eval = store.upsert_node(kind="artifact", name="evaluated")
    store.add_edge(src, c_prod, kind="produced_by")
    store.add_edge(src, c_eval, kind="evaluated_by")

    assert store.children(src, edge_kind="produced_by") == [c_prod]
    assert store.children(src, edge_kind="evaluated_by") == [c_eval]
    assert sorted(store.children(src)) == sorted([c_prod, c_eval])


def test_children_empty_for_leaf(lineage_store_factory) -> None:
    """A node with no outgoing edges has no children."""
    store = lineage_store_factory()
    leaf = store.upsert_node(kind="artifact", name="leaf")
    assert store.children(leaf) == []


# --------------------------------------------------------------------------- #
# ancestors_until: bounded BFS over derived_from + produced_by               #
# --------------------------------------------------------------------------- #


def test_ancestors_until_collects_matching_kind(lineage_store_factory) -> None:
    """Walk ``derived_from`` + ``produced_by`` backwards, keep only ``kind`` hits.

    Graph (edge src=parent → dst=child):
        run R  --produced_by-->  artifact A
        artifact A  --derived_from-->  artifact B  (start)
    ancestors_until(B, kind="run") must surface R (reached B→A via derived_from,
    A→R via produced_by). Covers lines 310-327.
    """
    store = lineage_store_factory()
    r = store.upsert_node(kind="run", name="R")
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    store.add_edge(r, a, kind="produced_by")
    store.add_edge(a, b, kind="derived_from")

    assert store.ancestors_until(b, kind="run") == [r]
    # Filtering by artifact surfaces A (B's derived_from parent) but not the
    # start node B itself only if B matches the kind — B IS an artifact, so it
    # is included as well (BFS visits the start node first).
    assert sorted(store.ancestors_until(b, kind="artifact")) == sorted([a, b])


def test_ancestors_until_includes_start_node_when_kind_matches(
    lineage_store_factory,
) -> None:
    """The start node itself is collected when its kind matches ``kind``.

    Pins current behaviour: ``ancestors_until`` does not exclude the start
    node (unlike ``cycle_check``). A lone artifact returns itself.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="solo")
    assert store.ancestors_until(a, kind="artifact") == [a]
    # A kind that does not match yields nothing.
    assert store.ancestors_until(a, kind="run") == []


def test_ancestors_until_is_cycle_safe(lineage_store_factory) -> None:
    """A cyclic derived_from graph terminates (visited set prevents re-walk).

    Covers the ``if n in visited: continue`` short-circuit (lines 316-317).
    Build A → B → A (a 2-cycle). The BFS must terminate and report both once.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    store.add_edge(a, b, kind="derived_from")
    store.add_edge(b, a, kind="derived_from")  # back-edge → cycle

    out = store.ancestors_until(b, kind="artifact")
    assert sorted(out) == sorted([a, b])
    # Each node appears at most once.
    assert len(out) == len(set(out))


def test_ancestors_until_skips_missing_parent_node(lineage_store_factory) -> None:
    """A dangling parent edge (node row deleted) does not crash the walk.

    Covers the ``if node and ...`` guard (line 320) where ``get_node`` is None.
    Delete B's row with FK off, leaving an orphan A→B derived_from edge from
    start C's perspective; the BFS must skip B gracefully.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="run", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    c = store.upsert_node(kind="artifact", name="C")
    store.add_edge(a, b, kind="produced_by")
    store.add_edge(b, c, kind="derived_from")

    store.conn.executescript("PRAGMA foreign_keys=OFF;")
    store.conn.execute("DELETE FROM nodes WHERE id = ?", (b,))
    store.conn.executescript("PRAGMA foreign_keys=ON;")

    # C → B(missing) → A: A is still reachable since the parent loop queues
    # B's parents regardless of B's node row existing.
    out = store.ancestors_until(c, kind="run")
    assert out == [a]


# --------------------------------------------------------------------------- #
# pin / unpin                                                                 #
# --------------------------------------------------------------------------- #


def test_unpin_clears_pinned_flag(lineage_store_factory) -> None:
    """``unpin`` flips ``pinned`` back to 0 after a ``pin`` (line 355)."""
    store = lineage_store_factory()
    nid = store.upsert_node(kind="artifact", name="A")
    store.pin(nid)
    assert store.get_node(nid)["pinned"] == 1
    store.unpin(nid)
    assert store.get_node(nid)["pinned"] == 0


# --------------------------------------------------------------------------- #
# by_tag                                                                      #
# --------------------------------------------------------------------------- #


def test_by_tag_without_kind_filter(lineage_store_factory) -> None:
    """``by_tag`` returns every node carrying the tag across all kinds.

    Covers line 367 (the ``iter_nodes()`` no-kind branch + list comp).
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    r = store.upsert_node(kind="run", name="R")
    untagged = store.upsert_node(kind="artifact", name="U")
    store.tag(a, "best")
    store.tag(r, "best")

    result = sorted(store.by_tag("best"))
    assert result == sorted([a, r])
    assert untagged not in result


def test_by_tag_with_kind_filter(lineage_store_factory) -> None:
    """``by_tag(kind=...)`` restricts to the given node kind (line 366 branch)."""
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    r = store.upsert_node(kind="run", name="R")
    store.tag(a, "best")
    store.tag(r, "best")

    assert store.by_tag("best", kind="artifact") == [a]
    assert store.by_tag("best", kind="run") == [r]


def test_by_tag_no_matches_returns_empty(lineage_store_factory) -> None:
    """A tag nobody carries yields an empty list."""
    store = lineage_store_factory()
    store.upsert_node(kind="artifact", name="A")
    assert store.by_tag("nonexistent") == []


# --------------------------------------------------------------------------- #
# resolve_ref: malformed refs                                                #
# --------------------------------------------------------------------------- #


def test_resolve_ref_bad_id_returns_none(lineage_store_factory) -> None:
    """``#<non-int>`` returns ``None`` rather than raising (lines 380-381)."""
    store = lineage_store_factory()
    assert store.resolve_ref("#notanumber") is None


def test_resolve_ref_too_many_parts_returns_none(lineage_store_factory) -> None:
    """A ref with >3 colon-separated parts is rejected (line 389)."""
    store = lineage_store_factory()
    assert store.resolve_ref("artifact:name:version:extra") is None


def test_resolve_ref_single_token_returns_none(lineage_store_factory) -> None:
    """A bare token (1 part, no colon) also falls through to ``None``.

    ``"foo".split(":")`` has len 1 → neither the ==2 nor ==3 branch → else.
    """
    store = lineage_store_factory()
    assert store.resolve_ref("justakind") is None


def test_resolve_ref_hash_id_roundtrip(lineage_store_factory) -> None:
    """A valid ``#<id>`` resolves to the integer id when the node exists, and
    returns ``None`` for a well-formed but non-existent id — existence is now
    verified (so ``invalidate #<phantom>`` exits 1 instead of silently passing).
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="artifact", name="A")
    assert store.resolve_ref(f"#{nid}") == nid
    # Well-formed but non-existent id → None (existence is checked).
    assert store.resolve_ref("#424242") is None


# --------------------------------------------------------------------------- #
# transaction: commit on success                                             #
# --------------------------------------------------------------------------- #


def test_transaction_commits_on_success(lineage_store_factory) -> None:
    """The success path of ``transaction`` issues COMMIT and persists writes.

    Covers line 408 (the ``else: COMMIT`` arm). After a clean ``with`` block,
    the edge added inside must be visible.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    with store.transaction():
        store.add_edge(a, b, kind="derived_from")
    assert len(store.edges_from(a, kind="derived_from")) == 1


# --------------------------------------------------------------------------- #
# close: idempotency + error swallowing                                      #
# --------------------------------------------------------------------------- #


def test_close_is_idempotent(lineage_store_factory) -> None:
    """Calling ``close`` twice does not raise (sqlite tolerates double-close)."""
    store = lineage_store_factory()
    store.close()
    store.close()  # second close is a harmless no-op


def test_close_swallows_and_logs_connection_error(
    lineage_store_factory, caplog
) -> None:
    """A failure inside ``conn.close()`` is swallowed and logged at WARNING.

    Pins lines 413-417: ``close`` must never propagate — a broken connection
    object should produce a warning, not crash teardown. We replace ``conn``
    with a stub whose ``close`` raises.
    """

    class _ExplodingConn:
        def close(self) -> None:
            raise RuntimeError("boom")

    store = lineage_store_factory()
    real = store.conn  # keep a handle so the factory teardown can close it too
    store.conn = _ExplodingConn()

    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.lineage.store"):
        store.close()  # must NOT raise

    assert any(
        "failed to close the SQLite connection" in r.getMessage()
        for r in caplog.records
    ), caplog.text

    # Restore the real connection so the fixture's teardown close() is clean.
    store.conn = real


# --------------------------------------------------------------------------- #
# context manager                                                            #
# --------------------------------------------------------------------------- #


def test_context_manager_closes_on_exit(tmp_path) -> None:
    """``with LineageStore(...) as s`` closes the connection on exit.

    Covers ``__enter__`` / ``__exit__``. After the block, a query must fail
    because the connection is closed.
    """
    with LineageStore(tmp_path / "ctx.sqlite") as s:
        nid = s.upsert_node(kind="artifact", name="A")
        assert s.get_node(nid)["name"] == "A"  # type: ignore[index]
    # Connection closed → using it raises ProgrammingError.
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        s.conn.execute("SELECT 1")
