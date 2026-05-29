"""Adversarial tests for ``LineageStore`` + ``gc_artifacts`` (retention).

Targets that the legacy ``tests/test_lineage_store.py`` misses:
  * Edge upsert semantics (``INSERT OR REPLACE``) — repeated add yields 1 row
  * Foreign-key cascade — deleting a node drops incident edges
  * Two-phase GC: first call marks deprecated; second call (after TTL) deletes
  * ``keep_best_by_metric`` with explicit ``mode=min`` / ``max`` semantics
  * ``keep_pinned`` / ``keep_tagged`` protections
  * Transaction rollback on exception
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lighttrain.lineage import LineageStore
from lighttrain.lineage.retention import RetentionPolicy, gc_artifacts


# --------------------------------------------------------------------------- #
# Edges                                                                       #
# --------------------------------------------------------------------------- #


def test_add_edge_idempotent_insert_or_replace(lineage_store_factory) -> None:
    """Adding the same (src, dst, kind) edge twice leaves a single row.

    Invariant: ``INSERT OR REPLACE`` keyed on the PRIMARY KEY (src, dst, kind)
    means the second add overwrites the first; queries return exactly one
    edge.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    store.add_edge(a, b, kind="derived_from", payload={"v": 1})
    store.add_edge(a, b, kind="derived_from", payload={"v": 2})

    edges = list(store.edges_from(a, kind="derived_from"))
    assert len(edges) == 1
    payload = json.loads(edges[0]["payload"])
    assert payload == {"v": 2}  # overwrite semantics


def test_delete_node_cascades_edges(lineage_store_factory) -> None:
    """Deleting a node also drops its incident edges via FK CASCADE.

    Invariant: the ``ON DELETE CASCADE`` declarations on ``edges.src`` and
    ``edges.dst`` (store.py:71-72) keep orphan edges from accumulating.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    store.add_edge(a, b, kind="derived_from")
    assert len(list(store.edges_from(a))) == 1

    store.delete_node(b)
    # After cascade, no edge with dst=b should remain.
    assert list(store.edges_to(b)) == []
    assert list(store.edges_from(a)) == []


def test_parents_children_filter_by_kind(lineage_store_factory) -> None:
    """``parents`` / ``children`` filter by ``edge_kind`` precisely.

    Input: two parents reach the same child via different edge kinds.
    Filtered queries return only the matching subset.
    """
    store = lineage_store_factory()
    p1 = store.upsert_node(kind="artifact", name="p1")
    p2 = store.upsert_node(kind="run", name="p2")
    c = store.upsert_node(kind="artifact", name="c")
    store.add_edge(p1, c, kind="derived_from")
    store.add_edge(p2, c, kind="produced_by")

    derived_parents = store.parents(c, edge_kind="derived_from")
    produced_parents = store.parents(c, edge_kind="produced_by")
    assert derived_parents == [p1]
    assert produced_parents == [p2]
    # No filter → both kinds.
    all_parents = sorted(store.parents(c))
    assert all_parents == sorted([p1, p2])


def test_add_edge_rejects_unknown_kind(lineage_store_factory) -> None:
    """Unknown edge kinds raise ``ValueError`` at insertion time.

    Contract: schema validation is eager.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    with pytest.raises(ValueError, match="edge kind"):
        store.add_edge(a, b, kind="totally_bogus")


# --------------------------------------------------------------------------- #
# Tagging / pinning protections                                               #
# --------------------------------------------------------------------------- #


def _mk_versions(store: LineageStore, *, name: str, n: int) -> list[int]:
    """Insert ``n`` versions of ``name`` with monotonically increasing ``ts``.

    Newer versions have higher ``ts`` so retention's "newest-first" sort
    picks them up first.
    """
    out: list[int] = []
    base = time.time()
    for i in range(n):
        nid = store.upsert_node(
            kind="artifact",
            name=name,
            version=f"v{i}",
            ts=base + i,
            payload_path=None,
        )
        out.append(nid)
    return out


def test_pin_protects_from_gc(lineage_store_factory) -> None:
    """Pinned node survives ``gc_artifacts`` even when it would otherwise be
    pruned by ``keep_last``.

    Input: 5 versions of artifact "alpha", oldest pinned. keep_last=2 keeps
    the 2 newest; the oldest (which would normally be deprecated) is also
    kept because pinned.
    """
    store = lineage_store_factory()
    ids = _mk_versions(store, name="alpha", n=5)
    oldest = ids[0]
    store.pin(oldest)

    report = gc_artifacts(
        store,
        policy=RetentionPolicy(keep_last=2, keep_pinned=True, keep_tagged=True),
    )
    # The 3 middle ones get deprecated; oldest (pinned) stays clean.
    assert oldest not in report.deprecated
    node = store.get_node(oldest)
    assert node["deprecated"] == 0


def test_tag_protects_from_gc(lineage_store_factory) -> None:
    """Tagged node survives when ``keep_tagged=True``."""
    store = lineage_store_factory()
    ids = _mk_versions(store, name="alpha", n=5)
    tagged = ids[1]
    store.tag(tagged, "production")

    report = gc_artifacts(
        store,
        policy=RetentionPolicy(keep_last=2, keep_pinned=True, keep_tagged=True),
    )
    assert tagged not in report.deprecated
    node = store.get_node(tagged)
    assert node["deprecated"] == 0


# --------------------------------------------------------------------------- #
# Two-phase GC: deprecate → delete after TTL                                  #
# --------------------------------------------------------------------------- #


def test_gc_deprecates_then_deletes_after_ttl(lineage_store_factory, tmp_path) -> None:
    """Two-phase semantics: first call marks ``deprecated=1``; only a second
    call after ``deprecated_ts + ttl_deprecated_hours`` actually unlinks the
    payload path.

    Input: 3 versions of artifact "alpha" with real ``payload_path`` dirs.
    keep_last=1, ttl_deprecated_hours=24.

    Analytical:
      pass 1: 2 oldest deprecated, none deleted (just-deprecated, well within
              grace period).
      pass 2 with ``now += 25h``: those 2 are now past TTL → deleted; their
              payload dirs unlinked.
    """
    store = lineage_store_factory()
    base = time.time()
    ids = []
    payload_dirs: list[Path] = []
    for i in range(3):
        d = tmp_path / f"art_{i}"
        d.mkdir()
        (d / "marker").write_text("x", encoding="utf-8")
        payload_dirs.append(d)
        ids.append(
            store.upsert_node(
                kind="artifact",
                name="alpha",
                version=f"v{i}",
                ts=base + i,
                payload_path=str(d),
            )
        )

    policy = RetentionPolicy(
        keep_last=1, keep_pinned=False, keep_tagged=False, ttl_deprecated_hours=24
    )

    # Pass 1: deprecate the 2 oldest. Payloads still on disk.
    r1 = gc_artifacts(store, policy=policy, now=base + 10)
    assert sorted(r1.deprecated) == sorted(ids[:2])
    assert r1.deleted == []
    for d in payload_dirs[:2]:
        assert d.exists()

    # Pass 2 within TTL: still no delete.
    r2 = gc_artifacts(store, policy=policy, now=base + 10 + 3600)
    assert r2.deleted == []
    for d in payload_dirs[:2]:
        assert d.exists()

    # Pass 3 past TTL: physical delete.
    r3 = gc_artifacts(store, policy=policy, now=base + 10 + 25 * 3600)
    assert sorted(r3.deleted) == sorted(ids[:2])
    for d in payload_dirs[:2]:
        assert not d.exists()
    # The kept (newest) version's payload is untouched.
    assert payload_dirs[2].exists()


# --------------------------------------------------------------------------- #
# keep_best_by_metric                                                         #
# --------------------------------------------------------------------------- #


def test_gc_keep_best_by_metric_min(lineage_store_factory) -> None:
    """Top-K by metric with ``mode=min`` keeps the lowest values; union with
    ``keep_last=1`` triggers eviction for the worst.

    Input: 3 artifacts with loss values 0.1, 0.5, 0.3 (insertion order →
    v0/v1/v2 in increasing ts). policy: keep_last=1 + keep_best k=2 mode=min.

    Analytical solution (union semantics from retention.py:75-88):
        keep_last=1            → keeps newest by ts: {v2}
        keep_best k=2 mode=min → keeps lowest loss:  {v0 (0.1), v2 (0.3)}
        Union                  → {v0, v2}
        Eviction               → v1 (loss 0.5) gets deprecated

    Note: ``keep_best_by_metric`` alone does NOT trigger eviction; the
    function's eviction is gated by ``keep_last`` or ``ttl_days`` being set
    (retention.py:97-98). Without one of those gates, no node is ever
    deprecated regardless of which best-metric set is computed.

    The store wires ``evaluated_by`` edges into ``_keep_best_by_metric``;
    each artifact has one such edge from a per-version "eval" run node.
    """
    store = lineage_store_factory()
    base = time.time()
    art_ids = []
    losses = [0.1, 0.5, 0.3]
    for i, loss in enumerate(losses):
        art = store.upsert_node(
            kind="artifact", name="alpha", version=f"v{i}", ts=base + i
        )
        eval_run = store.upsert_node(
            kind="run", name=f"eval_v{i}", ts=base + i + 0.5
        )
        store.add_edge(eval_run, art, kind="evaluated_by", payload={"loss": loss})
        art_ids.append(art)

    policy = RetentionPolicy(
        keep_last=1,
        keep_pinned=False,
        keep_tagged=False,
        keep_best_by_metric={"metric": "loss", "mode": "min", "k": 2},
    )

    report = gc_artifacts(store, policy=policy, now=base + 10)
    bad_idx = losses.index(0.5)
    assert art_ids[bad_idx] in report.deprecated
    # The two lower-loss survivors are NOT deprecated.
    assert art_ids[losses.index(0.1)] not in report.deprecated
    assert art_ids[losses.index(0.3)] not in report.deprecated


def test_gc_keep_last_n_keeps_newest(lineage_store_factory) -> None:
    """``keep_last=N`` keeps the N newest by ts, deprecates the rest."""
    store = lineage_store_factory()
    ids = _mk_versions(store, name="alpha", n=5)
    policy = RetentionPolicy(keep_last=2, keep_pinned=False, keep_tagged=False)
    report = gc_artifacts(store, policy=policy)
    # The 3 oldest are deprecated; the 2 newest survive.
    assert sorted(report.deprecated) == sorted(ids[:3])


# --------------------------------------------------------------------------- #
# resolve_ref                                                                 #
# --------------------------------------------------------------------------- #


def test_resolve_ref_by_name_and_version(lineage_store_factory) -> None:
    """``"<kind>:<name>:<version>"`` returns the matching node id; ``#<id>``
    parses to the raw id.

    Input: insert artifact "alpha" version "v1"; resolve via both forms.
    """
    store = lineage_store_factory()
    nid = store.upsert_node(kind="artifact", name="alpha", version="v1")
    assert store.resolve_ref(f"artifact:alpha:v1") == nid
    assert store.resolve_ref(f"#{nid}") == nid
    assert store.resolve_ref("artifact:alpha:does-not-exist") is None


# --------------------------------------------------------------------------- #
# Transactions                                                                #
# --------------------------------------------------------------------------- #


def test_invariant_gc_dry_run_does_not_mutate_store(lineage_store_factory) -> None:
    """``gc_artifacts(dry_run=True)`` reports what WOULD change but mutates nothing.

    Adversarial PR-reviewer pass: a lazy newcomer might "optimize" the GC
    code by hoisting ``store.invalidate(nid)`` out of the ``if not dry_run``
    branch (retention.py:101-102), reasoning that the report is the same
    either way. This test pins the contract that dry_run is a true preview
    — no rows change ``deprecated`` from 0 to 1.

    Input: 5 versions of "alpha"; ``gc_artifacts(keep_last=2, dry_run=True)``.
    Analytical: ``report.deprecated`` lists the 3 oldest as candidates, but
    each node's ``deprecated`` field on disk stays 0.
    """
    store = lineage_store_factory()
    ids = _mk_versions(store, name="alpha", n=5)
    report = gc_artifacts(
        store,
        policy=RetentionPolicy(keep_last=2, keep_pinned=False, keep_tagged=False),
        dry_run=True,
    )
    # The report reflects the would-be deprecations.
    assert sorted(report.deprecated) == sorted(ids[:3])
    # But every node's deprecated field is still 0 on disk.
    for nid in ids:
        node = store.get_node(nid)
        assert node["deprecated"] == 0, (
            f"dry_run=True wrongly mutated node {nid}: "
            f"deprecated={node['deprecated']}"
        )


def test_transaction_rolls_back_on_exception(lineage_store_factory) -> None:
    """An exception inside ``store.transaction()`` undoes pending mutations.

    Invariant: ``transaction()`` is a real DB transaction, not a no-op.
    """
    store = lineage_store_factory()
    a = store.upsert_node(kind="artifact", name="A")
    b = store.upsert_node(kind="artifact", name="B")
    pre_edges = len(list(store.iter_edges()))

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            store.add_edge(a, b, kind="derived_from")
            raise RuntimeError("rollback")

    post_edges = len(list(store.iter_edges()))
    assert post_edges == pre_edges
