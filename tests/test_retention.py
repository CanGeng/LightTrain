"""RetentionPolicy + GC — DESIGN §12.5."""

from __future__ import annotations

import time

from lighttrain.observability.lineage import (
    LineageStore,
    RetentionPolicy,
    gc_artifacts,
    prune_orphans,
)


def _seed_versions(store: LineageStore, name: str, n: int, *, base_ts: float):
    ids = []
    for i in range(n):
        nid = store.upsert_node(
            kind="artifact", name=name, version=f"v{i}",
            ts=base_ts + i,
        )
        ids.append(nid)
    return ids


def test_keep_last_marks_older_deprecated(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    _seed_versions(store, "A", 5, base_ts=time.time() - 100)
    report = gc_artifacts(store, policy=RetentionPolicy(keep_last=2),
                         delete_paths=False)
    # 5 - 2 = 3 deprecated on first pass
    assert len(report.deprecated) == 3
    assert len(report.deleted) == 0


def test_pin_survives_gc(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    ids = _seed_versions(store, "A", 4, base_ts=time.time() - 100)
    store.pin(ids[0])
    gc_artifacts(store, policy=RetentionPolicy(keep_last=2),
                         delete_paths=False)
    # Pinned one is preserved even though it's the oldest.
    survived = [n["id"] for n in store.iter_nodes() if not n.get("deprecated")]
    assert ids[0] in survived


def test_tag_survives_gc(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    ids = _seed_versions(store, "A", 4, base_ts=time.time() - 100)
    store.tag(ids[0], "best_eval")
    gc_artifacts(store, policy=RetentionPolicy(keep_last=1, keep_tagged=True),
                 delete_paths=False)
    survived = [n["id"] for n in store.iter_nodes() if not n.get("deprecated")]
    assert ids[0] in survived  # via tag


def test_prune_orphans_drops_missing_path_nodes(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    real = tmp_path / "exists"
    real.mkdir()
    a = store.upsert_node(kind="artifact", name="A", version="v1",
                         payload_path=str(real))
    b = store.upsert_node(kind="artifact", name="A", version="v2",
                         payload_path=str(tmp_path / "ghost"))
    removed = prune_orphans(store)
    assert b in removed
    assert a not in removed
