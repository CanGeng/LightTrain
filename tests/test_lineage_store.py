"""LineageStore basic operations — DESIGN §12.3 / §12.4."""

from __future__ import annotations

from lighttrain.lineage import LineageStore


def test_upsert_node_returns_stable_id(tmp_path):
    store = LineageStore(tmp_path / "lineage.sqlite")
    a = store.upsert_node(
        kind="artifact", name="teacher_logits", version="v1",
        schema_kind="artifact_header", schema_version="0.4",
        payload_path=str(tmp_path / "art"),
    )
    again = store.upsert_node(
        kind="artifact", name="teacher_logits", version="v1",
        hash_="x" * 64,
    )
    assert a == again
    node = store.get_node(a)
    assert node and node["hash"] == "x" * 64


def test_add_edge_and_query_parents_children(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    run = store.upsert_node(kind="run", name="R", version="1", run_id="R")
    art = store.upsert_node(kind="artifact", name="A", version="v1")
    store.add_edge(run, art, "produced_by")
    assert store.children(run, edge_kind="produced_by") == [art]
    assert store.parents(art, edge_kind="produced_by") == [run]


def test_tag_pin_invalidate(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    nid = store.upsert_node(kind="artifact", name="A", version="v1")
    store.tag(nid, "best_eval")
    store.tag(nid, "best_eval")  # idempotent
    assert "best_eval" in store._tags_of(nid)
    store.untag(nid, "best_eval")
    assert store._tags_of(nid) == []
    store.pin(nid)
    n = store.get_node(nid)
    assert n and n["pinned"] == 1
    store.invalidate(nid)
    n2 = store.get_node(nid)
    assert n2 and n2["deprecated"] == 1


def test_resolve_ref_supports_kind_name_version_forms(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    v1 = store.upsert_node(kind="artifact", name="A", version="v1")
    v2 = store.upsert_node(kind="artifact", name="A", version="v2")
    assert store.resolve_ref("artifact:A:v1") == v1
    assert store.resolve_ref("artifact:A:v2") == v2
    # latest = highest ts; v2 was inserted last
    assert store.resolve_ref("artifact:A") == v2
    assert store.resolve_ref("artifact:A:latest") == v2
    assert store.resolve_ref(f"#{v1}") == v1
    assert store.resolve_ref("artifact:not_there:v0") is None
