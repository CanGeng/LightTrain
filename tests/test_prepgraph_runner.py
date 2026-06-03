"""End-to-end PrepRunner tests — plan / cache hit / dry-run / leaf nodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data import processors as _processors  # noqa: F401 — register processors
from lighttrain.builtin_plugins.prepgraph import nodes  # noqa: F401 — registers prep_node entries
from lighttrain.prepgraph.dag import PrepGraph
from lighttrain.prepgraph.runner import PrepRunner


@pytest.fixture
def chat_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "chat.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": f"Q{i}"},
                      {"role": "assistant", "content": f"A{i}"}]}
        for i in range(8)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _spec(jsonl_path: Path) -> dict:
    return {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl_path}",
                "raw_data_version": "v0",
            },
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {
                        "name": "byte"
                    },
                },
            },
            {
                "name": "packed",
                "kind": "pack",
                "inputs": ["tok"],
                "seq_len": 64,
                "eos_id": 258,
                "pad_id": 256,
            },
            {
                "name": "out",
                "kind": "materialize",
                "inputs": ["packed"],
                "layout": "rows",
                "fmt": "jsonl",
            },
        ],
        "terminals": ["out"],
    }


def test_dag_layers(chat_jsonl: Path) -> None:
    g = PrepGraph.from_config(_spec(chat_jsonl))
    layers = g.layers
    assert layers[0] == ["raw"]
    assert layers[-1] == ["out"]
    flat = [n for layer in layers for n in layer]
    assert flat.index("raw") < flat.index("tok") < flat.index("packed") < flat.index("out")
    assert g.terminals == ["out"]


def test_plan_first_run_misses(chat_jsonl: Path, tmp_path: Path) -> None:
    g = PrepGraph.from_config(_spec(chat_jsonl))
    runner = PrepRunner(g, store_root=tmp_path / "store")
    plan = runner.plan()
    assert all(not entry.hit for entry in plan)
    assert plan[0].reason == "first_run"


def test_run_then_replay_is_cache_hit(chat_jsonl: Path, tmp_path: Path) -> None:
    g = PrepGraph.from_config(_spec(chat_jsonl))
    store = tmp_path / "store"
    runner = PrepRunner(g, store_root=store)
    results = runner.run()
    assert "out" in results
    assert results["out"].store is not None
    assert len(results["out"].store) > 0

    runner2 = PrepRunner(PrepGraph.from_config(_spec(chat_jsonl)), store_root=store)
    plan2 = runner2.plan()
    assert all(entry.hit for entry in plan2)
    assert all(entry.reason == "cache_hit" for entry in plan2)


def test_dry_run_writes_nothing(chat_jsonl: Path, tmp_path: Path) -> None:
    g = PrepGraph.from_config(_spec(chat_jsonl))
    store = tmp_path / "store"
    runner = PrepRunner(g, store_root=store)
    runner.dry_run()
    assert not store.exists() or not any(store.iterdir())


def test_config_change_invalidates_cache(chat_jsonl: Path, tmp_path: Path) -> None:
    g = PrepGraph.from_config(_spec(chat_jsonl))
    store = tmp_path / "store"
    PrepRunner(g, store_root=store).run()

    spec2 = _spec(chat_jsonl)
    # Bump seq_len → fingerprint changes for `packed` + downstream.
    for node in spec2["nodes"]:
        if node["name"] == "packed":
            node["seq_len"] = 128
    g2 = PrepGraph.from_config(spec2)
    plan = PrepRunner(g2, store_root=store).plan()
    by_name = {p.name: p for p in plan}
    assert by_name["raw"].hit
    assert by_name["tok"].hit
    assert not by_name["packed"].hit
    assert by_name["packed"].reason == "config_changed"
    assert not by_name["out"].hit
    assert by_name["out"].reason in ("upstream_changed", "first_run")


def test_cleanup_orphans_multi_node_no_keyerror(chat_jsonl: Path, tmp_path: Path) -> None:
    # Regression: cleanup_orphans on a graph whose nodes have inputs used to
    # raise KeyError because it never populated _fp_cache before _resolve.
    g = PrepGraph.from_config(_spec(chat_jsonl))
    store = tmp_path / "store"
    PrepRunner(g, store_root=store).run()
    removed = PrepRunner(
        PrepGraph.from_config(_spec(chat_jsonl)), store_root=store
    ).cleanup_orphans(dry_run=True)
    # All current fingerprints are live → nothing flagged orphan.
    assert removed == []


def test_validate_node_emits_report(chat_jsonl: Path, tmp_path: Path) -> None:
    spec = _spec(chat_jsonl)
    spec["nodes"].insert(2, {
        "name": "checked",
        "kind": "validate",
        "inputs": ["tok"],
        "vocab_size": 260,
        "min_rows": 1,
    })
    spec["nodes"][3]["inputs"] = ["checked"]  # rewire pack to read checked
    g = PrepGraph.from_config(spec)
    runner = PrepRunner(g, store_root=tmp_path / "store")
    runner.run()
    final_dir = next(
        (tmp_path / "store" / "validate" / "checked").iterdir()
    )
    report_path = final_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] >= 1
    assert "length_histogram" in report
    assert "label_keep_ratio" in report


def test_chunk_node_splits_long_rows(tmp_path: Path) -> None:
    p = tmp_path / "long.jsonl"
    long_text = "x" * 500
    p.write_text(json.dumps({"text": long_text}), encoding="utf-8")
    spec = {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{p}",
            },
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "tokenizer": {"name": "byte"},
                "text_field": "text",
            },
            {
                "name": "chunked",
                "kind": "chunk",
                "inputs": ["tok"],
                "max_len": 64,
                "overlap": 8,
            },
        ],
        "terminals": ["chunked"],
    }
    g = PrepGraph.from_config(spec)
    results = PrepRunner(g, store_root=tmp_path / "store").run()
    rows = results["chunked"].rows
    assert len(rows) > 1
    assert all(len(r["input_ids"]) <= 64 for r in rows)


def test_mix_node_round_robin(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("\n".join(json.dumps({"text": f"a{i}"}) for i in range(3)),
                  encoding="utf-8")
    p2.write_text("\n".join(json.dumps({"text": f"b{i}"}) for i in range(3)),
                  encoding="utf-8")
    spec = {
        "nodes": [
            {"name": "src_a", "kind": "load", "source": f"jsonl:{p1}"},
            {"name": "src_b", "kind": "load", "source": f"jsonl:{p2}"},
            {
                "name": "mixed",
                "kind": "mix",
                "inputs": ["src_a", "src_b"],
                "strategy": "round_robin",
            },
        ],
        "terminals": ["mixed"],
    }
    g = PrepGraph.from_config(spec)
    results = PrepRunner(g, store_root=tmp_path / "store").run()
    rows = results["mixed"].rows
    assert len(rows) == 6
    # round_robin alternates: a0 b0 a1 b1 ...
    assert rows[0]["text"].startswith("a")
    assert rows[1]["text"].startswith("b")


def test_cleanup_orphans_removes_unused(chat_jsonl: Path, tmp_path: Path) -> None:
    g1 = PrepGraph.from_config(_spec(chat_jsonl))
    store = tmp_path / "store"
    PrepRunner(g1, store_root=store).run()

    spec2 = _spec(chat_jsonl)
    for node in spec2["nodes"]:
        if node["name"] == "packed":
            node["seq_len"] = 128
    g2 = PrepGraph.from_config(spec2)
    runner2 = PrepRunner(g2, store_root=store)
    runner2.run()

    removed = runner2.cleanup_orphans()
    # The original `packed`/`out` fp dirs (seq_len=64) should be orphaned now.
    assert removed
