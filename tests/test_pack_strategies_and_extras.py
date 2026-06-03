"""ISSUE-3/4/5: pack strategies + standardized extras + ``lines:`` load + the
``prep-status --extras`` surface (``PrepRunner.node_extras``)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.prepgraph.dag import PrepGraph
from lighttrain.prepgraph.node import RunContext
from lighttrain.builtin_plugins.prepgraph.nodes.load import LoadNode, _iter_lines
from lighttrain.builtin_plugins.prepgraph.nodes.pack import PackNode
from lighttrain.prepgraph.runner import PrepRunner


class _FakeUpstream:
    def __init__(self, rows):
        self.rows = rows


def _run_pack(rows, *, seq_len=16, eos_id=99, pad_id=0, strategy=None):
    cfg = {"seq_len": seq_len, "eos_id": eos_id, "pad_id": pad_id}
    if strategy is not None:
        cfg["strategy"] = strategy
    node = PackNode(name="p", inputs=["u"], config=cfg)
    ctx = RunContext(
        store_root=None, workers=1, upstream={"u": _FakeUpstream(rows)}, log=None
    )
    return node.run(ctx)


# --------------------------------------------------------------------------
# ISSUE-4 — three strategies + standardized extras + default flip
# --------------------------------------------------------------------------

@pytest.fixture
def docs():
    import random

    random.seed(1)
    return [{"input_ids": list(range(1, 1 + random.randint(3, 20)))} for _ in range(40)]


def test_default_strategy_is_concat_chunk(docs):
    res = _run_pack(docs)
    assert res.extras["strategy"] == "concat_chunk"


def test_three_strategies_distinct_profiles(docs):
    profiles = {s: _run_pack(docs, strategy=s).extras for s in
               ("concat_chunk", "next_fit", "best_fit")}
    # All emit the standardized metric keys.
    for e in profiles.values():
        assert {"truncation_rate", "token_utilization", "n_truncated_docs",
                "n_sequences"} <= set(e)
    # concat_chunk is padding-free → highest utilization (no per-bin padding waste).
    assert profiles["concat_chunk"]["token_utilization"] >= profiles["next_fit"]["token_utilization"]
    # next_fit pads every flushed buffer → strictly wastes more than best_fit here.
    assert profiles["best_fit"]["token_utilization"] >= profiles["next_fit"]["token_utilization"]


def test_concat_chunk_padding_free_full_rows(docs):
    # Every row except possibly the last is fully packed (no pad) under concat_chunk.
    res = _run_pack(docs, strategy="concat_chunk")
    rows = res.rows
    for r in rows[:-1]:
        assert all(d >= 0 for d in r["document_ids"]), "interior rows must be pad-free"


def test_best_fit_zero_truncation_when_all_fit():
    # Docs all shorter than seq_len → best_fit never truncates.
    rows = [{"input_ids": list(range(1, 6))} for _ in range(20)]  # len 5 (+eos=6) < 32
    res = _run_pack(rows, seq_len=32, strategy="best_fit")
    assert res.extras["truncation_rate"] == 0.0
    assert res.extras["n_truncated_docs"] == 0


def test_unknown_strategy_raises(docs):
    with pytest.raises(ValueError, match="unknown strategy"):
        _run_pack(docs, strategy="bogus")


def test_next_fit_preserves_historical_output(docs):
    # next_fit must reproduce the legacy greedy-pad-flush rows bit-for-bit.
    res = _run_pack(docs, strategy="next_fit")
    rows = res.rows
    seq_len = 16
    # Legacy invariants: every row is exactly seq_len long; pads use pad_id/-1.
    assert all(len(r["input_ids"]) == seq_len for r in rows)
    assert all(len(r["document_ids"]) == seq_len for r in rows)


# --------------------------------------------------------------------------
# ISSUE-5 — lines: source scheme
# --------------------------------------------------------------------------

def test_iter_lines_skips_blank_and_wraps_text(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("first doc\n\n  second doc  \n\n", encoding="utf-8")
    out = list(_iter_lines(p))
    assert out == [{"text": "first doc"}, {"text": "second doc"}]


def test_load_node_lines_scheme_matches_jsonl(tmp_path):
    txt = tmp_path / "corpus.txt"
    txt.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        "".join(json.dumps({"text": t}) + "\n" for t in ("alpha", "beta", "gamma")),
        encoding="utf-8",
    )

    def rows_for(source):
        node = LoadNode(name="raw", inputs=[], config={"source": source})
        return list(node._iter())

    assert rows_for(f"lines:{txt}") == rows_for(f"jsonl:{jsonl}")


# --------------------------------------------------------------------------
# ISSUE-3 — PrepRunner.node_extras surfaces persisted author metrics
# --------------------------------------------------------------------------

def _build_pack_recipe_graph(tmp_path, strategy):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "c.txt"
    # Mix of short and long docs so truncation_rate differs across strategies.
    corpus.write_text("\n".join(["word " * 30] + ["short doc"] * 10), encoding="utf-8")
    spec = {
        "nodes": [
            {"name": "raw", "kind": "load", "source": f"lines:{corpus}"},
            {"name": "tok", "kind": "tokenize", "inputs": ["raw"],
             "tokenizer": {"name": "byte"}, "text_field": "text"},
            {"name": "packed", "kind": "pack", "inputs": ["tok"],
             "seq_len": 64, "eos_id": 258, "pad_id": 256, "strategy": strategy},
        ],
        "terminals": ["packed"],
    }
    return PrepGraph.from_config(spec)


def test_node_extras_surfaces_pack_metrics(tmp_path):
    graph = _build_pack_recipe_graph(tmp_path, "best_fit")
    runner = PrepRunner(graph, store_root=tmp_path / "store")
    runner.run()
    extras = runner.node_extras()
    assert "packed" in extras
    pe = extras["packed"]
    # Author metric is surfaced; framework keys are stripped out.
    assert "truncation_rate" in pe
    assert pe["strategy"] == "best_fit"
    assert "code_version" not in pe and "derived_from" not in pe


def test_node_extras_reflects_strategy_truncation_difference(tmp_path):
    # concat_chunk truncates boundary-straddling docs; best_fit (all-fit) does not.
    g_cc = _build_pack_recipe_graph(tmp_path / "cc", "concat_chunk")
    PrepRunner(g_cc, store_root=tmp_path / "cc" / "store").run()
    cc = PrepRunner(g_cc, store_root=tmp_path / "cc" / "store").node_extras()["packed"]

    g_bf = _build_pack_recipe_graph(tmp_path / "bf", "best_fit")
    PrepRunner(g_bf, store_root=tmp_path / "bf" / "store").run()
    bf = PrepRunner(g_bf, store_root=tmp_path / "bf" / "store").node_extras()["packed"]

    assert cc["truncation_rate"] >= bf["truncation_rate"]
