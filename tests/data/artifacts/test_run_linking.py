"""ModelForwardProducer.finalize uses the explicit run_node_id when given
instead of guessing from iter_nodes (REVIEW #9)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.artifacts import ModelForwardProducer
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.observability.lineage.store import LineageStore


def test_finalize_uses_explicit_run_node_id(tmp_path):
    model = TinyCausalLM(vocab_size=60, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    sample = {
        "id": "s_00",
        "input_ids": torch.randint(0, 60, (8,)),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "labels": torch.randint(0, 60, (8,)),
    }

    ls = LineageStore(tmp_path / "lineage.sqlite")
    # Two run nodes — older first, then the "correct" newer one.
    older = ls.upsert_node(
        kind="run", name="prev_run", version="prev_run",
        run_id="prev_run", payload={"started_ts": 1.0},
    )
    current = ls.upsert_node(
        kind="run", name="current_run", version="current_run",
        run_id="current_run", payload={"started_ts": 2.0},
    )

    producer = ModelForwardProducer(
        model=model,
        store={"name": "safetensors-shards", "root": str(tmp_path / "art"), "shard_size": 8},
        artifact_name="art", artifact_version="v1",
    )
    producer.prepare({
        "lineage_store": ls,
        "run_node_id": current,
        "artifact_name": "art",
    })
    producer.produce(sample)
    producer.finalize()

    art_nodes = [n for n in ls.iter_nodes(kind="artifact")]
    assert len(art_nodes) == 1
    art_id = art_nodes[0]["id"]

    # `produced_by` edge must point at the CORRECT run, not the older one.
    edges_to_art = ls.edges_to(art_id, kind="produced_by")
    assert len(edges_to_art) == 1
    assert int(edges_to_art[0]["src"]) == int(current)
    assert int(edges_to_art[0]["src"]) != int(older)
