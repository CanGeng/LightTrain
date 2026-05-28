"""JoinNode in PrepGraph: artifact join at prep time (REVIEW #11 / DESIGN §7.7.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lighttrain.artifacts import ModelForwardProducer
from lighttrain.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.prepgraph import nodes  # noqa: F401 — register prep_node entries
from lighttrain.prepgraph.node import NodeResult, RunContext
from lighttrain.prepgraph.nodes.join import JoinNode


def _build_artifact_store(root: Path, sample_ids: list[str]) -> Path:
    model = TinyCausalLM(vocab_size=60, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    producer = ModelForwardProducer(
        model=model,
        store={"name": "safetensors-shards", "root": str(root), "shard_size": 8},
        artifact_name="teacher", artifact_version="v1",
        header_overrides={"data_version": "v1", "model_id": "teacher_tiny_v1"},
    )
    producer.prepare()
    for sid in sample_ids:
        producer.produce({
            "id": sid,
            "input_ids": torch.randint(0, 60, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "labels": torch.randint(0, 60, (8,)),
        })
    producer.finalize()
    return root


def _make_ctx(upstream_rows: list[dict], tmp_path: Path) -> RunContext:
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {
        "up": NodeResult(fingerprint="x", schema_kind="rows", rows=list(upstream_rows))
    }
    return ctx


def test_join_require_emits_aux_fields(tmp_path):
    art_root = _build_artifact_store(tmp_path / "art", ["a0", "a1"])
    upstream_rows = [
        {"id": "a0", "input_ids": [1, 2, 3]},
        {"id": "a1", "input_ids": [4, 5, 6]},
    ]
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "teacher"}],
            "missing": "require",
        },
    )
    res = node.run(_make_ctx(upstream_rows, tmp_path))
    rows = list(res.rows)
    assert len(rows) == 2
    for row in rows:
        aux_keys = [k for k in row if k.startswith("aux.teacher.")]
        # At least logits should be present
        assert any(k.endswith("logits") for k in aux_keys), aux_keys


def test_join_drop_filters_missing(tmp_path):
    art_root = _build_artifact_store(tmp_path / "art", ["a0"])
    upstream_rows = [
        {"id": "a0", "input_ids": [1]},
        {"id": "missing", "input_ids": [2]},
    ]
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "teacher"}],
            "missing": "drop",
        },
    )
    res = node.run(_make_ctx(upstream_rows, tmp_path))
    rows = list(res.rows)
    assert [r["id"] for r in rows] == ["a0"]


def test_join_require_raises_on_missing(tmp_path):
    art_root = _build_artifact_store(tmp_path / "art", ["a0"])
    upstream_rows = [{"id": "missing", "input_ids": [1]}]
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "teacher"}],
            "missing": "require",
        },
    )
    with pytest.raises(KeyError):
        node.run(_make_ctx(upstream_rows, tmp_path))
