"""ModelForwardProducer — DESIGN §12.1."""

from __future__ import annotations

import torch

from lighttrain.artifacts import ModelForwardProducer, open_artifact_store
from lighttrain.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.models.extras import ExtraOutputSpec


def _tiny_samples(n: int = 4, T: int = 8, vocab: int = 60):
    samples = []
    for i in range(n):
        ids = torch.randint(0, vocab, (T,))
        samples.append({
            "id": f"sample_{i:02d}",
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        })
    return samples


def test_producer_writes_logits_and_hidden_states(tmp_path):
    model = TinyCausalLM(
        vocab_size=60, d_model=32, n_layers=2, n_heads=4, max_seq_len=8,
        output_hidden_states=True,
    )
    samples = _tiny_samples(3)
    producer = ModelForwardProducer(
        model=model,
        store={"name": "safetensors-shards", "root": str(tmp_path / "art"), "shard_size": 8},
        extras=[ExtraOutputSpec(name="logits_topk_8", source="lm_head",
                                transform={"topk": 8})],
        collect_hidden_states=True,
        header_overrides={"data_version": "tiny", "model_id": "test_tiny"},
        artifact_name="art", artifact_version="v1",
    )
    producer.prepare()
    for s in samples:
        producer.produce(s)
    manifest = producer.finalize()
    assert manifest.exists()

    store = open_artifact_store(tmp_path / "art")
    keys = sorted(store.iter_keys())
    assert keys == ["sample_00", "sample_01", "sample_02"]
    rec = store.get("sample_00")
    assert "logits_topk_8.values" in rec
    assert "logits_topk_8.indices" in rec
    assert rec["logits_topk_8.values"].shape[-1] == 8
    assert "hidden_states_layers" in rec
    assert rec["hidden_states_layers"].shape[0] == 3  # 2 blocks + 1 emb


def test_producer_resume_skips_already_present_samples(tmp_path):
    model = TinyCausalLM(vocab_size=60, d_model=32, n_layers=2, n_heads=4, max_seq_len=8)
    root = tmp_path / "art"
    samples = _tiny_samples(2)

    p1 = ModelForwardProducer(
        model=model,
        store={"name": "safetensors-shards", "root": str(root), "shard_size": 8},
    )
    p1.prepare()
    p1.produce(samples[0])
    # do NOT finalize — simulate crash

    p2 = ModelForwardProducer(
        model=model,
        store={"name": "safetensors-shards", "root": str(root), "shard_size": 8},
    )
    p2.prepare()
    p2.produce(samples[0])  # idempotent skip
    p2.produce(samples[1])
    p2.finalize()
    assert sorted(p2.store.iter_keys()) == ["sample_00", "sample_01"]
