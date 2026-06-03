"""Artifact store backends — DESIGN §12.1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.artifacts import (
    ArtifactHeader,
    ArtifactIncompleteError,
    MemmapFixedStore,
    SafetensorsShardStore,
    StaleArtifactError,
    open_artifact_store,
)


def test_safetensors_shards_put_get_contains_iter(tmp_path):
    store = SafetensorsShardStore(tmp_path / "art", shard_size=2)
    store.header.producer_signature = "test"
    store.put("s1", {"logits": torch.randn(3, 4)})
    store.put("s2", {"logits": torch.randn(3, 4)})
    store.put("s3", {"logits": torch.randn(3, 4)})  # triggers second shard
    assert store.contains("s1")
    assert store.contains("s3")
    g = store.get("s1")
    assert g["logits"].shape == (3, 4)
    manifest = store.finalize()
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["backend"] == "safetensors-shards"
    assert sorted(store.iter_keys()) == ["s1", "s2", "s3"]


def test_safetensors_shards_resume_safe(tmp_path):
    root = tmp_path / "art"
    s = SafetensorsShardStore(root, shard_size=4)
    s.put("a", {"v": torch.zeros(2)})
    s.put("b", {"v": torch.ones(2)})
    s._flush_shard()
    s._persist_index()

    # Re-open as fresh store; should see the prior index without finalize.
    s2 = SafetensorsShardStore(root, shard_size=4)
    assert s2.contains("a") and s2.contains("b")
    s2.put("a", {"v": torch.zeros(2)})  # idempotent — no error
    s2.put("c", {"v": torch.full((2,), 2.0)})
    manifest = s2.finalize()
    assert manifest.exists()


def test_memmap_fixed_enforces_shape_and_roundtrips(tmp_path):
    store = MemmapFixedStore(tmp_path / "mm")
    store.put("a", {"emb": torch.arange(8.0)})
    store.put("b", {"emb": torch.arange(8.0) + 100})
    with pytest.raises(ValueError, match="identical shapes"):
        store.put("c", {"emb": torch.zeros(16)})
    manifest = store.finalize()
    assert manifest.exists()
    rec = store.get("b")
    assert torch.allclose(rec["emb"], torch.arange(8.0) + 100)


def test_open_artifact_store_rejects_missing_manifest(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "header.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactIncompleteError):
        open_artifact_store(bad)


def test_open_artifact_store_stale_header_default_rejects(tmp_path):
    """F5 acceptance — DESIGN §25.3."""
    root = tmp_path / "art"
    s = SafetensorsShardStore(root)
    s.header.producer_signature = "v1"
    s.header.data_version = "alpha"
    s.put("s", {"x": torch.zeros(2)})
    s.finalize()

    # Mismatched expectation
    expected = ArtifactHeader(producer_signature="v1", data_version="beta")
    with pytest.raises(StaleArtifactError):
        open_artifact_store(root, expected_header=expected)
    # allow_stale=True bypasses
    opened = open_artifact_store(root, expected_header=expected, allow_stale=True)
    assert "s" in list(opened.iter_keys())


def test_open_artifact_store_matches_when_expected_empty(tmp_path):
    root = tmp_path / "art"
    s = SafetensorsShardStore(root)
    s.put("s", {"x": torch.zeros(2)})
    s.finalize()
    opened = open_artifact_store(root)  # no expectation → fine
    assert opened.contains("s")
