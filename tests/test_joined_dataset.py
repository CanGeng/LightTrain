"""ArtifactJoinedDataset — DESIGN §12.1 / §12.2."""

from __future__ import annotations

import pytest
import torch

from lighttrain.artifacts import ArtifactJoinedDataset, SafetensorsShardStore


def _make_store(root, ids, *, dim=4):
    s = SafetensorsShardStore(root, shard_size=16)
    s.header.producer_signature = "test"
    s.header.field_schema = {"emb": str((dim,))}
    s.header.dtype = "torch.float32"
    for sid in ids:
        s.put(sid, {"emb": torch.full((dim,), float(sid[-1]))})
    s.finalize()
    return s


def _base(ids):
    return [
        {"id": sid, "input_ids": [1, 2, 3], "labels": [1, 2, 3]}
        for sid in ids
    ]


def test_join_inserts_aux_namespace(tmp_path):
    _make_store(tmp_path / "store", ["a1", "a2", "a3"])
    base = _base(["a1", "a2", "a3"])
    ds = ArtifactJoinedDataset(
        base, join=[{"store": str(tmp_path / "store"), "namespace": "teacher"}]
    )
    row = ds[0]
    assert "aux.teacher.emb" in row
    assert torch.allclose(row["aux.teacher.emb"], torch.full((4,), 1.0))


def test_join_require_missing_raises(tmp_path):
    _make_store(tmp_path / "store", ["a1", "a2"])
    base = _base(["a1", "missing"])
    ds = ArtifactJoinedDataset(
        base, join=[{"store": str(tmp_path / "store"), "namespace": "teacher"}]
    )
    assert ds[0] is not None
    with pytest.raises(KeyError):
        ds[1]


def test_join_drop_returns_none_on_miss(tmp_path):
    _make_store(tmp_path / "store", ["a1"])
    base = _base(["a1", "missing"])
    ds = ArtifactJoinedDataset(
        base, join=[{"store": str(tmp_path / "store"), "namespace": "t",
                     "missing": "drop"}]
    )
    assert ds[0] is not None
    assert ds[1] is None


def test_join_fill_zero_uses_field_schema_shape(tmp_path):
    _make_store(tmp_path / "store", ["a1"])
    base = _base(["a1", "missing"])
    ds = ArtifactJoinedDataset(
        base, join=[{"store": str(tmp_path / "store"), "namespace": "t",
                     "missing": "fill_zero"}]
    )
    row = ds[1]
    assert row is not None
    aux = row["aux.t.emb"]
    assert aux.shape == (4,)
    assert torch.all(aux == 0)


def test_join_multi_source(tmp_path):
    _make_store(tmp_path / "store_a", ["s1"], dim=2)
    _make_store(tmp_path / "store_b", ["s1"], dim=3)
    base = _base(["s1"])
    ds = ArtifactJoinedDataset(
        base,
        join=[
            {"store": str(tmp_path / "store_a"), "namespace": "A"},
            {"store": str(tmp_path / "store_b"), "namespace": "B"},
        ],
    )
    row = ds[0]
    assert row is not None
    assert row["aux.A.emb"].shape == (2,)
    assert row["aux.B.emb"].shape == (3,)
