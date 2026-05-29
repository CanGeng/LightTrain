"""Adversarial tests for ``ArtifactJoinedDataset``.

No legacy dedicated test file existed, so this file establishes the baseline
behavioural pins:
  * ``aux.<namespace>.<field>`` keys carry the store's tensor values via
    ``torch.testing.assert_close`` (not just shape)
  * ``missing='require'`` → ``KeyError``
  * ``missing='drop'`` → ``None``; ``drop_none_collator`` filters
  * ``missing='fill_zero'`` → zero-valued tensor with the header's declared
    shape (ART-FILL invariant)
  * Multi-store join produces non-colliding aux keys per namespace
  * ``derive_sample_id`` fallback works when the base dataset omits the id
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lighttrain.artifacts import (
    ArtifactHeader,
    ArtifactJoinedDataset,
    SafetensorsShardStore,
    drop_none_collator,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _build_store(
    root: Path,
    *,
    samples: dict[str, dict[str, torch.Tensor]],
    field_schema: dict[str, str] | None = None,
) -> Path:
    """Write a finalized safetensors-shards store with the given samples."""
    header = ArtifactHeader(
        producer_signature="test",
        dtype="torch.float32",
        field_schema=field_schema or {},
    )
    store = SafetensorsShardStore(root, header=header)
    for sid, tensors in samples.items():
        store.put(sid, tensors)
    store.finalize()
    return root


class _ListDataset:
    """A minimal map-style dataset over a list of dicts."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


# --------------------------------------------------------------------------- #
# Happy path: aux.<ns>.<field> carries the right values                       #
# --------------------------------------------------------------------------- #


def test_joined_dataset_attaches_aux_namespace_keys(tmp_path: Path) -> None:
    """Aux tensors land under ``aux.<namespace>.<field>`` with value-level parity.

    Input: 2 samples, a store with logits per sample, namespace = "feat".
    The joined dataset must yield a merged dict where
    ``aux.feat.logits`` matches the stored tensor via ``assert_close``.
    """
    torch.manual_seed(0)
    expected = {"s1": torch.randn(3), "s2": torch.randn(3)}
    root = tmp_path / "art"
    _build_store(root, samples={k: {"logits": v} for k, v in expected.items()})

    base = _ListDataset([{"id": "s1"}, {"id": "s2"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "require"}],
    )
    for i, sid in enumerate(("s1", "s2")):
        row = ds[i]
        assert row is not None
        assert row["id"] == sid
        torch.testing.assert_close(
            row["aux.feat.logits"], expected[sid], atol=1e-5, rtol=1e-4
        )


# --------------------------------------------------------------------------- #
# missing policies                                                            #
# --------------------------------------------------------------------------- #


def test_joined_dataset_missing_require_raises(tmp_path: Path) -> None:
    """``missing='require'``: a sample absent from the store → ``KeyError``.

    Input: store has "s1" only; base dataset asks for "s2".
    Contract: require = no fallback, surface error.
    """
    root = tmp_path / "art"
    _build_store(root, samples={"s1": {"logits": torch.zeros(2)}})

    base = _ListDataset([{"id": "s2"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "require"}],
    )
    with pytest.raises(KeyError, match="not present"):
        ds[0]


def test_joined_dataset_missing_drop_returns_none_filtered_by_collator(
    tmp_path: Path,
) -> None:
    """``missing='drop'`` returns ``None``; ``drop_none_collator`` filters.

    Input: store has "s1"; base asks for s1 (hit) and s2 (miss).
    Pin: ds[1] is None; ``drop_none_collator`` wrapped around a list
    collator produces a batch with only s1.
    """
    root = tmp_path / "art"
    _build_store(root, samples={"s1": {"logits": torch.zeros(2)}})

    base = _ListDataset([{"id": "s1"}, {"id": "s2"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "drop"}],
    )
    assert ds[0] is not None
    assert ds[1] is None

    collator = drop_none_collator(lambda samples: samples)
    raw = [ds[0], ds[1]]
    batch = collator(raw)
    assert len(batch) == 1
    assert batch[0]["id"] == "s1"


def test_invariant_joined_dataset_fill_uses_header_schema_zero(
    tmp_path: Path,
) -> None:
    """``missing='fill_zero'`` substitutes ``torch.zeros`` of the header's
    declared shape; the filled tensor is bit-exactly zero.

    Invariant (ART-FILL): missing samples under fill_zero get tensors whose
    shape comes from ``header.field_schema[k]`` and whose values are all 0.
    """
    root = tmp_path / "art"
    # Tell the header that ``logits`` is shape (3,) via field_schema.
    _build_store(
        root,
        samples={"s1": {"logits": torch.ones(3)}},
        field_schema={"logits": "(3,)"},
    )

    base = _ListDataset([{"id": "missing-sample"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "fill_zero"}],
    )
    row = ds[0]
    assert row is not None
    tensor = row["aux.feat.logits"]
    assert tensor.shape == (3,)
    torch.testing.assert_close(
        tensor, torch.zeros(3), atol=1e-5, rtol=1e-4
    )


# --------------------------------------------------------------------------- #
# Multi-store: namespaces do not collide                                      #
# --------------------------------------------------------------------------- #


def test_joined_dataset_multi_store_merge(tmp_path: Path) -> None:
    """Two stores with distinct namespaces both contribute aux keys.

    Input: store_a (ns="alpha") + store_b (ns="beta"), both with the same
    sample id "s1" but different tensor values.
    Pin: merged dict has BOTH ``aux.alpha.feat`` and ``aux.beta.feat`` with
    the corresponding values.
    """
    root_a = tmp_path / "art_a"
    root_b = tmp_path / "art_b"
    torch.manual_seed(0)
    val_a = torch.randn(4)
    val_b = torch.randn(4)
    _build_store(root_a, samples={"s1": {"feat": val_a}})
    _build_store(root_b, samples={"s1": {"feat": val_b}})

    base = _ListDataset([{"id": "s1"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[
            {"store": str(root_a), "namespace": "alpha", "missing": "require"},
            {"store": str(root_b), "namespace": "beta", "missing": "require"},
        ],
    )
    row = ds[0]
    assert row is not None
    torch.testing.assert_close(row["aux.alpha.feat"], val_a, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(row["aux.beta.feat"], val_b, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Sample-id derive fallback                                                   #
# --------------------------------------------------------------------------- #


def test_joined_dataset_derive_sample_id_when_base_omits_id(
    tmp_path: Path,
) -> None:
    """When the base sample lacks ``id``, the joiner falls back to
    ``derive_sample_id``; if the store has that derived id, the aux key
    appears.

    Input: base sample has only ``input_ids`` (no ``id``); we compute the
    derived id ourselves and pre-fill the store under that id.
    Analytical: ``derive_sample_id({"input_ids": [...]})`` is deterministic
    over the same input list, so the same derived id appears at
    ``store.put`` and ``ds.__getitem__`` retrieval.
    """
    from lighttrain.data.core._schema import derive_sample_id

    payload = torch.tensor([7.0, 8.0])
    sample = {"input_ids": [1, 2, 3]}
    derived = derive_sample_id(sample)

    root = tmp_path / "art"
    _build_store(root, samples={derived: {"logits": payload}})

    base = _ListDataset([sample])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "require"}],
    )
    row = ds[0]
    assert row is not None
    assert row["id"] == derived
    torch.testing.assert_close(row["aux.feat.logits"], payload, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# drop_none_collator: empty batch error                                       #
# --------------------------------------------------------------------------- #


def test_invariant_joined_dataset_fill_with_empty_field_schema_no_op(
    tmp_path: Path,
) -> None:
    """``missing='fill_zero'`` with an empty ``header.field_schema`` is a silent
    no-op for that store: no ``aux.<ns>.*`` keys are added; no exception.

    Adversarial PR-reviewer pass: a lazy newcomer might rewrite the fill
    branch as ``raise KeyError if not header.field_schema``. This test
    pins the current "no-op" contract so that change would surface as a
    test failure rather than a silent semantic shift.

    Input: store with header missing ``field_schema``; missing sample
    requested under ``fill_zero``. The merged row must contain the base
    sample's keys but no ``aux.feat.*`` keys.
    """
    root = tmp_path / "art"
    # Empty field_schema = the producer never declared per-field shapes.
    _build_store(
        root,
        samples={"s1": {"logits": torch.ones(3)}},
        field_schema={},
    )

    base = _ListDataset([{"id": "missing-sample"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "feat", "missing": "fill_zero"}],
    )
    row = ds[0]
    assert row is not None
    assert row["id"] == "missing-sample"
    aux_keys = [k for k in row if k.startswith("aux.feat.")]
    assert aux_keys == [], (
        f"fill_zero with empty field_schema should add no aux keys; "
        f"got {aux_keys}"
    )


def test_drop_none_collator_raises_when_batch_emptied() -> None:
    """If every sample in a batch is ``None``, the collator raises.

    Contract: callers expect a non-empty batch downstream; silent empty
    batches would mis-track training step counts.
    """
    collator = drop_none_collator(lambda samples: samples)
    with pytest.raises(RuntimeError, match="entire batch"):
        collator([None, None])
