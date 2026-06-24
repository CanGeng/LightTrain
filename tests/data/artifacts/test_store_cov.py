"""Edge-case coverage for ``lighttrain.builtin_plugins.data.artifacts.store``.

Companion to ``test_store.py``; drives the branches that file leaves
uncovered. What this module pins:

  * ``SafetensorsShardStore.__init__`` – corrupt manifest.json falls back to
    empty index without propagating the JSONDecodeError (lines 96-97).
  * ``SafetensorsShardStore.get`` – KeyError for a sample absent from both
    index and pending (line 161).
  * ``MemmapFixedStore.__init__`` – corrupt manifest.json silently ignored
    (lines 209-210).
  * ``MemmapFixedStore.contains`` – True/False paths (line 246).
  * ``MemmapFixedStore.get`` – KeyError for unknown sample (line 253).
  * ``MemmapFixedStore`` resume-from-disk reload path via open_artifact_store.
  * ``ParquetRowStore.__init__`` – finalized-store reload branch (line 311).
  * ``ParquetRowStore._finalized_count`` – happy path, missing file, corrupt
    JSON (lines 327-333).
  * ``ParquetRowStore._reload_from_disk`` – missing manifest.json raises
    FileNotFoundError (line 344); empty sample_to_row with rows.parquet
    present raises ValueError (lines 362-364); rows.parquet missing raises
    FileNotFoundError (line 369-371); row count mismatch raises ValueError
    (line 377); row sample_id does not match manifest raises ValueError
    (line 385).
  * ``ParquetRowStore.contains`` – True/False paths (line 431).
  * ``ParquetRowStore.get`` – KeyError for unknown sample (line 438).
  * ``open_artifact_store`` – expected_header passed as a plain Mapping dict
    (line 495); unknown backend raises ValueError (line 510).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.data.artifacts import (
    ArtifactHeader,
    MemmapFixedStore,
    ParquetRowStore,
    SafetensorsShardStore,
    StaleArtifactError,
    open_artifact_store,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _finalize_store_simple(store_dir: Path, backend_cls, **kwargs) -> None:
    """Create a store, put one sample, finalize — utility for reload tests."""
    store = backend_cls(store_dir, **kwargs)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    store.finalize()


def _write_manifest_complete(
    root: Path,
    *,
    backend: str = "parquet-rows",
    count: int = 0,
) -> None:
    """Write a minimal MANIFEST_COMPLETE.json so open_artifact_store can open."""
    body = {
        "backend": backend,
        "count": count,
        "header": ArtifactHeader().to_dict(),
    }
    (root / "MANIFEST_COMPLETE.json").write_text(
        json.dumps(body), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# SafetensorsShardStore – corrupt manifest fallback (lines 96-97)
# --------------------------------------------------------------------------- #


def test_pin_shard_store_init_corrupt_manifest_falls_back_to_empty_index(
    tmp_path: Path,
) -> None:
    """``SafetensorsShardStore.__init__`` silently falls back to empty index
    when manifest.json contains invalid JSON (lines 96-97).

    Pin: the constructor must not propagate JSONDecodeError; subsequent
    puts are unaffected.
    """
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text("NOT_VALID_JSON{{{", encoding="utf-8")

    store = SafetensorsShardStore(root)
    assert store._index == {}, "corrupt manifest should yield empty index"
    # Putting a sample after corrupt-manifest load must work normally.
    store.put("s1", {"logits": torch.tensor([1.0])})
    assert store.contains("s1")


# --------------------------------------------------------------------------- #
# SafetensorsShardStore – get KeyError (line 161)
# --------------------------------------------------------------------------- #


def test_invariant_shard_store_get_unknown_sample_raises_key_error(
    tmp_path: Path,
) -> None:
    """``SafetensorsShardStore.get`` raises ``KeyError`` for a sample that is
    neither in ``_index`` nor in ``_pending`` (line 161).
    """
    store = SafetensorsShardStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0])})
    with pytest.raises(KeyError):
        store.get("missing_sample")


# --------------------------------------------------------------------------- #
# MemmapFixedStore – corrupt manifest fallback (lines 209-210)
# --------------------------------------------------------------------------- #


def test_pin_memmap_store_init_corrupt_manifest_falls_back_to_empty(
    tmp_path: Path,
) -> None:
    """``MemmapFixedStore.__init__`` silently ignores a corrupt manifest.json
    rather than raising JSONDecodeError (lines 209-210).

    Pin: the store initialises with an empty index; subsequent puts work.
    """
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text("BROKEN{{{JSON", encoding="utf-8")

    store = MemmapFixedStore(root)
    assert store._index == {}
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    assert store.contains("s1")


# --------------------------------------------------------------------------- #
# MemmapFixedStore – contains (line 246)
# --------------------------------------------------------------------------- #


def test_invariant_memmap_contains_true_and_false(tmp_path: Path) -> None:
    """``MemmapFixedStore.contains`` returns True for a known sample and False
    for an unknown one (line 246).
    """
    store = MemmapFixedStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0])})
    assert store.contains("s1") is True
    assert store.contains("absent") is False


# --------------------------------------------------------------------------- #
# MemmapFixedStore – get KeyError (line 253)
# --------------------------------------------------------------------------- #


def test_invariant_memmap_get_unknown_sample_raises_key_error(
    tmp_path: Path,
) -> None:
    """``MemmapFixedStore.get`` raises ``KeyError`` for an absent sample_id
    (line 253).
    """
    store = MemmapFixedStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0])})
    store.finalize()
    with pytest.raises(KeyError):
        store.get("missing")


# --------------------------------------------------------------------------- #
# MemmapFixedStore – reload via open_artifact_store
# --------------------------------------------------------------------------- #


def test_invariant_memmap_reload_after_finalize(tmp_path: Path) -> None:
    """``open_artifact_store`` on a memmap-fixed store reconstructs index from
    disk so ``get`` returns the same tensors written before finalize.
    """
    torch.manual_seed(42)
    root = tmp_path / "store"
    store = MemmapFixedStore(root)
    t = torch.randn(3, 4)
    store.put("s1", {"logits": t})
    store.finalize()

    re = open_artifact_store(root)
    out = re.get("s1")
    torch.testing.assert_close(out["logits"], t, atol=1e-5, rtol=1e-4)


def test_pin_memmap_reload_populates_next_row_from_max_index(
    tmp_path: Path,
) -> None:
    """After reopening a memmap store, ``_next_row`` is set to
    ``max(index.values()) + 1`` so a new put goes to the correct row.

    This exercises the ``_next_row = max(self._index.values()) + 1`` path in
    MemmapFixedStore.__init__ (line 207).
    """
    root = tmp_path / "store"
    store = MemmapFixedStore(root)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    store.put("s2", {"logits": torch.tensor([3.0, 4.0])})
    store.finalize()

    # Re-open via direct constructor (not finalized) to check _next_row.
    store2 = MemmapFixedStore(root)
    assert store2._next_row == 2, (
        f"expected _next_row=2 after 2-sample store, got {store2._next_row}"
    )


# --------------------------------------------------------------------------- #
# ParquetRowStore – finalized-store reload branch (line 311)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_init_finalized_store_reloads_from_disk(
    tmp_path: Path,
) -> None:
    """When ``ParquetRowStore.__init__`` detects ``MANIFEST_COMPLETE.json`` it
    calls ``_reload_from_disk`` (line 311), so the in-memory index reflects
    what was persisted.
    """
    root = tmp_path / "store"
    store = ParquetRowStore(root)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    store.finalize()

    # Re-instantiate directly — must reload without calling open_artifact_store.
    store2 = ParquetRowStore(root)
    assert store2.contains("s1"), "reload should have populated _index"
    out = store2.get("s1")
    torch.testing.assert_close(out["logits"], torch.tensor([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# ParquetRowStore._finalized_count (lines 327-333)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_finalized_count_happy_path(tmp_path: Path) -> None:
    """``_finalized_count`` returns the integer stored in
    ``MANIFEST_COMPLETE.json["count"]`` (line 331).
    """
    root = tmp_path / "store"
    root.mkdir()
    (root / "MANIFEST_COMPLETE.json").write_text(
        json.dumps({"count": 7}), encoding="utf-8"
    )
    store = ParquetRowStore.__new__(ParquetRowStore)
    store.root = root
    assert store._finalized_count() == 7


def test_invariant_parquet_finalized_count_missing_file_returns_zero(
    tmp_path: Path,
) -> None:
    """``_finalized_count`` returns 0 when ``MANIFEST_COMPLETE.json`` does not
    exist (line 328).
    """
    root = tmp_path / "store"
    root.mkdir()
    store = ParquetRowStore.__new__(ParquetRowStore)
    store.root = root
    assert store._finalized_count() == 0


def test_pin_parquet_finalized_count_corrupt_json_returns_zero(
    tmp_path: Path,
) -> None:
    """``_finalized_count`` returns 0 when the JSON is corrupt or the count
    field is not an int (lines 332-333).

    Pin: current behavior swallows the error and returns 0 rather than
    propagating JSONDecodeError / TypeError / ValueError.
    """
    root = tmp_path / "store"
    root.mkdir()
    (root / "MANIFEST_COMPLETE.json").write_text("BAD_JSON", encoding="utf-8")
    store = ParquetRowStore.__new__(ParquetRowStore)
    store.root = root
    assert store._finalized_count() == 0


def test_pin_parquet_finalized_count_non_int_count_returns_zero(
    tmp_path: Path,
) -> None:
    """``_finalized_count`` returns 0 when ``count`` is not coercible to int
    (lines 332-333).
    """
    root = tmp_path / "store"
    root.mkdir()
    (root / "MANIFEST_COMPLETE.json").write_text(
        json.dumps({"count": None}), encoding="utf-8"
    )
    store = ParquetRowStore.__new__(ParquetRowStore)
    store.root = root
    assert store._finalized_count() == 0


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – missing manifest (line 344)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_reload_missing_manifest_raises(tmp_path: Path) -> None:
    """``_reload_from_disk`` raises ``FileNotFoundError`` when ``manifest.json``
    is absent from a finalized store (line 344).
    """
    root = tmp_path / "store"
    root.mkdir()
    # Write MANIFEST_COMPLETE.json so _finalized == True, but omit manifest.json.
    _write_manifest_complete(root, backend="parquet-rows", count=1)

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        ParquetRowStore(root)


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – empty sample_to_row with data on disk
# (lines 362-364)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_reload_empty_index_but_rows_present_raises(
    tmp_path: Path,
) -> None:
    """When ``sample_to_row`` is empty but ``rows.parquet`` is present on disk,
    ``_reload_from_disk`` raises ``ValueError`` (lines 362-364).

    The store is corrupt: an empty manifest that claims no samples yet has a
    data file is inconsistent.
    """
    root = tmp_path / "store"
    # Create a minimal finalized store, then corrupt the manifest index.
    store = ParquetRowStore(root)
    store.put("s1", {"logits": torch.tensor([1.0])})
    store.finalize()

    # Zero out sample_to_row while leaving rows.parquet intact.
    (root / "manifest.json").write_text(
        json.dumps({"sample_to_row": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="corrupt manifest"):
        ParquetRowStore(root)


def test_invariant_parquet_reload_empty_index_positive_count_raises(
    tmp_path: Path,
) -> None:
    """Empty ``sample_to_row`` but a positive ``count`` in MANIFEST_COMPLETE
    also signals a corrupt manifest (lines 363-364).
    """
    root = tmp_path / "store"
    root.mkdir()

    # Craft a finalized state: MANIFEST_COMPLETE with count>0, empty index.
    _write_manifest_complete(root, backend="parquet-rows", count=3)
    (root / "manifest.json").write_text(
        json.dumps({"sample_to_row": {}}), encoding="utf-8"
    )
    # No rows.parquet on disk.

    with pytest.raises(ValueError, match="corrupt manifest"):
        ParquetRowStore(root)


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – rows.parquet missing (line 369-371)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_reload_samples_but_no_rows_file_raises(
    tmp_path: Path,
) -> None:
    """When ``sample_to_row`` names samples but ``rows.parquet`` does not exist,
    ``_reload_from_disk`` raises ``FileNotFoundError`` (lines 369-371).
    """
    root = tmp_path / "store"
    root.mkdir()

    _write_manifest_complete(root, backend="parquet-rows", count=1)
    (root / "manifest.json").write_text(
        json.dumps({"sample_to_row": {"s1": 0}}), encoding="utf-8"
    )
    # rows.parquet deliberately absent.

    with pytest.raises(FileNotFoundError, match="rows.parquet"):
        ParquetRowStore(root)


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – row count mismatch (line 377)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_reload_row_count_mismatch_raises(
    tmp_path: Path,
) -> None:
    """When ``rows.parquet`` has a different number of rows than ``sample_to_row``
    declares, ``_reload_from_disk`` raises ``ValueError`` (line 377).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "store"
    root.mkdir()

    # Write a rows.parquet with 2 rows.
    sample_id_col = pa.array(["s1", "s2"])
    table = pa.table({"sample_id": sample_id_col})
    pq.write_table(table, str(root / "rows.parquet"))

    # But manifest.json only lists 1 sample.
    _write_manifest_complete(root, backend="parquet-rows", count=1)
    (root / "manifest.json").write_text(
        json.dumps({"sample_to_row": {"s1": 0}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="inconsistent"):
        ParquetRowStore(root)


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – sample_id mismatch in row (line 385)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_reload_sample_id_mismatch_raises(
    tmp_path: Path,
) -> None:
    """When a row's ``sample_id`` does not match the row index in
    ``sample_to_row``, ``_reload_from_disk`` raises ``ValueError`` (line 385).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "store"
    root.mkdir()

    # The parquet has "s1" but the manifest maps "s1" to row 1 (not 0).
    sample_id_col = pa.array(["s1"])
    table = pa.table({"sample_id": sample_id_col})
    pq.write_table(table, str(root / "rows.parquet"))

    _write_manifest_complete(root, backend="parquet-rows", count=1)
    (root / "manifest.json").write_text(
        json.dumps({"sample_to_row": {"s1": 1}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        ParquetRowStore(root)


# --------------------------------------------------------------------------- #
# ParquetRowStore.contains (line 431)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_contains_true_and_false(tmp_path: Path) -> None:
    """``ParquetRowStore.contains`` returns True for a known sample and False for
    an unknown sample (line 431).
    """
    store = ParquetRowStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0])})
    assert store.contains("s1") is True
    assert store.contains("ghost") is False


# --------------------------------------------------------------------------- #
# ParquetRowStore.get – KeyError (line 438)
# --------------------------------------------------------------------------- #


def test_invariant_parquet_get_unknown_sample_raises_key_error(
    tmp_path: Path,
) -> None:
    """``ParquetRowStore.get`` raises ``KeyError`` for a sample_id not in
    the index (line 438).
    """
    store = ParquetRowStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0])})
    with pytest.raises(KeyError):
        store.get("missing")


# --------------------------------------------------------------------------- #
# open_artifact_store – expected_header as plain dict Mapping (line 495)
# --------------------------------------------------------------------------- #


def test_invariant_open_store_expected_header_as_mapping_dict(
    tmp_path: Path,
) -> None:
    """``open_artifact_store`` accepts ``expected_header`` as a plain ``dict``
    (Mapping) and converts it to an ``ArtifactHeader`` internally (line 495).

    A non-conflicting mapping must succeed without raising ``StaleArtifactError``.
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.header.data_version = "v1"
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    # Pass expected_header as a dict (Mapping) — must NOT raise.
    opened = open_artifact_store(
        root, expected_header={"data_version": "v1"}
    )
    assert opened.contains("s")


def test_invariant_open_store_expected_header_mapping_mismatch_raises(
    tmp_path: Path,
) -> None:
    """A plain-dict expected_header whose fields conflict with the on-disk header
    raises ``StaleArtifactError`` (lines 494-500), exercising the Mapping branch.
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.header.data_version = "alpha"
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    with pytest.raises(StaleArtifactError):
        open_artifact_store(root, expected_header={"data_version": "beta"})


# --------------------------------------------------------------------------- #
# open_artifact_store – unknown backend raises ValueError (line 510)
# --------------------------------------------------------------------------- #


def test_invariant_open_store_unknown_backend_raises_value_error(
    tmp_path: Path,
) -> None:
    """``open_artifact_store`` raises ``ValueError`` when the backend name is
    not one of the three known backends (line 510).
    """
    root = tmp_path / "store"
    root.mkdir()
    # Craft a MANIFEST_COMPLETE.json that declares a non-existent backend.
    body = {
        "backend": "totally-unknown-backend",
        "count": 0,
        "header": ArtifactHeader().to_dict(),
    }
    (root / "MANIFEST_COMPLETE.json").write_text(
        json.dumps(body), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown artifact_store backend"):
        open_artifact_store(root)


def test_invariant_open_store_explicit_unknown_backend_kwarg_raises(
    tmp_path: Path,
) -> None:
    """Passing an unknown ``backend`` keyword directly to ``open_artifact_store``
    also raises ``ValueError`` (line 510).
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    with pytest.raises(ValueError, match="unknown artifact_store backend"):
        open_artifact_store(root, backend="not-a-real-backend")


# --------------------------------------------------------------------------- #
# ParquetRowStore._reload_from_disk – genuinely empty store (no rows) is valid
# --------------------------------------------------------------------------- #


def test_invariant_parquet_empty_store_reload_is_valid(tmp_path: Path) -> None:
    """An empty ``ParquetRowStore`` (zero samples put) can be finalized and
    reopened without error.

    The reload path takes the early-return branch (line 369 is NOT reached)
    because ``sample_to_row`` is empty AND ``rows.parquet`` is absent AND
    ``count == 0`` — which is the legitimate empty-store case.
    """
    root = tmp_path / "store"
    store = ParquetRowStore(root)
    store.finalize()  # No samples.

    store2 = ParquetRowStore(root)
    assert list(store2.iter_keys()) == []


# --------------------------------------------------------------------------- #
# SafetensorsShardStore – contains after flush (regression)
# --------------------------------------------------------------------------- #


def test_invariant_shard_contains_true_after_flush(tmp_path: Path) -> None:
    """``contains`` returns True for a flushed sample (present in ``_index``)."""
    store = SafetensorsShardStore(tmp_path / "store", shard_size=1)
    store.put("s1", {"logits": torch.tensor([1.0])})
    store.put("s2", {"logits": torch.tensor([2.0])})  # triggers flush on s1
    assert store.contains("s1") is True
    assert store.contains("s2") is True
    assert store.contains("never_put") is False


# --------------------------------------------------------------------------- #
# MemmapFixedStore – resume across multiple sample puts
# --------------------------------------------------------------------------- #


def test_invariant_memmap_multi_sample_get_correct_values(
    tmp_path: Path,
) -> None:
    """MemmapFixedStore correctly retrieves each of N samples by row-indexed
    binary seek, ensuring put-order does not corrupt get-order.
    """
    torch.manual_seed(7)
    root = tmp_path / "store"
    store = MemmapFixedStore(root)
    tensors = {}
    for i in range(5):
        sid = f"sample_{i}"
        t = torch.arange(i * 4, i * 4 + 4, dtype=torch.float32)
        tensors[sid] = t
        store.put(sid, {"data": t})
    store.finalize()

    for sid, expected in tensors.items():
        got = store.get(sid)
        torch.testing.assert_close(got["data"], expected)
