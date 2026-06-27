"""Edge-case tests for ``lighttrain.data.cache._memmap``.

The fixed-shape memmap cache (``write_memmap`` / ``read_header`` /
``MemmapDataset``) had almost no direct coverage. This pins:

* **MemmapHeader**: ``to_dict`` / ``from_dict`` round-trip.
* **write_memmap**: default dtypes, per-row truncation (``[:seq_len]``) and
  zero-padding, missing-field skip, atomic overwrite of an existing dir.
* **read_header**: ``None`` when absent.
* **MemmapDataset**: round-trip values, ``__len__``, int-coerced ``__getitem__``,
  default ``attention_mask`` (1 where ``input_ids != 0``) and ``labels``
  (``-100`` on padding), and the two missing-file guards.
* **_row_bytes**: per-row byte width (currently unused internally — pinned
  directly).
"""

from __future__ import annotations

import pytest

from lighttrain.data.cache._memmap import (
    DATA_NAME,
    MemmapDataset,
    MemmapHeader,
    _row_bytes,
    read_header,
    write_memmap,
)

_FIELDS = ("input_ids", "position_ids", "document_ids")


# ---------------------------------------------------------------------------
# MemmapHeader / _row_bytes
# ---------------------------------------------------------------------------

def test_header_to_dict_from_dict_roundtrip():
    """Header survives a dict round-trip with coerced field types."""
    h = MemmapHeader(seq_len=4, n_rows=2, fields=["input_ids"], dtypes={"input_ids": "int64"})
    h2 = MemmapHeader.from_dict(h.to_dict())
    assert (h2.seq_len, h2.n_rows, h2.fields, h2.dtypes) == (4, 2, ["input_ids"], {"input_ids": "int64"})


def test_row_bytes_sums_field_widths():
    """``_row_bytes`` = sum over fields of ``seq_len * itemsize``."""
    h = MemmapHeader(seq_len=4, n_rows=1, fields=["a", "b"], dtypes={"a": "int64", "b": "int8"})
    assert _row_bytes(h) == 4 * 8 + 4 * 1


# ---------------------------------------------------------------------------
# write_memmap + MemmapDataset round-trip
# ---------------------------------------------------------------------------

def test_write_read_roundtrip_preserves_values(tmp_path):
    """A full-width write reads back the exact rows; header is materialized."""
    rows = [
        {"input_ids": [1, 2, 3, 4], "position_ids": [0, 1, 2, 3], "document_ids": [0, 0, 0, 0]},
        {"input_ids": [5, 6, 7, 8], "position_ids": [0, 1, 0, 1], "document_ids": [0, 0, 1, 1]},
    ]
    write_memmap(tmp_path, rows, seq_len=4)
    ds = MemmapDataset(tmp_path)
    assert len(ds) == 2
    assert ds[0]["input_ids"] == [1, 2, 3, 4]
    assert ds[1]["document_ids"] == [0, 0, 1, 1]


def test_getitem_builds_default_attention_mask_and_labels(tmp_path):
    """``__getitem__`` synthesizes attention_mask (1 where input_ids!=0) and
    labels (input_ids, -100 on padding) when those fields are absent."""
    # Short row → write_memmap zero-pads to seq_len=4.
    write_memmap(tmp_path, [{"input_ids": [5, 6], "position_ids": [0, 1], "document_ids": [0, 0]}], seq_len=4)
    item = MemmapDataset(tmp_path)[0]
    assert item["input_ids"] == [5, 6, 0, 0]
    assert item["attention_mask"] == [1, 1, 0, 0]
    assert item["labels"] == [5, 6, -100, -100]


def test_getitem_coerces_float_index(tmp_path):
    """``__getitem__`` coerces its index to int."""
    write_memmap(tmp_path, [{"input_ids": [1, 1]}, {"input_ids": [2, 2]}], seq_len=2)
    assert MemmapDataset(tmp_path)[1.0]["input_ids"] == [2, 2]  # type: ignore[index]


# ---------------------------------------------------------------------------
# write_memmap row-shaping edges
# ---------------------------------------------------------------------------

def test_write_truncates_row_longer_than_seq_len(tmp_path):
    """A row longer than ``seq_len`` is head-truncated to ``seq_len``."""
    write_memmap(tmp_path, [{"input_ids": [1, 2, 3, 4, 5, 6]}], seq_len=3)
    assert MemmapDataset(tmp_path)[0]["input_ids"] == [1, 2, 3]


def test_write_zero_pads_row_shorter_than_seq_len(tmp_path):
    """A row shorter than ``seq_len`` is zero-padded on the right."""
    write_memmap(tmp_path, [{"input_ids": [9]}], seq_len=4)
    assert MemmapDataset(tmp_path)[0]["input_ids"] == [9, 0, 0, 0]


def test_write_skips_missing_field_value(tmp_path):
    """A row missing a declared field leaves that field zero-filled (the
    ``v is None`` continue), without erroring."""
    write_memmap(tmp_path, [{"input_ids": [1, 2, 3, 4]}], seq_len=4)  # no position/document ids
    item = MemmapDataset(tmp_path)[0]
    assert item["input_ids"] == [1, 2, 3, 4]
    assert item["position_ids"] == [0, 0, 0, 0]
    assert item["document_ids"] == [0, 0, 0, 0]


def test_write_overwrites_existing_dir(tmp_path):
    """Writing twice to the same dir atomically replaces the old data/header."""
    write_memmap(tmp_path, [{"input_ids": [1, 1, 1, 1]}], seq_len=4)
    write_memmap(tmp_path, [{"input_ids": [2, 2]}, {"input_ids": [3, 3]}], seq_len=2)
    ds = MemmapDataset(tmp_path)
    assert len(ds) == 2
    assert ds[0]["input_ids"] == [2, 2]


def test_write_respects_custom_dtypes(tmp_path):
    """Non-default dtypes are honored end to end (offset arithmetic per field)."""
    dtypes = {f: "int32" for f in _FIELDS}
    write_memmap(tmp_path, [{"input_ids": [1, 2], "position_ids": [0, 1], "document_ids": [0, 0]}],
                 seq_len=2, dtypes=dtypes)
    hdr = read_header(tmp_path)
    assert hdr is not None
    assert hdr.dtypes == dtypes
    assert MemmapDataset(tmp_path)[0]["input_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# read_header / missing-file guards
# ---------------------------------------------------------------------------

def test_read_header_returns_none_when_absent(tmp_path):
    """No header sidecar → ``read_header`` returns None (not an error)."""
    assert read_header(tmp_path) is None


def test_dataset_missing_header_raises(tmp_path):
    """Constructing a dataset over a dir with no header raises."""
    with pytest.raises(FileNotFoundError, match="No memmap header"):
        MemmapDataset(tmp_path)


def test_dataset_missing_data_bin_raises(tmp_path):
    """Header present but ``data.bin`` deleted → distinct missing-data error."""
    write_memmap(tmp_path, [{"input_ids": [1, 2, 3, 4]}], seq_len=4)
    (tmp_path / DATA_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="No memmap data"):
        MemmapDataset(tmp_path)
