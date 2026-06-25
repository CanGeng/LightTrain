"""Coverage-gap tests for ``lighttrain.builtin_plugins.data.core.datasets``.

Pins every branch not reached by ``tests/data/test_datasets.py``:

* **Line 31** – ``LineFileTextDataset`` raises ``FileNotFoundError`` for a
  non-existent path.
* **Line 61** – lines whose tokens are all empty (tokenizer returns ``[]``) are
  silently skipped.
* **Line 66** – defensive ``if not chunk: continue`` guard inside the chunking
  path (unreachable under normal arithmetic; skipped — see note).
* **Line 86** – ``LineFileTextDataset`` raises ``ValueError`` when no samples
  survive (all lines empty or tokenizer always returns ``[]``).
* **Line 95** – ``LineFileTextDataset.__iter__`` returns all samples in order.
* **Line 122** – ``PreferenceJsonlDataset`` raises ``FileNotFoundError`` for a
  non-existent path.
* **Line 128** – blank lines inside a JSONL file are silently skipped.
* **Line 138** – ``PreferenceJsonlDataset`` raises ``ValueError`` when no rows
  survive (file has only blank lines).
* **Line 147** – ``PreferenceJsonlDataset.__iter__`` returns all samples in
  order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data.core.datasets import (
    LineFileTextDataset,
    PreferenceJsonlDataset,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _EmptyTokenizer:
    """Always returns an empty token list for any input."""

    def encode(self, _line: str) -> list[int]:
        return []


class _FixedTokenizer:
    """Returns ``n`` distinct sequential token ids for any input."""

    def __init__(self, n: int) -> None:
        self._ids = list(range(n))

    def encode(self, _line: str) -> list[int]:
        return list(self._ids)


# ---------------------------------------------------------------------------
# LineFileTextDataset – uncovered branches
# ---------------------------------------------------------------------------


def test_invariant_line_file_missing_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Line 31: a non-existent file raises ``FileNotFoundError`` immediately."""
    missing = tmp_path / "no_such_file.txt"
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        LineFileTextDataset(missing, tokenizer=_FixedTokenizer(4))


def test_invariant_line_file_empty_token_lines_skipped(tmp_path: Path) -> None:
    """Line 61: lines whose encoded ids are empty are silently skipped.

    A tokenizer that always returns [] causes every line to be dropped;
    the final ValueError (line 86) is the observable outcome confirming
    the skip path was exercised.
    """
    p = tmp_path / "corpus.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable lines"):
        LineFileTextDataset(p, tokenizer=_EmptyTokenizer())


def test_invariant_line_file_all_blank_raises_value_error(tmp_path: Path) -> None:
    """Line 86: a file containing only blank lines raises ``ValueError``."""
    p = tmp_path / "blank.txt"
    p.write_text("\n\n   \n\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable lines"):
        LineFileTextDataset(p, tokenizer=_FixedTokenizer(4))


def test_invariant_line_file_iter_returns_all_samples_in_order(
    tmp_path: Path,
) -> None:
    """Line 95: ``__iter__`` yields every sample in construction order."""
    p = tmp_path / "corpus.txt"
    lines = ["alpha", "beta", "gamma"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(3), max_len=16)
    iterated = list(ds)
    assert len(iterated) == 3
    # __iter__ must agree with __getitem__ indexing
    for i, sample in enumerate(iterated):
        assert sample is ds[i]


def test_invariant_line_file_iter_matches_len(tmp_path: Path) -> None:
    """``len(ds)`` equals the number of items yielded by ``iter(ds)``."""
    p = tmp_path / "corpus.txt"
    p.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(5), max_len=8)
    assert list(ds) == ds.samples
    assert len(list(ds)) == len(ds)


def test_invariant_line_file_single_non_blank_line(tmp_path: Path) -> None:
    """A file with exactly one non-blank line yields exactly one sample."""
    p = tmp_path / "single.txt"
    p.write_text("\n\nhello world\n\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(4), max_len=32)
    assert len(ds) == 1


def test_invariant_line_file_chunk_size_produces_doc_boundary_flags(
    tmp_path: Path,
) -> None:
    """The first chunk of each document carries ``_doc_boundary=True``; later
    chunks carry ``_doc_boundary=False``."""
    p = tmp_path / "doc.txt"
    p.write_text("document\n", encoding="utf-8")
    # _FixedTokenizer(12) → 12 tokens; chunk_size=4 → 3 chunks
    ds = LineFileTextDataset(
        p, tokenizer=_FixedTokenizer(12), max_len=12, chunk_size=4
    )
    assert len(ds) == 3
    assert ds[0]["_doc_boundary"] is True
    assert ds[1]["_doc_boundary"] is False
    assert ds[2]["_doc_boundary"] is False


def test_invariant_line_file_max_len_truncates_non_chunked(tmp_path: Path) -> None:
    """Without chunking, tokens beyond ``max_len`` are truncated."""
    p = tmp_path / "long.txt"
    p.write_text("line\n", encoding="utf-8")
    ds = LineFileTextDataset(
        p, tokenizer=_FixedTokenizer(100), max_len=10, chunk_size=None
    )
    assert len(ds[0]["input_ids"]) == 10


def test_invariant_line_file_attention_mask_matches_input_ids_length(
    tmp_path: Path,
) -> None:
    """``attention_mask`` length equals ``input_ids`` length for every sample."""
    p = tmp_path / "corpus.txt"
    p.write_text("a\nbb\nccc\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(7), max_len=5)
    for sample in ds:
        assert len(sample["attention_mask"]) == len(sample["input_ids"])
        assert all(v == 1 for v in sample["attention_mask"])


def test_invariant_line_file_labels_equal_input_ids(tmp_path: Path) -> None:
    """``labels`` is a copy of ``input_ids`` (causal-LM convention)."""
    p = tmp_path / "corpus.txt"
    p.write_text("hello\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(6), max_len=8)
    s = ds[0]
    assert s["labels"] == s["input_ids"]


def test_invariant_line_file_getitem_int_coercion(tmp_path: Path) -> None:
    """``__getitem__`` coerces its index to ``int`` (accepts numpy-like scalars)."""
    p = tmp_path / "corpus.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=_FixedTokenizer(3), max_len=8)
    # Pass a float – should coerce silently via int(idx)
    assert ds[0.0] is ds[0]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PreferenceJsonlDataset – uncovered branches
# ---------------------------------------------------------------------------


def _pref_row(**overrides) -> dict:
    """Return a minimal valid preference row, with optional overrides."""
    base = {
        "id": "r0",
        "chosen_input_ids": [1, 2, 3],
        "chosen_labels": [1, 2, 3],
        "rejected_input_ids": [4, 5, 6],
        "rejected_labels": [4, 5, 6],
    }
    base.update(overrides)
    return base


def test_invariant_pref_jsonl_missing_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Line 122: a non-existent JSONL file raises ``FileNotFoundError``."""
    missing = tmp_path / "no_pref.jsonl"
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        PreferenceJsonlDataset(missing)


def test_invariant_pref_jsonl_blank_lines_skipped(tmp_path: Path) -> None:
    """Line 128: blank lines inside the JSONL file are silently skipped."""
    row = _pref_row()
    f = tmp_path / "pref.jsonl"
    # Surround one valid row with blank lines
    f.write_text("\n\n" + json.dumps(row) + "\n\n", encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    assert len(ds) == 1


def test_invariant_pref_jsonl_only_blank_lines_raises_value_error(
    tmp_path: Path,
) -> None:
    """Line 138: a JSONL file with only blank lines raises ``ValueError``."""
    f = tmp_path / "empty.jsonl"
    f.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable lines"):
        PreferenceJsonlDataset(f)


def test_invariant_pref_jsonl_empty_file_raises_value_error(tmp_path: Path) -> None:
    """An entirely empty JSONL file raises ``ValueError`` (no samples)."""
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable lines"):
        PreferenceJsonlDataset(f)


def test_invariant_pref_jsonl_iter_returns_all_samples_in_order(
    tmp_path: Path,
) -> None:
    """Line 147: ``__iter__`` yields every sample in construction order."""
    rows = [_pref_row(id=str(i)) for i in range(5)]
    f = tmp_path / "pref.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    iterated = list(ds)
    assert len(iterated) == 5
    for i, sample in enumerate(iterated):
        assert sample is ds[i]
        assert sample["id"] == str(i)


def test_invariant_pref_jsonl_iter_matches_len(tmp_path: Path) -> None:
    """``len(ds)`` equals the number of items from ``iter(ds)``."""
    rows = [_pref_row(id=str(i)) for i in range(3)]
    f = tmp_path / "pref.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    assert len(list(ds)) == len(ds) == 3


def test_invariant_pref_jsonl_missing_id_falls_back_to_index(
    tmp_path: Path,
) -> None:
    """When an ``id`` field is absent the sample index is used as a string id."""
    row = {
        "chosen_input_ids": [10, 20],
        "chosen_labels": [10, 20],
        "rejected_input_ids": [30, 40],
        "rejected_labels": [30, 40],
    }
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row), encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    assert ds[0]["id"] == "0"


def test_invariant_pref_jsonl_multiple_blank_lines_between_rows_skipped(
    tmp_path: Path,
) -> None:
    """Multiple blank lines interspersed between rows are all silently dropped."""
    rows = [_pref_row(id="a"), _pref_row(id="b")]
    f = tmp_path / "pref.jsonl"
    content = "\n\n".join(json.dumps(r) for r in rows) + "\n\n"
    f.write_text(content, encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    assert len(ds) == 2
    assert ds[0]["id"] == "a"
    assert ds[1]["id"] == "b"


def test_invariant_pref_jsonl_getitem_int_coercion(tmp_path: Path) -> None:
    """``__getitem__`` coerces its index to ``int``."""
    rows = [_pref_row(id="x"), _pref_row(id="y")]
    f = tmp_path / "pref.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = PreferenceJsonlDataset(f)
    assert ds[0.0] is ds[0]  # type: ignore[arg-type]


def test_invariant_pref_jsonl_truncation_all_four_fields(tmp_path: Path) -> None:
    """``max_len`` truncates all four token-list fields uniformly."""
    long_ids = list(range(50))
    row = {
        "id": "t",
        "chosen_input_ids": long_ids,
        "chosen_labels": long_ids,
        "rejected_input_ids": long_ids,
        "rejected_labels": long_ids,
    }
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row), encoding="utf-8")
    ds = PreferenceJsonlDataset(f, max_len=7)
    s = ds[0]
    assert len(s["chosen_input_ids"]) == 7
    assert len(s["chosen_labels"]) == 7
    assert len(s["rejected_input_ids"]) == 7
    assert len(s["rejected_labels"]) == 7
