"""Mirror tests for the dataset / data-module source area.

These cover behaviors that the collator/tokenizer mirrors do NOT:

* ``LineFileTextDataset`` — blank-line dropping, ``chunk_size`` guards
  (``> max_len`` and non-positive fail loud), full-token coverage when
  chunking, and the ``chunk_size=None`` one-sample-per-line default.
* ``SimpleDataModule`` — the train loader yields correctly-shaped long-dtype
  batches end to end.
* ``ShuffleSampler`` — same seed ⇒ identical permutation (determinism).
* ``PreferenceJsonlDataset`` — registry name, load round-trip, truncation,
  silent tokenizer acceptance, and the shipped fixture used by
  ``dpo_offline.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.data.core._module import SimpleDataModule
from lighttrain.builtin_plugins.data.core.datasets import (
    LineFileTextDataset,
    PreferenceJsonlDataset,
)
from lighttrain.builtin_plugins.data.core.samplers import ShuffleSampler
from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
from lighttrain.registry import get as registry_get


class _FixedTokenizer:
    """Encodes any line to a fixed list of ``n`` distinct token ids."""

    def __init__(self, n: int) -> None:
        self._ids = list(range(n))

    def encode(self, _line: str) -> list[int]:
        return list(self._ids)


# --------------------------------------------------------------------------- #
# LineFileTextDataset                                                         #
# --------------------------------------------------------------------------- #


def test_line_file_dataset_drops_blank_lines(tmp_path: Path) -> None:
    """Blank lines are dropped — a file with 2 non-blank lines yields 2 samples."""
    p = tmp_path / "corpus.txt"
    p.write_text("first\n\nsecond\n\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=ByteTokenizer(), max_len=128)
    assert len(ds) == 2


def test_line_file_dataset_chunk_size_larger_than_max_len_fails_loud(
    tmp_path: Path,
) -> None:
    """``chunk_size > max_len`` would silently drop tokens past ``max_len`` →
    ValueError at construction."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be <= max_len"):
        LineFileTextDataset(
            p, tokenizer=_FixedTokenizer(100), max_len=20, chunk_size=50
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_line_file_dataset_chunk_size_non_positive_fails_loud(
    tmp_path: Path, bad: int
) -> None:
    """``chunk_size`` of 0 or negative is rejected (0 used to silently disable)."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive int"):
        LineFileTextDataset(
            p, tokenizer=_FixedTokenizer(100), max_len=20, chunk_size=bad
        )


def test_line_file_dataset_chunk_size_within_max_len_covers_all_tokens(
    tmp_path: Path,
) -> None:
    """``chunk_size <= max_len`` chunks a long doc without dropping any token."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    ds = LineFileTextDataset(
        p, tokenizer=_FixedTokenizer(100), max_len=50, chunk_size=20
    )
    covered: set[int] = set()
    for s in ds.samples:
        covered.update(s["input_ids"])
    assert covered == set(range(100))


def test_line_file_dataset_chunk_size_none_keeps_one_sample_per_line(
    tmp_path: Path,
) -> None:
    """``chunk_size=None`` keeps the one-line-per-sample default (no chunking)."""
    p = tmp_path / "corpus.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=ByteTokenizer(), max_len=64, chunk_size=None)
    assert len(ds) == 2


# --------------------------------------------------------------------------- #
# SimpleDataModule                                                            #
# --------------------------------------------------------------------------- #


def test_simple_data_module_train_loader_yields_long_dtype_batches(
    tmp_path: Path,
) -> None:
    """The wired-up train loader yields a dict batch with ``batch_size`` rows of
    long-dtype ``input_ids``."""
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(f"line-{i}" for i in range(16)) + "\n", encoding="utf-8")
    dm = SimpleDataModule(
        dataset={"name": "line_file_text", "path": str(p), "max_len": 64},
        tokenizer={"name": "byte"},
        collator={"name": "causal_lm", "max_len": 64},
        sampler={"name": "shuffle", "seed": 0},
        batch_size=4,
    )
    loader = dm.train_loader()
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].dtype == torch.long


# --------------------------------------------------------------------------- #
# ShuffleSampler determinism                                                  #
# --------------------------------------------------------------------------- #


def test_shuffle_sampler_same_seed_is_deterministic() -> None:
    """Two ``ShuffleSampler``s with the same seed produce identical orders."""
    n = 16
    a = ShuffleSampler(list(range(n)), seed=7)
    b = ShuffleSampler(list(range(n)), seed=7)
    assert list(iter(a)) == list(iter(b))


# --------------------------------------------------------------------------- #
# PreferenceJsonlDataset                                                      #
# --------------------------------------------------------------------------- #


def test_preference_jsonl_registered_under_preference_jsonl() -> None:
    """``PreferenceJsonlDataset`` is registry ``('dataset', 'preference_jsonl')``."""
    assert registry_get("dataset", "preference_jsonl") is PreferenceJsonlDataset


def test_preference_jsonl_loads_rows_with_chosen_rejected(tmp_path: Path) -> None:
    """Rows round-trip: ids and chosen/rejected token lists survive load."""
    data = [
        {"id": "a", "chosen_input_ids": [1, 2, 3], "chosen_labels": [1, 2, 3],
         "rejected_input_ids": [4, 5], "rejected_labels": [4, 5]},
        {"id": "b", "chosen_input_ids": [6, 7], "chosen_labels": [6, 7],
         "rejected_input_ids": [8, 9, 10], "rejected_labels": [8, 9, 10]},
    ]
    f = tmp_path / "pref.jsonl"
    f.write_text("\n".join(json.dumps(d) for d in data))
    ds = PreferenceJsonlDataset(f)
    assert len(ds) == 2
    s = ds[0]
    assert s["id"] == "a"
    assert s["chosen_input_ids"] == [1, 2, 3]
    assert s["rejected_input_ids"] == [4, 5]


def test_preference_jsonl_truncates_to_max_len(tmp_path: Path) -> None:
    """``max_len`` truncates both chosen and rejected token lists."""
    row = {"id": "x", "chosen_input_ids": list(range(20)), "chosen_labels": list(range(20)),
           "rejected_input_ids": list(range(20)), "rejected_labels": list(range(20))}
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row))
    ds = PreferenceJsonlDataset(f, max_len=5)
    assert len(ds[0]["chosen_input_ids"]) == 5


def test_preference_jsonl_silently_accepts_injected_tokenizer(tmp_path: Path) -> None:
    """A ``tokenizer`` kwarg (injected by ``SimpleDataModule._resolve_base``) is
    silently accepted even though the dataset is pre-tokenized."""
    row = {"id": "x", "chosen_input_ids": [1], "chosen_labels": [1],
           "rejected_input_ids": [2], "rejected_labels": [2]}
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row))
    ds = PreferenceJsonlDataset(f, tokenizer="any_value_ignored")
    assert len(ds) == 1


def test_preference_jsonl_shipped_fixture_loads() -> None:
    """Smoke-test the fixture used in dpo_offline.yaml (cwd-relative; pytest
    runs from repo root). Self-skips if the fixture is absent."""
    fixture = Path("tests/fixtures/tiny_preference.jsonl")
    if not fixture.exists():
        pytest.skip("fixture not found from current working directory")
    ds = PreferenceJsonlDataset(fixture)
    assert len(ds) >= 4
    s = ds[0]
    assert "chosen_input_ids" in s and "rejected_input_ids" in s
