"""Data core: ByteTokenizer round-trip, CausalLMCollator shapes, SimpleDataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.data.core.collators import CausalLMCollator
from lighttrain.builtin_plugins.data.core.datasets import LineFileTextDataset
from lighttrain.builtin_plugins.data.core.samplers import ShuffleSampler
from lighttrain.builtin_plugins.data.core.tokenizers import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)
from lighttrain.builtin_plugins.data.core._module import SimpleDataModule


def test_byte_tokenizer_round_trip():
    tk = ByteTokenizer(add_bos=True, add_eos=True)
    text = "hello, lighttrain!"
    ids = tk.encode(text)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    decoded = tk.decode(ids)
    assert decoded == text
    assert tk.vocab_size == VOCAB_SIZE
    assert tk.pad_id == PAD_ID
    assert tk.unk_id == UNK_ID


def test_byte_tokenizer_handles_unicode():
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    text = "你好"  # multi-byte
    ids = tk.encode(text)
    assert tk.decode(ids) == text
    assert all(0 <= i < 256 for i in ids)


def test_collator_pads_and_marks_labels(tmp_path: Path):
    tk = ByteTokenizer(add_eos=False)
    samples = [
        {"input_ids": tk.encode("a"), "attention_mask": None, "labels": tk.encode("a")},
        {"input_ids": tk.encode("longer string"), "attention_mask": None, "labels": tk.encode("longer string")},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert batch["input_ids"].shape == (2, len(samples[1]["input_ids"]))
    assert batch["attention_mask"].sum() == sum(len(s["input_ids"]) for s in samples)
    # Pad positions in labels are -100.
    assert (batch["labels"][0, len(samples[0]["input_ids"]):] == -100).all()


def test_line_file_dataset_drops_blank_lines(tmp_path: Path):
    p = tmp_path / "corpus.txt"
    p.write_text("first\n\nsecond\n\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=ByteTokenizer(), max_len=128)
    assert len(ds) == 2


class _FixedTokenizer:
    """Encodes any line to a fixed list of ``n`` distinct token ids."""

    def __init__(self, n: int) -> None:
        self._ids = list(range(n))

    def encode(self, _line: str) -> list[int]:
        return list(self._ids)


def test_chunk_size_larger_than_max_len_fails_loud(tmp_path: Path):
    """chunk_size > max_len would silently drop tokens past max_len → ValueError."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be <= max_len"):
        LineFileTextDataset(
            p, tokenizer=_FixedTokenizer(100), max_len=20, chunk_size=50
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_chunk_size_non_positive_fails_loud(tmp_path: Path, bad: int):
    """chunk_size of 0 or negative is rejected (0 used to silently disable)."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive int"):
        LineFileTextDataset(
            p, tokenizer=_FixedTokenizer(100), max_len=20, chunk_size=bad
        )


def test_chunk_size_within_max_len_covers_all_tokens(tmp_path: Path):
    """chunk_size <= max_len chunks a long doc without dropping any token."""
    p = tmp_path / "corpus.txt"
    p.write_text("anything\n", encoding="utf-8")
    ds = LineFileTextDataset(
        p, tokenizer=_FixedTokenizer(100), max_len=50, chunk_size=20
    )
    covered: set[int] = set()
    for s in ds.samples:
        covered.update(s["input_ids"])
    assert covered == set(range(100))


def test_chunk_size_none_keeps_one_sample_per_line(tmp_path: Path):
    """chunk_size=None keeps the one-line-per-sample default (no chunking)."""
    p = tmp_path / "corpus.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    ds = LineFileTextDataset(p, tokenizer=ByteTokenizer(), max_len=64, chunk_size=None)
    assert len(ds) == 2


def test_simple_data_module_train_loader_yields_batches(tmp_path: Path):
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


def test_shuffle_sampler_is_deterministic_with_seed():
    n = 16
    a = ShuffleSampler(list(range(n)), seed=7)
    b = ShuffleSampler(list(range(n)), seed=7)
    assert list(iter(a)) == list(iter(b))
