"""Data core: ByteTokenizer round-trip, CausalLMCollator shapes, SimpleDataModule."""

from __future__ import annotations

from pathlib import Path

import torch

from lighttrain.data.core.collators import CausalLMCollator
from lighttrain.data.core.datasets import LineFileTextDataset
from lighttrain.data.core.samplers import ShuffleSampler
from lighttrain.data.core.tokenizers import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)
from lighttrain.data.core._module import SimpleDataModule


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
