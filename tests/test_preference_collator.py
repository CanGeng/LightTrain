"""Tests for PreferenceCollator (registered as ('collator', 'preference'))
and PreferenceJsonlDataset (registered as ('dataset', 'preference_jsonl')).
"""
from __future__ import annotations

import json
import pytest
import torch

from lighttrain.builtin_plugins.data.core.collators import PreferenceCollator
from lighttrain.builtin_plugins.data.core.datasets import PreferenceJsonlDataset
from lighttrain.registry import get as registry_get


# ---------------------------------------------------------------------------
# PreferenceCollator
# ---------------------------------------------------------------------------

def _make_sample(chosen_len: int, rejected_len: int):
    return {
        "chosen_input_ids": list(range(chosen_len)),
        "chosen_labels": list(range(chosen_len)),
        "rejected_input_ids": list(range(rejected_len)),
        "rejected_labels": list(range(rejected_len)),
    }


def test_preference_collator_registered():
    cls = registry_get("collator", "preference")
    assert cls is PreferenceCollator


def test_output_keys():
    collator = PreferenceCollator(pad_id=0, max_len=16)
    batch = collator([_make_sample(5, 7)])
    expected = {
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
    }
    assert set(batch.keys()) == expected


def test_output_shapes():
    collator = PreferenceCollator(pad_id=0, max_len=16)
    samples = [_make_sample(5, 7), _make_sample(3, 10)]
    batch = collator(samples)
    B = 2
    assert batch["chosen_input_ids"].shape == (B, 5)
    assert batch["chosen_attention_mask"].shape == (B, 5)
    assert batch["chosen_labels"].shape == (B, 5)
    assert batch["rejected_input_ids"].shape == (B, 10)
    assert batch["rejected_attention_mask"].shape == (B, 10)
    assert batch["rejected_labels"].shape == (B, 10)


def test_padding_and_attention_mask():
    collator = PreferenceCollator(pad_id=99, max_len=16, ignore_index=-100)
    samples = [_make_sample(3, 3), _make_sample(5, 3)]
    batch = collator(samples)
    # Row 0 chosen: 3 tokens out of 5 → [1,1,1,0,0]
    assert batch["chosen_attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["chosen_attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    assert batch["chosen_input_ids"][0, 3].item() == 99
    assert batch["chosen_labels"][0, 3].item() == -100


def test_truncation():
    collator = PreferenceCollator(pad_id=0, max_len=4)
    samples = [_make_sample(10, 10)]
    batch = collator(samples)
    assert batch["chosen_input_ids"].shape == (1, 4)
    assert batch["rejected_input_ids"].shape == (1, 4)


def test_empty_batch_raises():
    collator = PreferenceCollator(pad_id=0)
    with pytest.raises(ValueError, match="Empty batch"):
        collator([])


# ---------------------------------------------------------------------------
# PreferenceJsonlDataset
# ---------------------------------------------------------------------------

def test_preference_jsonl_registered():
    cls = registry_get("dataset", "preference_jsonl")
    assert cls is PreferenceJsonlDataset


def test_preference_jsonl_loads(tmp_path):
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


def test_preference_jsonl_truncation(tmp_path):
    row = {"id": "x", "chosen_input_ids": list(range(20)), "chosen_labels": list(range(20)),
           "rejected_input_ids": list(range(20)), "rejected_labels": list(range(20))}
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row))
    ds = PreferenceJsonlDataset(f, max_len=5)
    assert len(ds[0]["chosen_input_ids"]) == 5


def test_preference_jsonl_accepts_tokenizer(tmp_path):
    row = {"id": "x", "chosen_input_ids": [1], "chosen_labels": [1],
           "rejected_input_ids": [2], "rejected_labels": [2]}
    f = tmp_path / "pref.jsonl"
    f.write_text(json.dumps(row))
    # tokenizer injected by SimpleDataModule / _resolve_base; must be silently accepted
    ds = PreferenceJsonlDataset(f, tokenizer="any_value_ignored")
    assert len(ds) == 1


def test_preference_jsonl_fixture_loads():
    """Smoke-test the fixture used in dpo_offline.yaml."""
    from pathlib import Path
    fixture = Path("tests/fixtures/tiny_preference.jsonl")
    if not fixture.exists():
        pytest.skip("fixture not found from current working directory")
    ds = PreferenceJsonlDataset(fixture)
    assert len(ds) >= 4
    s = ds[0]
    assert "chosen_input_ids" in s and "rejected_input_ids" in s


# ---------------------------------------------------------------------------
# aux.* passthrough (Fix 3 / REVIEW_ROUND3 finding #3)
# ---------------------------------------------------------------------------

def test_aux_scalar_passthrough():
    """aux.ref.* scalar logprobs must survive PreferenceCollator."""
    import torch
    samples = [
        {
            "chosen_input_ids": [1, 2, 3],
            "chosen_labels": [1, 2, 3],
            "rejected_input_ids": [4, 5],
            "rejected_labels": [4, 5],
            "aux.ref.chosen_logprobs": torch.tensor(-1.5),
            "aux.ref.rejected_logprobs": torch.tensor(-2.0),
        },
        {
            "chosen_input_ids": [6, 7],
            "chosen_labels": [6, 7],
            "rejected_input_ids": [8, 9, 10],
            "rejected_labels": [8, 9, 10],
            "aux.ref.chosen_logprobs": torch.tensor(-1.2),
            "aux.ref.rejected_logprobs": torch.tensor(-1.8),
        },
    ]
    collator = PreferenceCollator(pad_id=0, max_len=16)
    batch = collator(samples)
    assert "aux.ref.chosen_logprobs" in batch, "scalar aux key must be in output batch"
    assert "aux.ref.rejected_logprobs" in batch
    assert batch["aux.ref.chosen_logprobs"].shape == (2,)
    assert batch["aux.ref.rejected_logprobs"].shape == (2,)


def test_aux_token_level_passthrough():
    """aux keys with token-level tensors (B, T) must also survive collation."""
    import torch
    T = 4
    samples = [
        {
            "chosen_input_ids": [1, 2],
            "chosen_labels": [1, 2],
            "rejected_input_ids": [3, 4],
            "rejected_labels": [3, 4],
            "aux.ref.token_logps": torch.zeros(T),
        },
        {
            "chosen_input_ids": [5, 6],
            "chosen_labels": [5, 6],
            "rejected_input_ids": [7, 8],
            "rejected_labels": [7, 8],
            "aux.ref.token_logps": torch.ones(T),
        },
    ]
    collator = PreferenceCollator(pad_id=0, max_len=16)
    batch = collator(samples)
    assert "aux.ref.token_logps" in batch
    assert batch["aux.ref.token_logps"].shape == (2, T)
