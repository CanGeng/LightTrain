"""MultiModalCollator tests — text-only path + image path + masking."""

from __future__ import annotations

import numpy as np
import torch

from lighttrain.builtin_plugins.data.collators.multimodal import MultiModalCollator


def test_multimodal_collator_text_only_path_matches_causal_lm():
    coll = MultiModalCollator(pad_id=0, max_len=8)
    samples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8]},
    ]
    out = coll(samples)
    assert "modality_inputs" not in out
    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].sum().item() == 3 + 5


def test_multimodal_collator_image_padding_and_mask():
    coll = MultiModalCollator(pad_id=0, max_len=8, max_images=2)
    img1 = np.zeros((3, 4, 4), dtype=np.float32)
    img2 = np.ones((3, 4, 4), dtype=np.float32)
    samples = [
        {
            "input_ids": [1, 2, 3],
            "modality_inputs": {"image": img1},
        },
        {
            "input_ids": [4, 5],
            "modality_inputs": {"image": [img1, img2]},
        },
    ]
    out = coll(samples)
    assert "modality_inputs" in out
    mi = out["modality_inputs"]
    assert mi["image"].shape == (2, 2, 3, 4, 4)
    assert torch.equal(mi["image_mask"], torch.tensor([[1, 0], [1, 1]], dtype=torch.long))


def test_multimodal_collator_audio_pads_time_axis():
    coll = MultiModalCollator(pad_id=0, max_len=8, max_audios=1)
    a1 = np.random.randn(20, 50).astype(np.float32)  # (n_mels, T)
    a2 = np.random.randn(20, 80).astype(np.float32)
    samples = [
        {"input_ids": [1, 2], "modality_inputs": {"audio": a1}},
        {"input_ids": [3, 4], "modality_inputs": {"audio": a2}},
    ]
    out = coll(samples)
    audio = out["modality_inputs"]["audio"]
    assert audio.shape == (2, 1, 20, 80)
    mask = out["modality_inputs"]["audio_mask"]
    assert mask[0, 0, 50:].sum().item() == 0
    assert mask[1, 0, :].sum().item() == 80
