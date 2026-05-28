"""Hermetic processor tests — image / audio / text basics."""

from __future__ import annotations

import numpy as np
import pytest

from lighttrain.data.processors.audio import MelSpectrogramProcessor
from lighttrain.data.processors.image import SimpleImageProcessor
from lighttrain.data.processors.text import ChatTemplateProcessor
from lighttrain.data.core.tokenizers import ByteTokenizer


def test_simple_image_single_array():
    proc = SimpleImageProcessor(size=32)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 255
    out = proc(img)
    assert out["modality"] == "image"
    arr = out["pixel_values"]
    assert arr.shape == (3, 32, 32)
    assert arr.dtype == np.float32
    # half white / half black after resize → mean roughly between mean of mean,std
    assert -2.0 < arr.mean() < 2.0


def test_simple_image_batch_list():
    proc = SimpleImageProcessor(size=16)
    imgs = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    out = proc(imgs)
    assert out["pixel_values"].shape == (3, 3, 16, 16)


def test_mel_spectrogram_shapes():
    proc = MelSpectrogramProcessor(
        sample_rate=16_000, n_fft=400, hop_length=160, n_mels=40
    )
    wav = np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000).astype(np.float32)
    out = proc(wav)
    assert out["modality"] == "audio"
    feats = out["audio_features"]
    assert feats.ndim == 2
    assert feats.shape[0] == 40  # n_mels
    assert feats.shape[1] > 0


def test_mel_spectrogram_finite():
    proc = MelSpectrogramProcessor(n_mels=20)
    wav = np.random.RandomState(0).randn(8000).astype(np.float32)
    out = proc(wav)
    assert np.isfinite(out["audio_features"]).all()


def test_chat_template_processor_response_only_mask():
    tk = ByteTokenizer()
    proc = ChatTemplateProcessor(tokenizer=tk, response_only_mask=True)
    out = proc([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ])
    assert "input_ids" in out
    assert "labels" in out
    assert out["modality"] == "text"
    # At least *some* labels are masked (-100) and *some* survive.
    masked = [x for x in out["labels"] if x == -100]
    kept = [x for x in out["labels"] if x != -100]
    assert masked
    assert kept


def test_chat_template_processor_no_mask_keeps_everything():
    tk = ByteTokenizer()
    proc = ChatTemplateProcessor(tokenizer=tk, response_only_mask=False)
    out = proc([
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ])
    assert all(x != -100 for x in out["labels"])
