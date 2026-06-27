"""Edge-case and coverage tests for
``lighttrain.builtin_plugins.data.collators.multimodal``.

Pins / coverage targets
-----------------------
* ``_stack_pad`` — empty array list (lines 48-49), shape-mismatch branch
  (lines 56-58), normal single-element path (lines 60-62).
* ``MultiModalCollator.__call__`` — empty-batch ValueError (line 98).
* Image path — 4-D ndarray auto-squeeze (line 116), max_images cap (line 120),
  per-sample image-shape mismatch raises (line 130).
* Audio path — 3-D ndarray auto-squeeze (line 147), multi-audio cap
  (line 152), variable time axis padding + mask (lines 163-165).
* Video path — 5-D ndarray auto-squeeze (line 178-179), max_videos cap
  (line 183), video shape mismatch raises (line 191-193), tensor/mask
  values (lines 195-197).
* Pure-text delegation — output has no ``modality_inputs`` key (line 101).
* Registry: ``MultiModalCollator`` registered as ``('collator', 'multimodal')``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch

from lighttrain.builtin_plugins.data.collators.multimodal import (
    MultiModalCollator,
    _stack_pad,
)

# ---------------------------------------------------------------------------
# _stack_pad — unit tests (lines 44-63)
# ---------------------------------------------------------------------------


def test_stack_pad_empty_list_returns_zero_arrays():
    """Empty input to ``_stack_pad`` returns (0,) float32 + (0, 0) int64
    sentinel arrays (lines 48-49).

    Closed form: no arrays → zeros with shapes (0,) and (0, 0).
    """
    out, mask = _stack_pad([])
    assert out.shape == (0,)
    assert out.dtype == np.float32
    assert mask.shape == (0, 0)
    assert mask.dtype == np.int64


def test_stack_pad_single_array_no_padding_needed():
    """Single 2-D array passes through with mask all-ones (lines 50-62).

    Setup: one (2, 5) array of ones.
    Expected: out shape (1, 2, 5); mask shape (1, 5) all-ones.
    """
    arr = np.ones((2, 5), dtype=np.float32)
    out, mask = _stack_pad([arr])
    assert out.shape == (1, 2, 5)
    assert mask.shape == (1, 5)
    assert mask[0].sum() == 5
    assert (out[0] == 1.0).all()


def test_stack_pad_pads_last_axis_to_max_length():
    """Two arrays with different last-dim lengths: shorter gets zero-padded
    and its mask marks only real positions as 1 (lines 53-62).

    Setup: arr0=(1, 3), arr1=(1, 7).
    Expected: out shape (2, 1, 7); mask[0, 3:] == 0; mask[1, :] == 1.
    """
    a0 = np.ones((1, 3), dtype=np.float32)
    a1 = np.ones((1, 7), dtype=np.float32)
    out, mask = _stack_pad([a0, a1])
    assert out.shape == (2, 1, 7)
    assert mask.shape == (2, 7)
    assert mask[0, :3].sum() == 3
    assert mask[0, 3:].sum() == 0
    assert mask[1, :].sum() == 7
    # Padding positions are exactly 0.0
    assert (out[0, 0, 3:] == 0.0).all()


def test_stack_pad_uses_custom_pad_value():
    """``pad_value`` parameter fills the padded positions (lines 53).

    Setup: arr0=(1, 2), arr1=(1, 4); ``pad_value=-1.0``.
    Expected: out[0, 0, 2:] == -1.0.
    """
    a0 = np.zeros((1, 2), dtype=np.float32)
    a1 = np.zeros((1, 4), dtype=np.float32)
    out, _ = _stack_pad([a0, a1], pad_value=-1.0)
    assert (out[0, 0, 2:] == -1.0).all()


def test_stack_pad_shape_mismatch_raises_value_error():
    """Arrays with mismatched head-shapes raise ``ValueError`` (lines 56-58).

    Setup: arr0=(2, 5), arr1=(3, 5) — head shapes differ.
    Expected: ``ValueError`` mentioning "mismatch".
    """
    a0 = np.ones((2, 5), dtype=np.float32)
    a1 = np.ones((3, 5), dtype=np.float32)  # head_shape mismatch
    with pytest.raises(ValueError, match="mismatch"):
        _stack_pad([a0, a1])


# ---------------------------------------------------------------------------
# MultiModalCollator — constructor + pure-text delegation
# ---------------------------------------------------------------------------


def test_invariant_empty_batch_raises_value_error():
    """Calling collator with an empty list raises ``ValueError`` (line 98).

    Goal: catch the regression where an empty batch silently produced a
    shape-(0, 0) tensor and propagated into the model.
    """
    coll = MultiModalCollator(pad_id=0, max_len=8)
    with pytest.raises(ValueError, match="empty"):
        coll([])


def test_invariant_pure_text_batch_no_modality_inputs_key():
    """Samples without ``modality_inputs`` delegate to ``CausalLMCollator``
    and the output dict MUST NOT contain ``modality_inputs`` (line 101).

    Setup: two text-only samples.
    Expected: ``"modality_inputs"`` absent from output.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    out = coll(samples)
    assert "modality_inputs" not in out
    assert out["input_ids"].shape == (2, 3)


def test_invariant_constructor_stores_params():
    """Constructor stores all parameters as int attributes (lines 84-93)."""
    coll = MultiModalCollator(
        pad_id=7, max_len=512, label_ignore=-50,
        max_images=3, max_audios=4, max_videos=2,
    )
    assert coll.pad_id == 7
    assert coll.max_len == 512
    assert coll.label_ignore == -50
    assert coll.max_images == 3
    assert coll.max_audios == 4
    assert coll.max_videos == 2


def test_invariant_registry_name():
    """``MultiModalCollator`` is registered as ``('collator', 'multimodal')``."""
    from lighttrain.registry import get
    assert get("collator", "multimodal") is MultiModalCollator


# ---------------------------------------------------------------------------
# Image path (lines 107-136)
# ---------------------------------------------------------------------------


def test_invariant_image_tensor_shape_single_image_per_sample():
    """Single 3-D image (C, H, W) per sample is auto-unsqueezed to (1, C, H, W)
    (line 116) and collated to (B, max_n, C, H, W).

    Setup: 2 samples, each with one 3×4×4 image.
    Expected: output image shape (2, 1, 3, 4, 4); mask all-ones.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=1)
    img = np.ones((3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1, 2], "modality_inputs": {"image": img}},
        {"input_ids": [3, 4], "modality_inputs": {"image": img}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["image"].shape == (2, 1, 3, 4, 4)
    assert mi["image_mask"].tolist() == [[1], [1]]


def test_invariant_image_4d_input_treated_as_multiple_images():
    """A 4-D numpy array (N_img, C, H, W) is treated as N_img images
    per sample — no extra unsqueeze is applied (line 115-117).

    Setup: sample 0 has 2 images (4-D), sample 1 has 1 image.
    Expected: output shape (2, 2, C, H, W); sample 1 slot 2 masked 0.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=2)
    two_imgs = np.ones((2, 3, 4, 4), dtype=np.float32)
    one_img = np.zeros((1, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1, 2], "modality_inputs": {"image": two_imgs}},
        {"input_ids": [3, 4], "modality_inputs": {"image": one_img}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["image"].shape == (2, 2, 3, 4, 4)
    # Sample 1 has only 1 image — slot 0 is 1, slot 1 is 0
    assert mi["image_mask"][1, 0].item() == 1
    assert mi["image_mask"][1, 1].item() == 0


def test_invariant_max_images_caps_image_count():
    """When a sample has more images than ``max_images``, extra images are
    silently discarded (line 120: ``min(self.max_images, ...)``, line 128:
    ``imgs[:max_n]``).

    Setup: one sample with 3 images; ``max_images=2``.
    Expected: output image shape (1, 2, C, H, W).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=2)
    three_imgs = np.ones((3, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1, 2], "modality_inputs": {"image": three_imgs}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["image"].shape == (1, 2, 3, 4, 4)


def test_invariant_image_shape_mismatch_raises_value_error():
    """Images within the same batch that have different (C, H, W) shapes
    raise ``ValueError`` (lines 130-132).

    Setup: sample 0 has image (3, 4, 4); sample 1 has image (3, 8, 8).
    Expected: ``ValueError`` mentioning "mismatch".
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=1)
    img_small = np.zeros((3, 4, 4), dtype=np.float32)
    img_large = np.zeros((3, 8, 8), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1, 2], "modality_inputs": {"image": img_small}},
        {"input_ids": [3, 4], "modality_inputs": {"image": img_large}},
    ]
    with pytest.raises(ValueError, match="mismatch"):
        coll(samples)


def test_invariant_image_tensor_dtype_is_float32():
    """Output image tensor dtype is float32 (from np.zeros float32 init)."""
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=1)
    img = np.ones((3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1, 2], "modality_inputs": {"image": img}}]
    out = coll(samples)
    assert out["modality_inputs"]["image"].dtype == torch.float32


def test_invariant_sample_without_image_gets_zero_slot_and_zero_mask():
    """A sample missing the 'image' key in modality_inputs has its image
    slots zeroed out and mask zeroed (lines 111-113).

    Setup: sample 0 has image; sample 1 has no image.
    Expected: image_mask[1] == 0; image slot [1] is all zeros.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_images=1)
    img = np.ones((3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1, 2], "modality_inputs": {"image": img}},
        {"input_ids": [3, 4], "modality_inputs": {}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["image_mask"][1, 0].item() == 0
    assert mi["image"][1].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Audio path (lines 139-167)
# ---------------------------------------------------------------------------


def test_invariant_audio_2d_input_auto_unsqueezed():
    """A 2-D audio array (n_mels, T) is treated as a single audio clip
    (line 147: ``arr = arr[None]``).

    Setup: 1 sample with 2-D audio (20, 50).
    Expected: output shape (1, 1, 20, 50).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_audios=1)
    audio = np.ones((20, 50), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"audio": audio}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["audio"].shape == (1, 1, 20, 50)
    assert mi["audio_mask"].shape == (1, 1, 50)
    assert mi["audio_mask"][0, 0, :].sum().item() == 50


def test_invariant_audio_time_axis_padded_to_max():
    """Audio clips of different T lengths are padded on the time axis
    (last dim) to max_T (lines 156-165).

    Setup: 2 samples with T=30 and T=70; n_mels=16.
    Expected: output (2, 1, 16, 70); mask[0, 0, 30:] == 0.
    """
    torch.manual_seed(0)
    coll = MultiModalCollator(pad_id=0, max_len=16, max_audios=1)
    a0 = np.ones((16, 30), dtype=np.float32)
    a1 = np.ones((16, 70), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1], "modality_inputs": {"audio": a0}},
        {"input_ids": [2], "modality_inputs": {"audio": a1}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["audio"].shape == (2, 1, 16, 70)
    assert mi["audio_mask"][0, 0, :30].sum().item() == 30
    assert mi["audio_mask"][0, 0, 30:].sum().item() == 0
    assert mi["audio_mask"][1, 0, :].sum().item() == 70


def test_invariant_audio_mask_dtype_is_long():
    """``audio_mask`` tensor is dtype ``torch.long`` (int64)."""
    coll = MultiModalCollator(pad_id=0, max_len=16, max_audios=1)
    audio = np.ones((8, 20), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"audio": audio}}]
    out = coll(samples)
    assert out["modality_inputs"]["audio_mask"].dtype == torch.long


def test_invariant_max_audios_caps_audio_count():
    """More audio clips than ``max_audios`` per sample are capped (line 152).

    Setup: 1 sample with 3 audio clips (3-D: (3, n_mels, T)); ``max_audios=2``.
    Expected: output shape (1, 2, n_mels, T).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_audios=2)
    three_clips = np.ones((3, 16, 50), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"audio": three_clips}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["audio"].shape == (1, 2, 16, 50)


def test_invariant_sample_without_audio_gets_zero_slot_and_zero_mask():
    """A sample without 'audio' in modality_inputs has its audio tensors
    zeroed out and mask set to 0 (lines 143-145).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_audios=1)
    audio = np.ones((8, 20), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1], "modality_inputs": {"audio": audio}},
        {"input_ids": [2], "modality_inputs": {}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["audio_mask"][1, 0, :].sum().item() == 0
    assert mi["audio"][1].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Video path (lines 169-198)
# ---------------------------------------------------------------------------


def test_invariant_video_4d_input_auto_unsqueezed():
    """A 4-D video array (T_v, C, H, W) is unsqueezed to (1, T_v, C, H, W)
    (lines 178-179: ``arr.ndim == 4`` → ``arr = arr[None]``).

    Setup: 1 sample with 4-D video (8, 3, 4, 4).
    Expected: output shape (1, 1, 8, 3, 4, 4).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    vid = np.ones((8, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"video": vid}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["video"].shape == (1, 1, 8, 3, 4, 4)
    assert mi["video_mask"].shape == (1, 1)
    assert mi["video_mask"][0, 0].item() == 1


def test_invariant_video_5d_input_treated_as_multiple_clips():
    """A 5-D video array (N_vid, T_v, C, H, W) is treated as N_vid clips
    (no unsqueeze applied since ndim == 5, lines 177-180).

    Setup: 1 sample with 2 clips (5-D: (2, 8, 3, 4, 4)).
    Expected: output shape (1, 2, 8, 3, 4, 4); mask all-ones.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=2)
    two_clips = np.ones((2, 8, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"video": two_clips}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["video"].shape == (1, 2, 8, 3, 4, 4)
    assert mi["video_mask"].tolist() == [[1, 1]]


def test_invariant_max_videos_caps_video_count():
    """More video clips than ``max_videos`` are capped (line 183).

    Setup: 1 sample with 3 clips; ``max_videos=1``.
    Expected: output shape (1, 1, T_v, C, H, W).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    three_clips = np.ones((3, 8, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"video": three_clips}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["video"].shape == (1, 1, 8, 3, 4, 4)


def test_invariant_video_shape_mismatch_raises_value_error():
    """Video clips with different (T_v, C, H, W) shapes in the same batch
    raise ``ValueError`` (lines 191-193).

    Setup: sample 0 has 4-D video (8, 3, 4, 4); sample 1 has (16, 3, 4, 4)
    — different T_v.
    Expected: ``ValueError`` mentioning "mismatch".
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    vid0 = np.ones((8, 3, 4, 4), dtype=np.float32)
    vid1 = np.ones((16, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1], "modality_inputs": {"video": vid0}},
        {"input_ids": [2], "modality_inputs": {"video": vid1}},
    ]
    with pytest.raises(ValueError, match="mismatch"):
        coll(samples)


def test_invariant_video_tensor_values_and_mask():
    """Video tensor slot is filled with the input data and mask is set to 1
    (lines 195-196); absent samples keep zeros and mask 0.

    Setup: 2 samples — sample 0 has video; sample 1 has no video.
    Expected: video[0] == original values; video_mask[0, 0] == 1;
              video[1] == zeros; video_mask[1, 0] == 0.
    """
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    vid = np.full((8, 3, 4, 4), 0.5, dtype=np.float32)
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1], "modality_inputs": {"video": vid}},
        {"input_ids": [2], "modality_inputs": {}},
    ]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert mi["video"].shape == (2, 1, 8, 3, 4, 4)
    assert mi["video_mask"][0, 0].item() == 1
    assert mi["video_mask"][1, 0].item() == 0
    assert mi["video"][0, 0].sum().item() == pytest.approx(
        np.full((8, 3, 4, 4), 0.5).sum(), rel=1e-4
    )
    assert mi["video"][1].sum().item() == 0.0


def test_invariant_video_tensor_dtype_is_float32():
    """Output video tensor dtype is float32 (from np.zeros float32 init)."""
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    vid = np.ones((4, 3, 2, 2), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"video": vid}}]
    out = coll(samples)
    assert out["modality_inputs"]["video"].dtype == torch.float32


def test_invariant_video_mask_dtype_is_long():
    """``video_mask`` tensor is dtype ``torch.long`` (int64)."""
    coll = MultiModalCollator(pad_id=0, max_len=16, max_videos=1)
    vid = np.ones((4, 3, 2, 2), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"video": vid}}]
    out = coll(samples)
    assert out["modality_inputs"]["video_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# Mixed modalities in a single batch
# ---------------------------------------------------------------------------


def test_invariant_image_audio_video_combined_in_single_batch():
    """A batch with image + audio + video produces all six modality keys.

    Setup: single sample with all three modalities.
    Expected: output ``modality_inputs`` contains image, image_mask, audio,
    audio_mask, video, video_mask.
    """
    coll = MultiModalCollator(
        pad_id=0, max_len=16,
        max_images=1, max_audios=1, max_videos=1,
    )
    img = np.ones((3, 4, 4), dtype=np.float32)
    aud = np.ones((16, 30), dtype=np.float32)
    vid = np.ones((8, 3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{
        "input_ids": [1, 2],
        "modality_inputs": {"image": img, "audio": aud, "video": vid},
    }]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert "image" in mi
    assert "image_mask" in mi
    assert "audio" in mi
    assert "audio_mask" in mi
    assert "video" in mi
    assert "video_mask" in mi


def test_invariant_image_only_no_audio_video_keys():
    """A batch with only images does not produce audio or video keys."""
    coll = MultiModalCollator(pad_id=0, max_len=16)
    img = np.ones((3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {"image": img}}]
    out = coll(samples)
    mi = out["modality_inputs"]
    assert "image" in mi
    assert "audio" not in mi
    assert "video" not in mi


def test_invariant_text_keys_always_present_in_multimodal_batch():
    """Input_ids, attention_mask, labels survive into multimodal output
    (line 103: ``out = dict(text_batch)``).
    """
    coll = MultiModalCollator(pad_id=0, max_len=16)
    img = np.ones((3, 4, 4), dtype=np.float32)
    samples: list[Mapping[str, Any]] = [{"input_ids": [1, 2], "modality_inputs": {"image": img}}]
    out = coll(samples)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert "labels" in out
    assert "modality_inputs" in out


# ---------------------------------------------------------------------------
# _has_modality helper
# ---------------------------------------------------------------------------


def test_has_modality_returns_false_when_no_modality_inputs():
    """``_has_modality`` returns False when no sample has 'modality_inputs'."""
    from lighttrain.builtin_plugins.data.collators.multimodal import _has_modality
    samples: list[Mapping[str, Any]] = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
    assert _has_modality(samples) is False


def test_has_modality_returns_true_when_any_sample_has_modality_inputs():
    """``_has_modality`` returns True when at least one sample has a truthy
    'modality_inputs' value.
    """
    from lighttrain.builtin_plugins.data.collators.multimodal import _has_modality
    samples: list[Mapping[str, Any]] = [
        {"input_ids": [1, 2]},
        {"input_ids": [3, 4], "modality_inputs": {"image": np.ones((3, 4, 4))}},
    ]
    assert _has_modality(samples) is True


def test_has_modality_returns_false_when_modality_inputs_is_empty_dict():
    """``_has_modality`` is falsy for an empty dict (since ``{}`` is falsy
    in Python).

    Pin (current behavior): an empty ``modality_inputs`` dict does NOT
    trigger the multimodal path.
    """
    from lighttrain.builtin_plugins.data.collators.multimodal import _has_modality
    samples: list[Mapping[str, Any]] = [{"input_ids": [1], "modality_inputs": {}}]
    # {} is falsy → _has_modality returns False (current behavior)
    assert _has_modality(samples) is False
