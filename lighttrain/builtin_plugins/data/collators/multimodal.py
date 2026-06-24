"""Multimodal collator.

Pads text to longest-in-batch + stacks per-modality tensors with masks.
Delegates pure-text batches to ``CausalLMCollator`` for byte-identical
behavior with R1 / R2.

Output shape::

    {
      "input_ids": (B, T) int64,
      "attention_mask": (B, T) int64,
      "labels": (B, T) int64,
      "modality_inputs": {
          "image":      (B, N_img, C, H, W) float32,
          "image_mask": (B, N_img) int64,           # 1 = present, 0 = pad
          "audio":      (B, N_aud, C_mel, T_aud)    # padded T_aud
          "audio_mask": (B, N_aud, T_aud) int64,
          "video":      (B, N_vid, T_v, C, H, W),
          "video_mask": (B, N_vid) int64,
      },
    }

Image / audio / video pad token ids in ``input_ids`` are inserted by the
processor / sample builder, not by this collator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from lighttrain.registry import register

from .text import CausalLMCollator


def _has_modality(samples: Sequence[Mapping[str, Any]]) -> bool:
    return any(s.get("modality_inputs") for s in samples)


def _stack_pad(arrs: list[np.ndarray], *, pad_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of (..., D_var, ...) arrays along axis 0 padding the
    *last* axis to the max length. Returns (padded, mask) where mask is
    (N, T_max) int64 marking real positions."""
    if not arrs:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.int64)
    ndim = arrs[0].ndim
    max_last = max(int(a.shape[-1]) for a in arrs)
    head_shape = arrs[0].shape[:-1]
    out = np.full((len(arrs), *head_shape, max_last), pad_value, dtype=np.float32)
    mask = np.zeros((len(arrs), max_last), dtype=np.int64)
    for i, a in enumerate(arrs):
        if a.ndim != ndim or a.shape[:-1] != head_shape:
            raise ValueError(
                f"audio shape mismatch in batch: {a.shape} vs leader {arrs[0].shape}"
            )
        t = int(a.shape[-1])
        out[i, ..., :t] = a
        mask[i, :t] = 1
    return out, mask


@register("collator", "multimodal")
class MultiModalCollator:
    """Per-modality pad / stack + token-level placeholder ids.

    Pure-text batches are delegated to ``CausalLMCollator`` so existing
    recipes don't pay any extra cost.
    """

    def __init__(
        self,
        pad_id: int,
        max_len: int = 1024,
        label_ignore: int = -100,
        *,
        max_images: int = 1,
        max_audios: int = 1,
        max_videos: int = 1,
    ) -> None:
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.label_ignore = int(label_ignore)
        self.max_images = int(max_images)
        self.max_audios = int(max_audios)
        self.max_videos = int(max_videos)
        self._text_collator = CausalLMCollator(
            pad_id=self.pad_id,
            max_len=self.max_len,
            label_ignore=self.label_ignore,
        )

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("empty batch")
        text_batch = self._text_collator(samples)
        if not _has_modality(samples):
            return text_batch

        out: dict[str, Any] = dict(text_batch)
        modality_inputs: dict[str, torch.Tensor] = {}

        # ---------- images ----------
        per_sample_images: list[list[np.ndarray]] = []
        for s in samples:
            mi = s.get("modality_inputs") or {}
            img = mi.get("image")
            if img is None:
                per_sample_images.append([])
                continue
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None]
            per_sample_images.append([arr[i] for i in range(arr.shape[0])])

        if any(per_sample_images):
            max_n = min(self.max_images, max(len(x) for x in per_sample_images) or 1)
            # Determine canonical image shape from the first non-empty entry.
            ref = next(a[0] for a in per_sample_images if a)
            c, h, w = ref.shape
            B = len(samples)
            img_tensor = np.zeros((B, max_n, c, h, w), dtype=np.float32)
            img_mask = np.zeros((B, max_n), dtype=np.int64)
            for bi, imgs in enumerate(per_sample_images):
                for ki, im in enumerate(imgs[:max_n]):
                    if im.shape != ref.shape:
                        raise ValueError(
                            f"image shape mismatch: {im.shape} vs ref {ref.shape}"
                        )
                    img_tensor[bi, ki] = im
                    img_mask[bi, ki] = 1
            modality_inputs["image"] = torch.from_numpy(img_tensor)
            modality_inputs["image_mask"] = torch.from_numpy(img_mask)

        # ---------- audio ----------
        per_sample_audio: list[list[np.ndarray]] = []
        for s in samples:
            mi = s.get("modality_inputs") or {}
            au = mi.get("audio")
            if au is None:
                per_sample_audio.append([])
                continue
            arr = np.asarray(au, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None]  # (1, n_mels, T)
            per_sample_audio.append([arr[i] for i in range(arr.shape[0])])

        if any(per_sample_audio):
            max_n = min(self.max_audios, max(len(x) for x in per_sample_audio) or 1)
            ref = next(a[0] for a in per_sample_audio if a)
            n_mels = int(ref.shape[0])
            B = len(samples)
            max_t = max(
                int(a.shape[-1]) for arrs in per_sample_audio for a in arrs[:max_n]
            )
            audio_tensor = np.zeros((B, max_n, n_mels, max_t), dtype=np.float32)
            audio_mask = np.zeros((B, max_n, max_t), dtype=np.int64)
            for bi, arrs in enumerate(per_sample_audio):
                for ki, a in enumerate(arrs[:max_n]):
                    t = int(a.shape[-1])
                    audio_tensor[bi, ki, :, :t] = a
                    audio_mask[bi, ki, :t] = 1
            modality_inputs["audio"] = torch.from_numpy(audio_tensor)
            modality_inputs["audio_mask"] = torch.from_numpy(audio_mask)

        # ---------- video ----------
        per_sample_video: list[list[np.ndarray]] = []
        for s in samples:
            mi = s.get("modality_inputs") or {}
            vid = mi.get("video")
            if vid is None:
                per_sample_video.append([])
                continue
            arr = np.asarray(vid, dtype=np.float32)
            if arr.ndim == 4:
                arr = arr[None]  # (1, T, C, H, W)
            per_sample_video.append([arr[i] for i in range(arr.shape[0])])

        if any(per_sample_video):
            max_n = min(self.max_videos, max(len(x) for x in per_sample_video) or 1)
            ref = next(a[0] for a in per_sample_video if a)
            t_v, c, h, w = ref.shape
            B = len(samples)
            video_tensor = np.zeros((B, max_n, t_v, c, h, w), dtype=np.float32)
            video_mask = np.zeros((B, max_n), dtype=np.int64)
            for bi, arrs in enumerate(per_sample_video):
                for ki, v in enumerate(arrs[:max_n]):
                    if v.shape != ref.shape:
                        raise ValueError(
                            f"video shape mismatch: {v.shape} vs ref {ref.shape}"
                        )
                    video_tensor[bi, ki] = v
                    video_mask[bi, ki] = 1
            modality_inputs["video"] = torch.from_numpy(video_tensor)
            modality_inputs["video_mask"] = torch.from_numpy(video_mask)

        out["modality_inputs"] = modality_inputs
        return out


__all__ = ["MultiModalCollator"]
