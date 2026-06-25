"""Image processors.

Two flavours:

* ``SimpleImageProcessor`` — Pillow-only, hermetic. Resize + CHW float32
  normalize. Default for tests and ``recipes/vlm_sft.yaml`` when no HF model
  is available.

* ``HFImageProcessor`` — wraps ``transformers.AutoImageProcessor``; lazy
  imports so the default test suite stays offline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lighttrain.registry import register


def _open_image(src: Any) -> Any:
    """Best-effort opener: PIL.Image.Image, ndarray, str/Path."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for SimpleImageProcessor") from exc
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, np.ndarray):
        arr = src
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(src, (str, Path)):
        return Image.open(str(src)).convert("RGB")
    raise TypeError(f"unsupported image source type: {type(src).__name__}")


def _to_chw_float32(
    img: Any,
    *,
    size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    img = img.resize(size)
    arr: np.ndarray = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr.astype(np.float32, copy=False)


@register("processor", "simple_image")
class SimpleImageProcessor:
    """Pillow + numpy hermetic image processor.

    Returned record::

        {
          "pixel_values": np.ndarray (C, H, W) float32,
          "modality": "image",
        }

    Multiple inputs (list/tuple) yield ``pixel_values`` shape (N, C, H, W).
    """

    modality = "image"

    def __init__(
        self,
        *,
        size: tuple[int, int] | int = 224,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        if isinstance(size, int):
            size = (int(size), int(size))
        self.size = (int(size[0]), int(size[1]))
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)

    def _process_one(self, src: Any) -> np.ndarray:
        img = _open_image(src)
        return _to_chw_float32(img, size=self.size, mean=self.mean, std=self.std)

    def __call__(self, images: Any, **_: Any) -> dict[str, Any]:
        if isinstance(images, (list, tuple)):
            arrs = [self._process_one(im) for im in images]
            return {
                "pixel_values": np.stack(arrs, axis=0),
                "modality": "image",
            }
        return {
            "pixel_values": self._process_one(images),
            "modality": "image",
        }


@register("processor", "hf_image")
class HFImageProcessor:
    """Wrap ``transformers.AutoImageProcessor``; lazy-loads on first call."""

    modality = "image"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        from_pretrained_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._fp_kwargs = dict(from_pretrained_kwargs or {})
        self._processor: Any | None = None

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path, **self._fp_kwargs
            )
        return self._processor

    def __call__(self, images: Any, **_: Any) -> dict[str, Any]:
        proc = self._ensure_processor()
        if not isinstance(images, (list, tuple)):
            images = [images]
        pil_images = [_open_image(im) for im in images]
        out = proc(images=pil_images, return_tensors="np")
        pv = np.asarray(out["pixel_values"], dtype=np.float32)
        if pv.shape[0] == 1:
            pv = pv[0]
        return {"pixel_values": pv, "modality": "image"}


__all__ = ["SimpleImageProcessor", "HFImageProcessor"]
