"""Image-classification collator.

Stacks per-sample ``pixel_values`` ``(C, H, W)`` into a ``(B, C, H, W)`` float
batch and integer class labels into ``(B,)``. The reference collator for
supervised vision (image -> label), pairing with a dataset that yields
``{"pixel_values", "label"}`` samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from lighttrain.registry import register


@register("collator", "image")
class ImageClassificationCollator:
    """Stack ``pixel_values`` ``(B, C, H, W)`` + integer ``labels`` ``(B,)``."""

    def __init__(self, *, pad_id: int | None = None) -> None:  # noqa: ARG002
        # pad_id is injected by the data module; image batches don't pad.
        pass

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("empty batch")
        pixel_values = torch.stack(
            [torch.as_tensor(s["pixel_values"], dtype=torch.float) for s in samples]
        )
        labels = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


__all__ = ["ImageClassificationCollator"]
