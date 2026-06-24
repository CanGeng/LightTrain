"""TinyCNNClassifier — the reference supervised-vision model.

A minimal convolutional image classifier: a small ``conv -> relu -> maxpool``
stack, then adaptive average pooling and a linear head. Input ``pixel_values``
of shape ``(B, C, H, W)``; output ``logits`` of shape ``(B, num_classes)``.

The adaptive pool collapses spatial dims, so the same model fits any input
resolution. Drives the ``image_cls`` recipe via the generic ``pretrain``
trainer + the ``classification`` loss — no objective/trainer subclassing.

Registered as ``@register("model", "tiny_cnn")``.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register


@register("model", "tiny_cnn")
class TinyCNNClassifier(nn.Module):
    """Small conv-net image classifier (conv stack -> GAP -> linear head)."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, ...] | list[int] = (16, 32),
        **kwargs: Any,  # tolerate extra recipe keys (e.g. image_size) — unused
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        dims = [int(in_channels), *(int(c) for c in channels)]
        blocks: list[nn.Module] = []
        for cin, cout in zip(dims[:-1], dims[1:], strict=True):
            blocks += [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], self.num_classes)

    def forward(self, **batch: Any) -> ModelOutput:
        x = batch["pixel_values"].float()
        h = self.features(x)
        h = self.pool(h).flatten(1)   # (B, C_last)
        logits = self.head(h)         # (B, num_classes)
        return ModelOutput(outputs={"logits": logits})


__all__ = ["TinyCNNClassifier"]
