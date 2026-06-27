"""Demo-only vision components for the image-classification recipe.

DEMO ONLY. A tiny synthetic image-classification dataset registered via the
``image_cls`` recipe's ``user_modules:`` so it runs end-to-end with no external
data. Each class owns a fixed random prototype image; samples are the prototype
plus Gaussian noise, so a small CNN separates them and the loss decreases. Not
a core capability — do not depend on it outside the bundled recipe.

Registered components:
    dataset   synthetic_image   → {"pixel_values": (C, H, W), "label": int}
(the first-party ``image`` collator stacks these into a batch.)
"""

from __future__ import annotations

from typing import Any

import torch

from lighttrain import register


@register("dataset", "synthetic_image")
class SyntheticImageDataset:
    """Class-separable random images: per-class prototype + Gaussian noise."""

    def __init__(
        self,
        *,
        num_classes: int = 4,
        in_channels: int = 3,
        image_size: int = 8,
        num_samples: int = 512,
        noise: float = 0.3,
        seed: int = 0,
        tokenizer: Any = None,  # injected by the data module; unused
    ) -> None:
        g = torch.Generator().manual_seed(int(seed))
        c, h, w = int(in_channels), int(image_size), int(image_size)
        protos = torch.randn(int(num_classes), c, h, w, generator=g)
        self.labels = torch.randint(0, int(num_classes), (int(num_samples),), generator=g)
        self.images = protos[self.labels] + float(noise) * torch.randn(
            int(num_samples), c, h, w, generator=g
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"pixel_values": self.images[idx], "label": int(self.labels[idx])}
