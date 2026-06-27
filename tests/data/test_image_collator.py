"""ImageClassificationCollator — stack pixel_values + integer labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from lighttrain.builtin_plugins.data.collators.image import ImageClassificationCollator


def test_stacks_pixels_and_labels():
    coll = ImageClassificationCollator()
    samples: list[Mapping[str, Any]] = [{"pixel_values": torch.randn(3, 8, 8), "label": i % 3} for i in range(5)]
    batch = coll(samples)
    assert batch["pixel_values"].shape == (5, 3, 8, 8)
    assert batch["pixel_values"].dtype == torch.float32
    assert batch["labels"].shape == (5,)
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [0, 1, 2, 0, 1]


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        ImageClassificationCollator()([])


def test_pad_id_injected_but_ignored():
    # pad_id is injected by the data module; image batches don't pad.
    coll = ImageClassificationCollator(pad_id=0)
    batch = coll([{"pixel_values": torch.zeros(1, 4, 4), "label": 2}])
    assert batch["pixel_values"].shape == (1, 1, 4, 4)
    assert batch["labels"].tolist() == [2]
