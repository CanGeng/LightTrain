"""TinyCNNClassifier — forward shape + resolution-agnostic pooling."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.models.vision.tiny_cnn import TinyCNNClassifier
from lighttrain.protocols import ModelOutput


def test_forward_logits_shape():
    m = TinyCNNClassifier(in_channels=3, num_classes=5, channels=(8, 16))
    out = m(pixel_values=torch.randn(4, 3, 16, 16))
    assert isinstance(out, ModelOutput)
    assert out.outputs["logits"].shape == (4, 5)


def test_resolution_agnostic():
    # AdaptiveAvgPool2d collapses spatial dims → one model fits any H, W.
    m = TinyCNNClassifier(in_channels=1, num_classes=3, channels=(4,))
    for hw in (8, 13, 32):
        out = m(pixel_values=torch.randn(2, 1, hw, hw))
        assert out.outputs["logits"].shape == (2, 3)


def test_extra_batch_keys_tolerated():
    # forward(**batch) must ignore non-pixel keys (e.g. labels) the loop passes.
    m = TinyCNNClassifier(in_channels=3, num_classes=2)
    out = m(pixel_values=torch.randn(1, 3, 8, 8), labels=torch.tensor([0]))
    assert out.outputs["logits"].shape == (1, 2)
