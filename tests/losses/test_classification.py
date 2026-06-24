"""ClassificationLoss — rank-2 CE + top-1 accuracy + loss_family."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.losses.classification import ClassificationLoss
from lighttrain.protocols import LossContext, ModelOutput


def _ctx() -> LossContext:
    return LossContext(step=0, epoch=0)


def test_loss_and_perfect_accuracy():
    loss_fn = ClassificationLoss()
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    out = loss_fn(
        ModelOutput(outputs={"logits": logits}), {"labels": torch.tensor([0, 1, 0])}, _ctx()
    )
    assert out["loss"].item() < 0.05
    assert float(out["acc"]) == 1.0
    assert isinstance(out["acc"], torch.Tensor)  # tensor so the update rule logs it


def test_half_accuracy():
    loss_fn = ClassificationLoss()
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])  # both predict class 0
    out = loss_fn(
        ModelOutput(outputs={"logits": logits}), {"labels": torch.tensor([0, 1])}, _ctx()
    )
    assert float(out["acc"]) == 0.5


def test_ignore_index_excluded_from_accuracy():
    loss_fn = ClassificationLoss(ignore_index=-100)
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])  # 2nd would be wrong, but ignored
    out = loss_fn(
        ModelOutput(outputs={"logits": logits}), {"labels": torch.tensor([0, -100])}, _ctx()
    )
    assert float(out["acc"]) == 1.0  # only the valid sample counts


def test_loss_family_is_classification():
    assert ClassificationLoss.loss_family == "classification"
