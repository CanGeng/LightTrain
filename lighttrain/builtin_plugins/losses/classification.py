"""Classification loss — supervised cross-entropy with top-1 accuracy.

The discriminative counterpart to the LM ``cross_entropy`` loss: expects
rank-2 ``logits`` ``(B, C)`` and rank-1 integer ``labels`` ``(B,)``, and reports
top-1 accuracy alongside the loss. ``loss_family = "classification"`` so
``LossOnlyObjective`` and downstream callbacks specialise by paradigm rather
than inheriting the LM ``"next_token"`` tag.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


@register("loss", "classification")
class ClassificationLoss:
    """Single-label classification cross-entropy + top-1 accuracy.

    Expects ``logits`` ``(B, C)`` and ``labels`` ``(B,)`` integer class ids.
    Returns ``{"loss": ..., "acc": ...}`` — ``acc`` is a tensor so the update
    rule logs it automatically.
    """

    # Paradigm tag so ``LossOnlyObjective`` inherits a meaningful loss_family
    # when a recipe writes ``loss: classification``.
    loss_family: str = "classification"

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        if isinstance(model_output, ModelOutput):
            logits = model_output.outputs["logits"]
        else:
            logits = model_output["logits"]
        labels = batch["labels"].long()
        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            valid = labels != self.ignore_index
            correct = (logits.argmax(dim=-1) == labels) & valid
            acc = correct.sum().float() / valid.sum().clamp(min=1)
        return {"loss": loss, "acc": acc}


__all__ = ["ClassificationLoss"]
