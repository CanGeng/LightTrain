"""Verify CrossEntropyLoss does next-token shift (DESIGN §9.5, fixes REVIEW #1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.protocols import LossContext, ModelOutput


def test_ce_loss_shifts_labels_off_by_one():
    """logits[:, :-1, :] aligned with labels[:, 1:] — model must predict next."""
    torch.manual_seed(0)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    # labels mirror input_ids; pick deterministic ids in-range
    labels = torch.tensor([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]], dtype=torch.long)

    out = CrossEntropyLoss()(ModelOutput(outputs={"logits": logits}),
                             {"labels": labels},
                             LossContext())

    # Manual shift reference
    ref = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["loss"], ref, atol=1e-6)


def test_ce_loss_perfect_shift_yields_near_zero():
    """If a model perfectly predicts ids[t+1] at position t, CE → ~0."""
    B, T, V = 1, 6, 10
    labels = torch.tensor([[3, 1, 4, 1, 5, 9]], dtype=torch.long)

    # Build logits so that argmax at position t equals labels[t+1]
    logits = torch.full((B, T, V), -10.0)
    for t in range(T - 1):
        logits[0, t, labels[0, t + 1]] = 20.0
    # last position unconstrained (no shifted label there)

    out = CrossEntropyLoss()(ModelOutput(outputs={"logits": logits}),
                             {"labels": labels},
                             LossContext())
    assert out["loss"].item() < 1e-4
