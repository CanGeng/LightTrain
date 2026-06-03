"""Loss functions.

Core losses (CE / MaskedLM / Z-Loss / Composite) live in ``core``. The
distillation family (KL on top-K, hidden-state MSE / cosine, attention transfer)
lives in ``distill``. Preference (DPO/IPO/SimPO/...) and RL (PPO/GRPO) losses
are in their own modules.
"""

from __future__ import annotations

from .core import CompositeLoss, CrossEntropyLoss, MaskedLMLoss, ZLoss
from .distill import (
    AttentionTransferLoss,
    HiddenStatesCosineLoss,
    HiddenStatesMSELoss,
    KLDivLoss,
    LayerMapping,
)

__all__ = [
    "AttentionTransferLoss",
    "CompositeLoss",
    "CrossEntropyLoss",
    "HiddenStatesCosineLoss",
    "HiddenStatesMSELoss",
    "KLDivLoss",
    "LayerMapping",
    "MaskedLMLoss",
    "ZLoss",
]
