"""SimPOTrainer — Simple Preference Optimization (reference-free)."""

from __future__ import annotations

from typing import Any

from ..losses.preference import SimPOLoss
from ..registry import register
from ._preference_base import PreferenceTrainer


@register("trainer", "simpo")
class SimPOTrainer(PreferenceTrainer):
    """Offline SimPO trainer (reference-free; no artifact join required)."""

    def __init__(self, *, beta: float = 2.5, gamma: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loss_fn = SimPOLoss(beta=beta, gamma=gamma)


__all__ = ["SimPOTrainer"]
