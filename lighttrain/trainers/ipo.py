"""IPOTrainer — Identity Preference Optimization."""

from __future__ import annotations

from typing import Any

from ..losses.preference import IPOLoss
from ..registry import register
from ._preference_base import PreferenceTrainer


@register("trainer", "ipo")
class IPOTrainer(PreferenceTrainer):
    """Offline IPO trainer."""

    def __init__(self, *, beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loss_fn = IPOLoss(beta=beta)


__all__ = ["IPOTrainer"]
