"""ORPOTrainer — Odds Ratio Preference Optimization (reference-free)."""

from __future__ import annotations

from typing import Any

from ..losses.preference import ORPOLoss
from ..registry import register
from ._preference_base import PreferenceTrainer


@register("trainer", "orpo")
class ORPOTrainer(PreferenceTrainer):
    """Offline ORPO trainer (reference-free; no artifact join required).

    The SFT component uses ``chosen_nll_loss`` computed by PreferenceTrainer base.
    """

    def __init__(self, *, lam: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loss_fn = ORPOLoss(lam=lam)


__all__ = ["ORPOTrainer"]
