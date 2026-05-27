"""KTOTrainer — Kahneman-Tversky Optimization."""

from __future__ import annotations

from typing import Any

from ..losses.preference import KTOLoss
from ..registry import register
from ._preference_base import PreferenceTrainer


@register("trainer", "kto")
class KTOTrainer(PreferenceTrainer):
    """Offline KTO trainer.

    Requires reference log-probs from artifact (same as DPO).
    """

    def __init__(
        self,
        *,
        beta: float = 0.1,
        lambda_desirable: float = 1.0,
        lambda_undesirable: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._loss_fn = KTOLoss(
            beta=beta,
            lambda_desirable=lambda_desirable,
            lambda_undesirable=lambda_undesirable,
        )


__all__ = ["KTOTrainer"]
