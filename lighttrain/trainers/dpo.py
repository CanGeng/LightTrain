"""DPOTrainer — Direct Preference Optimization."""

from __future__ import annotations

from typing import Any

from ..losses.preference import DPOLoss
from ..registry import register
from ._preference_base import PreferenceTrainer


@register("trainer", "dpo")
class DPOTrainer(PreferenceTrainer):
    """Offline DPO trainer.

    Requires ``aux.<ref_namespace>.chosen_logprobs`` and
    ``aux.<ref_namespace>.rejected_logprobs`` in the batch (populated by
    an ArtifactJoinedDataset with a pre-computed reference-policy artifact).
    """

    def __init__(self, *, beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loss_fn = DPOLoss(beta=beta)


__all__ = ["DPOTrainer"]
