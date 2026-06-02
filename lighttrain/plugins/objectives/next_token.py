"""NextTokenObjective — wraps cross-entropy for standard LM training.

This is a thin ObjectiveProfile shim so that Trainer code that dispatches on
``loss_family`` works uniformly whether the user passes a bare ``CrossEntropyLoss``
or this objective wrapper.
"""

from __future__ import annotations

from typing import Any

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


@register("objective", "next_token")
class NextTokenObjective:
    """Standard next-token prediction objective.

    Wraps ``CrossEntropyLoss`` (or any registered CE-compatible loss) and
    sets ``LossContext.loss_family = "next_token"`` on every step.
    """

    loss_family: str = "next_token"

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self._loss_fn: Any = None

    # ------------------------------------------------------------------
    # Lazy loss fn construction (avoids circular import at module load)
    # ------------------------------------------------------------------

    def _get_loss_fn(self) -> Any:
        if self._loss_fn is None:
            from lighttrain.losses.core import CrossEntropyLoss
            self._loss_fn = CrossEntropyLoss(ignore_index=self.ignore_index)
        return self._loss_fn

    # ------------------------------------------------------------------
    # ObjectiveProfile protocol
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """No-op: next-token training needs no extra batch transforms."""
        return batch

    def __call__(
        self,
        outputs: ModelOutput,
        batch: dict,
        ctx: LossContext,
    ) -> dict:
        ctx.loss_family = self.loss_family
        return self._get_loss_fn()(outputs, batch, ctx)


__all__ = ["NextTokenObjective"]
