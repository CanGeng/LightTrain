"""PretrainTrainer — the canonical causal-LM pretraining trainer.

After the keystone refactor this is just the flat :class:`~lighttrain.trainers.base.Trainer`
under the ``pretrain`` registry name: the loop, the device-move ``produce_batch``,
the engine-routed ``forward_loss`` (forward + loss + backward via
``StandardUpdateRule``), eval/predict, checkpointing, crash bundle and resume
all live on the base. Flow::

    on_train_start
      while step < max_steps:
        on_epoch_begin (when (re-)building the iter)
          on_train_batch_start
            engine.step(batch, ctx)   # owns the per-step events
          on_train_batch_end
        on_epoch_end (on iter exhaustion)
        every val_every steps   → eval()
        every ckpt_every steps  → ckpt_manager.save(...)
        every log_every steps   → logger.log_dict(metrics)
      on_train_end → final ckpt → logger.close()

Honors callback ``Signal.STOP_TRAINING`` to break the loop early. The
``BestCheckpointCallback`` is opt-in: if it lives in ``self.callbacks`` and
flags ``should_save`` after a step/eval, the trainer triggers a ``best`` save.
"""

from __future__ import annotations

from typing import Any

from lighttrain.registry import register
from lighttrain.trainers.base import Trainer


@register("trainer", "pretrain")
class PretrainTrainer(Trainer):
    """Single-GPU causal-LM pretraining trainer (the flat Trainer)."""

    def default_objective(self) -> Any:
        """Next-token cross-entropy when the recipe omits loss/objective.

        The CE default lives here (not on the abstract base) so core stays free
        of any concrete loss (DESIGN §3.3); ``CrossEntropyLoss`` is a registered
        impl in ``lighttrain.builtin_plugins.losses``.
        """
        from lighttrain.architectures.profile import LossOnlyObjective
        from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss

        return LossOnlyObjective(CrossEntropyLoss(), loss_family="next_token")


__all__ = ["PretrainTrainer"]
