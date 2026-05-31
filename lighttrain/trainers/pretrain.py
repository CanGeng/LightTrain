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

from ..registry import register
from .base import Trainer


@register("trainer", "pretrain")
class PretrainTrainer(Trainer):
    """Single-GPU causal-LM pretraining trainer (the flat Trainer)."""


__all__ = ["PretrainTrainer"]
