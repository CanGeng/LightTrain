"""Trainer base — owns the EventBus, dispatches lifecycle, holds shared state.

Concrete subclasses implement ``fit``/``eval``/``predict``. The base class
deliberately stays thin: it only exposes the wiring every paradigm shares
(EventBus, CheckpointManager, LoggerBus, StepContext) so per-paradigm
trainers (pretrain/sft/rl/diffusion/…) can stack on top without adopting
unrelated machinery.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

from ..callbacks.base import EventBus
from ..distributed._context import ParallelContext
from ..engine._context import StepContext
from ..protocols import StepOutput

# Keys that describe the *recipe*, not the run progress. They are written to
# state_dict() for audit, but load_state_dict() never restores them — resuming
# from a step-5 checkpoint with a recipe asking for max_steps=10 must keep
# max_steps=10, otherwise fit() returns immediately with no training (Issue #8).
_RECIPE_CONTROLLED_KEYS: tuple[str, ...] = ("max_steps", "max_epochs")


class Trainer(ABC):
    """Common scaffolding for every Trainer subclass."""

    def __init__(
        self,
        *,
        engine: Any,
        data_module: Any,
        optimizer: Any,
        scheduler: Any | None = None,
        callbacks: list[Any] | None = None,
        logger: Any | None = None,
        ckpt_manager: Any | None = None,
        max_steps: int = 1000,
    ) -> None:
        self.engine = engine
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.max_steps = int(max_steps)

        self.callbacks = list(callbacks or [])
        self.bus = EventBus(self.callbacks)
        self.ctx = StepContext()
        self.ctx.bus = self.bus
        self.ctx.optimizer = optimizer
        self.ctx.scheduler = scheduler
        self.ctx.logger = logger

    # ---- abstract surface --------------------------------------------------

    @abstractmethod
    def fit(self, *, steps: int | None = None) -> Any:
        raise NotImplementedError

    def eval(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def _step(self, batch: dict[str, Any]) -> StepOutput | dict[str, Any]:
        """Execute one gradient update and return metrics.

        All concrete trainers must override this.  Returns a :class:`StepOutput`
        (preferred) or a plain ``dict`` with at minimum a ``"loss"`` key.

        Side effects: updates model weights, steps optimizer, zeros grad.

        For RL trainers (PPO / GRPO) this corresponds to **one inner-epoch
        minibatch policy update**, not one outer rollout step.  The outer
        rollout loop and GAE computation remain in ``fit()``.

        Trainers that need to clear ``ctx.extras["loss_signal"]`` before the
        underlying engine / algorithm call should do so here rather than in
        ``fit()``, since ``train_step()`` delegates directly to this method.
        """
        raise NotImplementedError

    def train_step(self, batch: dict[str, Any]) -> StepOutput:
        """Public entry point for a single training step.

        Calls ``_step()`` and normalises the result to a :class:`StepOutput`.
        ``fit()`` loops must call this method rather than ``_step()`` directly
        so that future hook/callback extensions can be added here without
        touching every trainer's ``fit()`` loop.
        """
        result = self._step(batch)
        return self._normalize_step_output(result)

    def _normalize_step_output(self, result: Any) -> StepOutput:
        if isinstance(result, StepOutput):
            return result
        if isinstance(result, dict):
            return StepOutput(loss=result.get("loss"), metrics=dict(result))
        raise TypeError(
            f"_step() must return StepOutput or dict, got {type(result).__name__}"
        )

    # ---- distributed helpers -----------------------------------------------

    @property
    def _pctx(self) -> ParallelContext:
        """Active ParallelContext; falls back to single_gpu() when not distributed."""
        return getattr(self.ctx, "parallel_ctx", None) or ParallelContext.single_gpu()

    def _is_main(self) -> bool:
        """True only on global rank 0. Guards checkpoint writes and logging."""
        return self._pctx.is_main_process

    # ---- state ------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.ctx.step),
            "epoch": int(self.ctx.epoch),
            "global_step": int(self.ctx.global_step),
            "max_steps": int(self.max_steps),
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        sd = dict(sd)
        for key in _RECIPE_CONTROLLED_KEYS:
            saved = sd.pop(key, None)
            if saved is None:
                continue
            current = getattr(self, key, None)
            if current is not None and int(saved) != int(current):
                warnings.warn(
                    f"Trainer.{key} from checkpoint ({saved}) differs from "
                    f"current recipe value ({current}); keeping current.",
                    UserWarning,
                    stacklevel=2,
                )
        self.ctx.step = int(sd.get("step", 0))
        self.ctx.epoch = int(sd.get("epoch", 0))
        self.ctx.global_step = int(sd.get("global_step", self.ctx.step))


__all__ = ["StepOutput", "Trainer"]
