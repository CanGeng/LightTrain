"""RewardModelTrainer — Bradley-Terry reward model training.

Trains a reward model that scores (prompt, response) pairs. The model is
expected to have a value head that produces a scalar reward for each sequence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..config._resolver import resolve as _resolve
from ..protocols import LossContext, ModelOutput, StepOutput
from ..registry import register
from ..rl.value_heads import LinearValueHead  # re-exported; registers value_head/linear
from ..update_rules._primitives import apply_update
from ._preference_base import PreferenceTrainer, _device_of, _move_batch
from ._utils import validate_batch


@register("trainer", "reward_model")
class RewardModelTrainer(PreferenceTrainer):
    """Trains a reward model with pairwise Bradley-Terry loss.

    The backbone + value head forward pass is performed inside the training
    step. The backbone is the ``model`` passed at construction; the value head
    is constructed lazily on the first batch (auto-detected hidden size).

    Parameters
    ----------
    margin : float
        BT margin added to the reward difference (default 0).
    shared_tower : bool
        If True, a single forward computes rewards for both chosen and rejected
        by concatenating sequences (faster). If False, two separate forwards
        are used (more memory-efficient for long sequences).
    """

    def __init__(
        self,
        *,
        engine: Any,
        data_module: Any,
        optimizer: Any,
        model: Any | None = None,
        scheduler: Any | None = None,
        callbacks: list[Any] | None = None,
        logger: Any | None = None,
        ckpt_manager: Any | None = None,
        max_steps: int = 1000,
        val_every: int = 0,
        ckpt_every: int = 500,
        log_every: int = 50,
        device: str | torch.device | None = None,
        margin: float = 0.0,
        shared_tower: bool = True,
        grad_clip: float = 1.0,
        value_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine=engine,
            data_module=data_module,
            optimizer=optimizer,
            model=model,
            scheduler=scheduler,
            callbacks=callbacks,
            logger=logger,
            ckpt_manager=ckpt_manager,
            max_steps=max_steps,
            val_every=val_every,
            ckpt_every=ckpt_every,
            log_every=log_every,
            device=device,
        )
        self.margin = float(margin)
        self.shared_tower = bool(shared_tower)
        # F3: grad_clip is now a knob. Default is 1.0 (matches ppo/grpo/preference
        # — RM training routinely benefits from clipping); set 0.0 for the legacy
        # no-clip behaviour.
        self.grad_clip = float(grad_clip)
        self._value_head_spec = value_head
        self._value_head: LinearValueHead | None = None

    def _get_value_head(self, model: Any) -> LinearValueHead:
        if self._value_head is None:
            # Auto-detect hidden size from model config.
            hidden_size = getattr(
                getattr(model, "config", None), "hidden_size", None
            ) or getattr(
                getattr(model, "config", None), "n_embd", None
            )
            if hidden_size is None:
                raise RuntimeError(
                    "RewardModelTrainer: cannot auto-detect hidden_size from model.config. "
                    "Set model.config.hidden_size manually."
                )
            # F4: resolve via the value_head registry; default spec reproduces the
            # old reward head (last-token, no bias, default init) → bit-identical.
            spec = {"name": "linear", "bias": False, "zero_init": False,
                    "reduction": "last"}
            if self._value_head_spec:
                spec.update(self._value_head_spec)
            spec["hidden_size"] = int(hidden_size)
            self._value_head = _resolve(spec, category="value_head").to(
                device=self.device, dtype=next(model.parameters()).dtype
            )
        return self._value_head

    def _score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run backbone + value head → (B,) reward scalars."""
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.model(**kwargs)
        if isinstance(out, ModelOutput):
            if out.hidden_states is not None:
                hidden = out.hidden_states[-1]   # (B, T, H)
            else:
                raise RuntimeError(
                    "RewardModelTrainer: model must return hidden_states. "
                    "Set output_hidden_states=True on the model adapter."
                )
        else:
            raise TypeError(
                "RewardModelTrainer expects model to return ModelOutput with hidden_states."
            )
        vhead = self._get_value_head(self.model)
        return vhead(hidden)   # (B,)

    def _reward_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute Bradley-Terry reward scores and update parameters.

        Semantically different from PreferenceTrainer._preference_step():
        this method scores sequences with a LinearValueHead rather than
        computing log-probs for a preference loss function.
        """
        validate_batch(batch, [
            "chosen_input_ids", "rejected_input_ids",
        ], "RewardModelTrainer")
        self.model.train()

        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch.get("chosen_attention_mask")
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch.get("rejected_attention_mask")

        chosen_rewards = self._score(chosen_ids, chosen_mask)     # (B,)
        rejected_rewards = self._score(rejected_ids, rejected_mask)  # (B,)

        # Bradley-Terry pairwise loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards - self.margin).mean()

        self.bus.dispatch("on_loss_computed", loss=loss, batch=batch, ctx=self.ctx)
        # Backward / step / scheduler via the shared primitive. RM has always
        # used a bare backward with no gradient clipping, so pass accelerator=None
        # and grad_clip=0.0 to preserve that exactly; micro_state is required
        # even at accumulate=1 (RM does not accumulate).
        apply_update(
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
            ctx=self.ctx,
            micro_state=self._micro,
            scheduler=self.scheduler,
            accelerator=None,   # AMP-bypass is keystone seam#3; unchanged here
            grad_clip=self.grad_clip,
            accumulate_grad_batches=1,
            bus=self.bus,
        )

        return {
            "loss": float(loss.detach()),
            "reward_chosen": float(chosen_rewards.mean().detach()),
            "reward_rejected": float(rejected_rewards.mean().detach()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean().detach()),
        }

    # Backward-compat alias: old tests / external code calling _preference_step still work.
    _preference_step = _reward_step  # type: ignore[assignment]

    def _step(self, batch: dict[str, Any]) -> StepOutput:  # type: ignore[override]
        """Bridge to _reward_step() for the unified train_step() protocol."""
        raw = self._reward_step(batch)
        return StepOutput(loss=raw.get("loss"), metrics=dict(raw))


__all__ = ["LinearValueHead", "RewardModelTrainer"]
