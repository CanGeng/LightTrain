"""GRPOTrainer — Group Relative Policy Optimization.

Collects G responses per prompt, normalizes advantages within each group,
then applies a clipped surrogate update. No value model is needed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..losses.rl import GRPOLoss
from ..protocols import ModelOutput, StepOutput
from ..registry import register
from ..rl.buffers import RolloutBuffer
from ..rl.ref_policy import freeze_as_ref
from ..rl.rollout import HFGenerateBackend, RolloutEngine
from ..update_rules.rl import RLUpdateRule
from ._utils import _device_of, _move_batch, validate_batch
from .base import Trainer

_log = logging.getLogger(__name__)


@register("trainer", "grpo")
class GRPOTrainer(Trainer):
    """Single-GPU GRPO trainer.

    Parameters
    ----------
    group_size : int
        Number of responses generated per prompt (G).
    rollout_prompts : int
        Number of distinct prompts per rollout step (batch size of the prompt loader).
    ppo_epochs : int
        Inner epochs over the rollout buffer.
    mini_batch_size : int
        Mini-batch size for the inner loop.
    clip_eps, beta_kl : float
        GRPOLoss parameters.
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
        val_every: int = 0,  # accepted for runtime compat; RL loop doesn't use val
        ckpt_every: int = 500,
        log_every: int = 10,
        device: str | torch.device | None = None,
        reward_fn: Callable[[Any, Any], list[float]] | None = None,
        group_size: int = 4,
        ppo_epochs: int = 1,
        mini_batch_size: int = 8,
        clip_eps: float = 0.2,
        beta_kl: float = 0.0,
        lora_base_as_ref: bool = False,
        max_new_tokens: int = 128,
        ignore_index: int = -100,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__(
            engine=engine,
            data_module=data_module,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            logger=logger,
            ckpt_manager=ckpt_manager,
            max_steps=max_steps,
        )
        self.model = model
        if model is not None:
            self.ctx.model = model
        if device is not None:
            self.device = torch.device(device)
            if model is not None:
                self.model = model.to(self.device)
                self.ctx.model = self.model
        else:
            self.device = _device_of(self.model) if self.model is not None else None

        self.reward_fn = reward_fn
        self.group_size = int(group_size)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.ckpt_every = int(ckpt_every)
        self.log_every = max(1, int(log_every))
        self.ignore_index = int(ignore_index)
        self._stop_requested = False

        self._loss_fn = GRPOLoss(clip_eps=clip_eps, beta_kl=beta_kl)
        self._rl_rule = RLUpdateRule(grad_clip=grad_clip)
        backend = HFGenerateBackend(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_return_sequences=group_size,
        )
        self._rollout_engine = RolloutEngine(backend, ignore_index=ignore_index)
        self._buffer = RolloutBuffer(max_size=2048)
        self._lora_base_as_ref = bool(lora_base_as_ref)

    # ------------------------------------------------------------------ fit

    def fit(self, *, steps: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        if self.model is None:
            raise RuntimeError("GRPOTrainer.fit: model is not set.")
        if self.optimizer is None:
            raise RuntimeError("GRPOTrainer.fit: optimizer is not set.")
        if self.reward_fn is None:
            raise RuntimeError("GRPOTrainer.fit: reward_fn is not set.")

        ref_policy = freeze_as_ref(self.model, lora_base_as_ref=self._lora_base_as_ref)

        target = int(steps) if steps is not None else self.max_steps
        loader = self.data_module.train_loader()
        iterator = iter(loader)

        self.bus.dispatch("on_train_start", trainer=self, ctx=self.ctx)
        self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)

        last_metrics: dict[str, Any] = {}
        try:
            while self.ctx.step < target and not self._stop_requested:
                try:
                    raw = next(iterator)
                except StopIteration:
                    self.bus.dispatch("on_epoch_end", epoch=self.ctx.epoch, ctx=self.ctx)
                    self.ctx.epoch += 1
                    iterator = iter(loader)
                    self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)
                    raw = next(iterator)

                batch = _move_batch(raw, self.device)
                self.bus.dispatch("on_rollout_begin", ctx=self.ctx)

                prompt_ids = batch.get("input_ids", batch.get("prompt_input_ids"))
                prompt_mask = batch.get("attention_mask", batch.get("prompt_attention_mask"))

                self._buffer.clear()
                episodes = self._rollout_engine.rollout(
                    self.model, prompt_ids, prompt_mask, self.reward_fn
                )
                for ep in episodes:
                    self._buffer.add(ep)

                self.bus.dispatch("on_rollout_end", episodes=episodes, ctx=self.ctx)

                # Group-relative advantages (computed in GRPOLoss per-batch)
                rewards = self._buffer.all_rewards()

                inner_metrics_list: list[dict[str, Any]] = []
                for _epoch in range(self.ppo_epochs):
                    for mb in self._buffer.batches(self.mini_batch_size, shuffle=True):
                        mb = _move_batch(mb, self.device)
                        step_out = self.train_step(mb)
                        step_m = step_out.metrics
                        inner_metrics_list.append(step_m)

                if inner_metrics_list:
                    last_metrics = {
                        k: float(sum(m.get(k, 0.0) for m in inner_metrics_list) / len(inner_metrics_list))
                        for k in inner_metrics_list[0]
                    }
                    last_metrics["mean_reward"] = float(rewards.mean())

                self.bus.dispatch("on_reward_computed", rewards=rewards, ctx=self.ctx)

                self.ctx.step += 1
                self.ctx.global_step = self.ctx.step

                if self.ctx.step % self.log_every == 0 and self.logger is not None and self._is_main():
                    scalar = {k: v for k, v in last_metrics.items()
                              if isinstance(v, float) and math.isfinite(v)}
                    self.logger.log_dict(scalar, step=self.ctx.step)

                if (
                    self.ckpt_manager is not None
                    and self.ckpt_every > 0
                    and self.ctx.step % self.ckpt_every == 0
                ):
                    self.ckpt_manager.save(
                        step=self.ctx.step,
                        state={"model": self.model.state_dict(), "trainer": self.state_dict()},
                        kind="step",
                        extras={"metrics": last_metrics},
                        parallel_ctx=self._pctx,
                    )

        except BaseException as exc:  # noqa: BLE001
            try:
                self.bus.dispatch(
                    "on_exception", trainer=self, exception=exc, step=self.ctx.step, batch=None
                )
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed secondary exception in on_exception dispatch", exc_info=True)
            raise
        finally:
            try:
                self.bus.dispatch("on_train_end", trainer=self, ctx=self.ctx, metrics=last_metrics)
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed exception in on_train_end dispatch", exc_info=True)
            if self.logger is not None:
                try:
                    self.logger.flush()
                except Exception:  # noqa: BLE001
                    _log.warning("Suppressed exception in logger.flush", exc_info=True)

        return last_metrics

    # ------------------------------------------------------------------ grpo step

    def _grpo_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch, [
            "input_ids", "log_probs_old", "group_ids", "rewards",
        ], "GRPOTrainer")
        self.model.train()

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        log_probs_old = batch["log_probs_old"]
        group_ids = batch.get("group_ids")
        rewards = batch.get("rewards")

        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        # TODO: Wrap this forward pass in accelerator.autocast() or torch.autocast() in the future for AMP memory efficiency.
        out = self.model(**kwargs)
        logits = out.outputs["logits"] if isinstance(out, ModelOutput) else out["logits"]

        if labels is not None:
            # NOTE: `labels` is used as a None-sentinel only; the actual next-token
            # targets come from input_ids[:, 1:]. Prompt positions are masked by
            # GRPOLoss via (labels != -100), not by skipping them here.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lp = F.log_softmax(shift_logits, dim=-1)
            log_probs_new = torch.gather(lp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_new = torch.cat(
                [torch.zeros_like(log_probs_new[:, :1]), log_probs_new], dim=1
            )
        else:
            log_probs_new = torch.zeros_like(log_probs_old)

        # Sequence-level advantages = per-sample rewards (group norm in GRPOLoss)
        advantages = rewards if rewards is not None else torch.zeros(input_ids.size(0))
        if advantages.device != log_probs_new.device:
            advantages = advantages.to(log_probs_new.device)

        # Populate ctx for RLUpdateRule (backward/callbacks delegated below)
        self.ctx.extras.update({
            "log_probs_new": log_probs_new,
            "log_probs_old": log_probs_old,
            "advantages": advantages,
            "group_ids": group_ids,
            "model": self.model,
        })
        self.ctx.loss_fn = self._loss_fn
        self.ctx.model = self.model
        return self._rl_rule.step(self.model, batch, self.ctx)

    def _step(self, batch: dict[str, Any]) -> StepOutput:  # type: ignore[override]
        """Bridge to _grpo_step() for the unified train_step() protocol.

        For GRPOTrainer, train_step() / _step() corresponds to **one inner-epoch
        minibatch policy update**.  Group-relative advantages are computed inside
        GRPOLoss per batch; the outer group rollout loop remains in fit().
        """
        self.ctx.extras.pop("loss_signal", None)
        raw = self._grpo_step(batch)
        return StepOutput(loss=raw.get("loss"), metrics=dict(raw))

    def eval(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        return []


__all__ = ["GRPOTrainer"]
