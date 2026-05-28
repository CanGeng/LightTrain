"""PPOTrainer — Proximal Policy Optimization online RL trainer.

Training loop::

    for each outer step:
        1. Rollout: collect N episodes with RolloutEngine
        2. Score: reward_fn assigns rewards
        3. Compute GAE advantages + returns
        4. PPO inner epochs: K passes over the buffer in mini-batches
           - compute new log-probs + value estimates
           - apply PPOSurrogateLoss
           - backprop + optimizer step
        5. Log metrics; checkpoint if due
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

import torch.nn as nn

from ..losses.rl import PPOSurrogateLoss
from ..protocols import ModelOutput, StepOutput
from ..registry import register
from ..rl.buffers import RolloutBuffer
from ..rl.gae import compute_gae, normalize_advantages
from ..rl.ref_policy import ReferencePolicy, freeze_as_ref
from ..rl.rollout import HFGenerateBackend, RolloutEngine
from ..update_rules.rl import RLUpdateRule
from ._utils import _device_of, _move_batch, validate_batch
from .base import Trainer


class LinearValueHead(nn.Module):
    """Scalar value head for PPO — projects last hidden state to V(s).

    Connects to PPOTrainer when ``use_value_head=True``.
    The value head is a single linear layer that reads the last hidden state
    of the model (accessed via ``ModelOutput.hidden_states[-1]``) and outputs
    a per-token value estimate.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H)
        Returns:
            values: (B, T)
        """
        return self.linear(hidden_states).squeeze(-1)


@register("trainer", "ppo")
class PPOTrainer(Trainer):
    """Single-GPU PPO trainer with adaptive KL penalty.

    Parameters
    ----------
    rollout_steps : int
        Number of rollout episodes to collect before each PPO update.
    ppo_epochs : int
        Number of passes over the rollout buffer per outer step (K).
    mini_batch_size : int
        Mini-batch size within each PPO epoch.
    gamma, lam : float
        GAE discount and lambda.
    clip_eps : float
        PPO clip range ε.
    vf_coef, ent_coef : float
        Value function and entropy coefficients.
    target_kl : float or None
        If set, abort PPO epochs when approx KL exceeds this threshold.
    lora_base_as_ref : bool
        Use LoRA base weights as the reference policy (no deepcopy).
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
        rollout_steps: int = 32,
        ppo_epochs: int = 4,
        mini_batch_size: int = 8,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        target_kl: float | None = 0.02,
        lora_base_as_ref: bool = False,
        max_new_tokens: int = 128,
        ignore_index: int = -100,
        use_value_head: bool = False,
        value_head_dim: int | None = None,
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
        self.rollout_steps = int(rollout_steps)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.ckpt_every = int(ckpt_every)
        self.log_every = max(1, int(log_every))
        self.target_kl = float(target_kl) if target_kl is not None else None
        self.lora_base_as_ref = bool(lora_base_as_ref)
        self.ignore_index = int(ignore_index)
        self._stop_requested = False

        self._loss_fn = PPOSurrogateLoss(
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef
        )
        self._rl_rule = RLUpdateRule(grad_clip=grad_clip)
        backend = HFGenerateBackend(
            max_new_tokens=max_new_tokens, do_sample=True
        )
        self._rollout_engine = RolloutEngine(backend, ignore_index=ignore_index)
        self._buffer = RolloutBuffer(max_size=rollout_steps * 4)
        self._ref_policy: ReferencePolicy | None = None
        # optional value head for advantage estimation
        self._use_value_head = bool(use_value_head)
        self._value_head: LinearValueHead | None = None
        self._value_head_dim = value_head_dim

    # ------------------------------------------------------------------ fit

    def fit(self, *, steps: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        if self.model is None:
            raise RuntimeError("PPOTrainer.fit: model is not set.")
        if self.optimizer is None:
            raise RuntimeError("PPOTrainer.fit: optimizer is not set.")
        if self.reward_fn is None:
            raise RuntimeError("PPOTrainer.fit: reward_fn is not set.")

        # Freeze reference policy once before training.
        self._ref_policy = freeze_as_ref(
            self.model, lora_base_as_ref=self.lora_base_as_ref
        )

        target = int(steps) if steps is not None else self.max_steps
        loader = self.data_module.train_loader()
        iterator = iter(loader)

        self.bus.dispatch("on_train_start", trainer=self, ctx=self.ctx)
        self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)

        last_metrics: dict[str, Any] = {}
        try:
            while self.ctx.step < target and not self._stop_requested:
                # ---- rollout phase ----------------------------------------
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
                    self.model,
                    prompt_ids,
                    prompt_mask,
                    self.reward_fn,
                )
                for ep in episodes:
                    self._buffer.add(ep)

                self.bus.dispatch(
                    "on_rollout_end",
                    episodes=episodes,
                    ctx=self.ctx,
                )

                # ---- compute GAE -------------------------------------------
                rewards = self._buffer.all_rewards()   # (N,)

                # Populate ep.values via batched value-head inference (noop if head not ready)
                self._compute_buffer_values()

                # use value head if enabled, else zero baseline
                if self._use_value_head and self._value_head is not None:
                    values = self._buffer.all_values()  # (N, 1) — ep.values is (1,) scalar
                    if values is not None:
                        values_for_gae = values.to(rewards.device)      # (N, 1) ✅
                    else:
                        values_for_gae = torch.zeros_like(rewards.unsqueeze(1))
                else:
                    values_for_gae = torch.zeros_like(rewards.unsqueeze(1))  # (N, 1)

                advantages_seq, returns_seq = compute_gae(
                    rewards.unsqueeze(1).expand(-1, 1),   # (N, 1)
                    values_for_gae,
                    gamma=self.gamma,
                    lam=self.lam,
                )
                advantages_seq = normalize_advantages(advantages_seq)

                # ---- PPO inner epochs ---------------------------------------
                inner_metrics: list[dict[str, Any]] = []
                for _epoch in range(self.ppo_epochs):
                    early_stop = False
                    for minibatch in self._buffer.batches(
                        self.mini_batch_size,
                        shuffle=True,
                        advantages=advantages_seq.squeeze(1),
                        returns=returns_seq.squeeze(1),
                    ):
                        mb = _move_batch(minibatch, self.device)
                        step_out = self.train_step(mb)
                        step_metrics = step_out.metrics
                        inner_metrics.append(step_metrics)
                        if (
                            self.target_kl is not None
                            and step_metrics.get("approx_kl", 0.0) > self.target_kl
                        ):
                            early_stop = True
                            break
                    if early_stop:
                        break

                # ---- aggregate metrics ------------------------------------
                if inner_metrics:
                    last_metrics = {
                        k: float(sum(m.get(k, 0.0) for m in inner_metrics) / len(inner_metrics))
                        for k in inner_metrics[0]
                    }
                    last_metrics["mean_reward"] = float(rewards.mean())

                self.bus.dispatch("on_reward_computed", rewards=rewards, ctx=self.ctx)

                self.ctx.step += 1
                self.ctx.global_step = self.ctx.step

                if self.ctx.step % self.log_every == 0 and self.logger is not None and self._is_main():
                    scalar = {
                        k: v for k, v in last_metrics.items()
                        if isinstance(v, float) and math.isfinite(v)
                    }
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
                pass
            raise
        finally:
            try:
                self.bus.dispatch("on_train_end", trainer=self, ctx=self.ctx, metrics=last_metrics)
            except Exception:  # noqa: BLE001
                pass
            if self.logger is not None:
                try:
                    self.logger.flush()
                except Exception:  # noqa: BLE001
                    pass

        return last_metrics

    # ------------------------------------------------------------------ ppo step

    def _ppo_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch, [
            "input_ids", "log_probs_old", "advantages_buf",
        ], "PPOTrainer")
        self.model.train()

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        log_probs_old = batch["log_probs_old"]
        advantages = batch.get("advantages_buf")
        returns = batch.get("returns_buf")

        # New log-probs from current policy
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        # TODO: Wrap this forward pass in accelerator.autocast() or torch.autocast() in the future for AMP memory efficiency.
        out = self.model(**kwargs)
        if isinstance(out, ModelOutput):
            logits = out.outputs["logits"]
            hidden = out.hidden_states[-1] if out.hidden_states else None
        else:
            logits = out["logits"]
            hidden = None

        # Per-token log-probs
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lp = F.log_softmax(shift_logits, dim=-1)
            log_probs_new = torch.gather(lp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_new = torch.cat(
                [torch.zeros_like(log_probs_new[:, :1]), log_probs_new], dim=1
            )
        else:
            log_probs_new = torch.zeros_like(log_probs_old)

        # value head — compute V(s) from hidden states if available
        values_new: torch.Tensor | None = None
        if self._use_value_head and hidden is not None:
            if self._value_head is None:
                # Lazy init: infer hidden size from tensor
                self._value_head = LinearValueHead(hidden.shape[-1]).to(hidden.device)
                # Register value head params with optimizer
                if hasattr(self.optimizer, "add_param_group"):
                    self.optimizer.add_param_group({"params": list(self._value_head.parameters())})
            values_new = self._value_head(hidden)   # (B, T)

        # Populate ctx for RLUpdateRule (backward/callbacks delegated below)
        self.ctx.extras.update({
            "log_probs_new": log_probs_new,
            "log_probs_old": log_probs_old,
            "advantages": advantages if advantages is not None else torch.zeros_like(log_probs_old[:, 0]),
            "values": values_new if values_new is not None else torch.zeros_like(log_probs_old),
            "returns": returns.unsqueeze(1) if returns is not None else torch.zeros_like(log_probs_old),
            "model": self.model,
        })
        self.ctx.loss_fn = self._loss_fn
        self.ctx.model = self.model
        return self._rl_rule.step(self.model, batch, self.ctx)

    def _compute_buffer_values(self) -> None:
        """Batch-infer response-mean value for all buffered episodes via value head.

        Populates ep.values = (1,) scalar for each episode so all_values() returns
        (N, 1) suitable for the (N, 1)-shaped GAE call.  Noop if value head is not
        yet initialized (first outer step → zero baseline GAE).
        """
        import warnings
        if self._value_head is None or not self._buffer._episodes:
            return
        episodes = self._buffer._episodes
        max_len = max(ep.input_ids.size(0) for ep in episodes)

        ids_list, mask_list = [], []
        for ep in episodes:
            T = ep.input_ids.size(0)
            pad = max_len - T
            ids_list.append(torch.cat([ep.input_ids,
                                       torch.zeros(pad, dtype=ep.input_ids.dtype)]))
            mask_list.append(torch.cat([torch.ones(T, dtype=torch.long),
                                        torch.zeros(pad, dtype=torch.long)]))
        input_ids = torch.stack(ids_list).to(self.device)
        attn_mask = torch.stack(mask_list).to(self.device)

        self.model.eval()
        with torch.no_grad():
            try:
                out = self.model(input_ids=input_ids, attention_mask=attn_mask,
                                 output_hidden_states=True)
            except TypeError:
                out = self.model(input_ids=input_ids, attention_mask=attn_mask)

            if isinstance(out, ModelOutput):
                hidden = out.hidden_states[-1] if out.hidden_states else None
            else:
                hidden = None

            if hidden is None:
                warnings.warn(
                    "PPOTrainer: value head enabled but model did not return hidden_states. "
                    "Falling back to zero baseline. Set output_hidden_states=True in model config.",
                    stacklevel=2,
                )
                self.model.train()
                return

            token_vals = self._value_head(hidden)   # (N, max_len)
            for i, ep in enumerate(episodes):
                T = ep.input_ids.size(0)
                vals = token_vals[i, :T]
                resp_mask = (ep.labels != -100).float().to(vals.device)
                denom = resp_mask.sum().clamp(min=1)
                ep.values = ((vals * resp_mask).sum() / denom).unsqueeze(0).cpu()
        self.model.train()

    def _step(self, batch: dict[str, Any]) -> StepOutput:  # type: ignore[override]
        """Bridge to _ppo_step() for the unified train_step() protocol.

        For PPOTrainer, train_step() / _step() corresponds to **one inner-epoch
        minibatch policy update**, not one outer rollout step.  The outer rollout
        loop, GAE computation, and inner-epoch loop all remain in fit().
        """
        self.ctx.extras.pop("loss_signal", None)
        raw = self._ppo_step(batch)
        return StepOutput(loss=raw.get("loss"), metrics=dict(raw))

    def eval(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        return []


__all__ = ["LinearValueHead", "PPOTrainer"]
