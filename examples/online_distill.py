"""Minimal two-model online distillation trainer — a shippable template.

This is the in-repo example the docs point to for the **multi-model + online-RL**
seam: the shape that papers like On-Policy Distillation / self-rewarding /
actor-critic need, and the one the built-in ``ppo`` / ``grpo`` trainers cannot
host (they assume a *single* model and a sequence-level scalar reward). It is
~160 lines, registered purely through a recipe's ``user_modules:`` — zero core
edits.

What it demonstrates
--------------------
* a TRAINABLE student and a FROZEN, *separate* teacher (not a clone of the
  student), obtained from the recipe's ``models:`` block as
  ``self.models["student"]`` / ``self.models["teacher"]`` — the seam documented
  in docs/registry_and_protocols.md §4.11. The runtime always passes the named
  model set to the trainer; declaring ``models=`` / ``optimizers=`` on the
  ``__init__`` signature is what lets this trainer receive it.
* on-policy rollout: the student autoregressively samples its own completions
  each step (true multinomial sampling — not a stub);
* a *per-token* reward from the teacher and a REINFORCE update that moves the
  student toward the teacher (on-policy distillation, reverse KL).

Run
---
    lighttrain train -c recipes/online_distill_demo.yaml

For real distillation, point the teacher at a pretrained checkpoint (see the
recipe). With a random-init teacher the loop still runs end-to-end and the
monitored ``reverse_kl`` still descends (the student learns to match whatever
fixed teacher distribution it is given) — which is all this template asserts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from lighttrain import register
from lighttrain.protocols import ModelOutput, StepOutput
from lighttrain.registry import get as registry_get
from lighttrain.trainers.base import Trainer


def _logits(out: Any) -> torch.Tensor:
    return out.outputs["logits"] if isinstance(out, ModelOutput) else out["logits"]


@register("trainer", "online_distill")
class OnlineDistillTrainer(Trainer):
    """Student rolls out on-policy; a frozen teacher scores each sampled token;
    a REINFORCE surrogate moves the student toward the teacher."""

    # Explicit signature (no ``**kwargs``): the config resolver filters recipe /
    # runtime kwargs against it, so declaring ``models`` / ``optimizers`` is what
    # lets this trainer receive the named model set (and the frozen teacher).
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
        max_steps: int = 200,
        val_every: int = 0,
        ckpt_every: int = 0,
        log_every: int = 10,
        device: str | torch.device | None = None,
        arch_profile: Any | None = None,
        models: dict[str, Any] | None = None,
        optimizers: dict[str, Any] | None = None,
        # --- this paradigm's own knobs ---
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        grad_clip: float = 1.0,
        teacher: Mapping[str, Any] | None = None,  # fallback if no `models:` teacher
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
            arch_profile=arch_profile,
            models=models,
            optimizers=optimizers,
        )

        # Student = the primary trainable model (the base already set self.model
        # and self.optimizer to the primary entry of the model/optimizer set).
        self.student = self.model if self.model is not None else self.models.get("student")
        if self.student is None:
            raise RuntimeError("OnlineDistillTrainer: no student model was provided.")
        self.device = torch.device(self.device or next(self.student.parameters()).device)
        self.student = self.student.to(self.device)
        self.model = self.student
        self.ctx.model = self.student

        # Teacher = a *separate* frozen model. Primary path: the recipe's
        # ``models:`` block (self.models["teacher"]). Fallback: build it from a
        # ``teacher:`` spec under ``trainer:`` (kept so the template runs even on
        # an older runtime that doesn't inject the model set).
        teacher_model = self.models.get("teacher")
        if teacher_model is None:
            teacher_model = self._build_teacher(teacher)
        self.teacher = teacher_model.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = max(1e-6, float(temperature))
        self.grad_clip = float(grad_clip)
        self._max_len = int(getattr(self.student, "max_seq_len", 128))

    # -- teacher fallback builder (only used when no `models:` teacher) ------
    @staticmethod
    def _build_teacher(cfg: Mapping[str, Any] | None) -> Any:
        if not cfg:
            raise RuntimeError(
                "OnlineDistillTrainer needs a teacher: declare a frozen `teacher` "
                "in the recipe's `models:` block (trainable: false), or pass a "
                "`teacher:` spec under `trainer:`."
            )
        cfg = dict(cfg)
        spec = dict(cfg.get("spec") or cfg.get("profile") or {})
        name = spec.pop("name", "tiny_lm")
        teacher = registry_get("model", name)(**spec)
        ckpt = cfg.get("checkpoint")
        if ckpt:
            teacher.load_state_dict(_load_state_dict(Path(ckpt)), strict=False)
        return teacher

    # -- rollout: true autoregressive multinomial sampling from the student --
    @torch.no_grad()
    def _rollout(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        was_training = self.student.training
        self.student.eval()
        seq = prompt_ids
        for _ in range(self.max_new_tokens):
            if seq.size(1) >= self._max_len:
                break
            logits = _logits(self.student(input_ids=seq))[:, -1, :] / self.temperature
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
        if was_training:
            self.student.train()
        return seq

    # -- one update: teacher-scored REINFORCE over the response tokens -------
    def _step(self, batch: dict[str, Any]) -> StepOutput:
        seq = batch["input_ids"]                              # (B, L) prompt+response
        resp_mask = batch["response_mask"][:, 1:].float()     # (B, L-1) 1 on response targets
        self.student.train()

        student_logits = _logits(self.student(input_ids=seq))[:, :-1, :].float()   # carries grad
        with torch.no_grad():
            teacher_logits = _logits(self.teacher(input_ids=seq))[:, :-1, :].float()  # detached
        targets = seq[:, 1:]                                  # (B, L-1) sampled next token
        n = resp_mask.sum().clamp_min(1.0)

        log_p = (
            F.log_softmax(student_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )  # (B, L-1) grad
        with torch.no_grad():
            log_q = (
                F.log_softmax(teacher_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            )
            # Per-token reward: teacher prefers the sampled token more than the
            # student does. Maximizing it descends the reverse KL D(p_student||q_teacher).
            reward = log_q - log_p.detach()
            baseline = (reward * resp_mask).sum() / n         # variance-reduction baseline
            adv = reward - baseline

        # REINFORCE surrogate (score-function): ascending reward == descending reverse KL.
        loss = -((adv * log_p) * resp_mask).sum() / n
        reverse_kl = float(((log_p.detach() - log_q) * resp_mask).sum() / n)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        if self.grad_clip > 0:
            params = [p for p in self.student.parameters() if p.grad is not None]
            if params:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.grad_clip))
        self.optimizer.step()
        if self.scheduler is not None and getattr(self.scheduler, "step_per_batch", True):
            self.scheduler.step()

        metrics = {
            "loss": float(loss.detach()),
            "reverse_kl": reverse_kl,                          # MONITORED: should descend
            "mean_reward": float((reward * resp_mask).sum() / n),
            "grad_norm": grad_norm,
            "resp_tokens": int(n.item()),
        }
        self.ctx.metrics.update(metrics)
        return StepOutput(loss=metrics["loss"], metrics=metrics)

    # -- training loop ------------------------------------------------------
    def fit(self, *, steps: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        target = int(steps) if steps is not None else self.max_steps
        loader = self.data_module.train_loader()
        it = iter(loader)
        self.bus.dispatch("on_train_start", trainer=self, ctx=self.ctx)
        self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)

        last: dict[str, Any] = {}
        try:
            while self.ctx.step < target and not self._stop_requested:
                try:
                    raw = next(it)
                except StopIteration:
                    self.bus.dispatch("on_epoch_end", epoch=self.ctx.epoch, ctx=self.ctx)
                    self.ctx.epoch += 1
                    it = iter(loader)
                    self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)
                    raw = next(it)

                prompt_ids = raw.get("input_ids", raw.get("prompt_input_ids")).to(self.device)
                prompt_len = prompt_ids.size(1)

                self.bus.dispatch("on_rollout_begin", ctx=self.ctx)
                seq = self._rollout(prompt_ids)
                self.bus.dispatch("on_rollout_end", ctx=self.ctx, episodes=int(seq.size(0)))

                resp_mask = torch.zeros_like(seq, dtype=torch.float)
                resp_mask[:, prompt_len:] = 1.0
                last = dict(self.train_step({"input_ids": seq, "response_mask": resp_mask}).metrics)

                self.ctx.step += 1
                self.ctx.global_step = self.ctx.step

                if self.ctx.step % self.log_every == 0 and self.logger is not None and self._is_main():
                    scalar = {
                        k: v for k, v in last.items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))
                    }
                    self.logger.log_dict(scalar, step=self.ctx.step)
                self._maybe_save(last)
        finally:
            try:
                self.bus.dispatch("on_train_end", trainer=self, ctx=self.ctx, metrics=last)
            except Exception:  # noqa: BLE001
                pass
            if self.logger is not None:
                try:
                    self.logger.flush()
                except Exception:  # noqa: BLE001
                    pass
        return last

    def eval(self, *a: Any, **k: Any) -> dict[str, float]:  # type: ignore[override]
        return {}

    def predict(self, *a: Any, **k: Any) -> list[Any]:  # type: ignore[override]
        return []


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load weights from a lighttrain checkpoint capsule (dir) or a file."""
    if path.is_dir():
        for cand in ("model.safetensors", "model.pt"):
            if (path / cand).exists():
                path = path / cand
                break
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return obj["model"]
    return obj


__all__ = ["OnlineDistillTrainer"]
