"""Shared offline preference training pipeline.

All offline preference trainers (DPO / IPO / SimPO / ORPO / KTO) inherit
from :class:`PreferenceTrainer`.  The mixin handles:

1. Double forward pass on concatenated (chosen, rejected) sequences.
2. Extraction of per-sample mean log-probs and NLL for chosen sequences.
3. Reference log-prob injection from artifact batch keys.
4. Backward + optimizer step via the engine's UpdateRule.

Batch keys expected from the DataModule / collator:

    chosen_input_ids        (B, T)
    chosen_attention_mask   (B, T)
    chosen_labels           (B, T)   — padding = -100
    rejected_input_ids      (B, T)
    rejected_attention_mask (B, T)
    rejected_labels         (B, T)

Reference log-probs (optional; skipped by SimPO/ORPO):

    aux.<ref_namespace>.chosen_logprobs    (B,)
    aux.<ref_namespace>.rejected_logprobs  (B,)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..callbacks.base import Signal
from ..protocols import ModelOutput, StepOutput
from ..registry import register
from ..update_rules.rl import RLUpdateRule
from ..utils.seed import restore_rng_state, rng_state
from ._utils import _device_of, _move_batch, validate_batch
from .base import Trainer

_log = logging.getLogger(__name__)


def _seq_logps_and_nll(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return (mean_logps, mean_nll) per sample.

    Returns
    -------
    mean_logps : (B,)  — mean per-token log-prob (signed negative NLL mean)
    mean_nll   : (B,)  — mean per-token NLL (positive)
    """
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    out = model(**kwargs)
    if isinstance(out, ModelOutput):
        logits = out.outputs["logits"]
    elif isinstance(out, Mapping):
        logits = out["logits"]
    else:
        logits = out

    B = input_ids.size(0)
    # Causal shift: predict t+1 from t context.
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_ids = shift_labels.clamp(min=0)
    gathered = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    mask = (shift_labels != ignore_index).float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    mean_logps = (gathered * mask).sum(dim=-1) / denom      # (B,)
    mean_nll = (-gathered * mask).sum(dim=-1) / denom       # (B,)
    return mean_logps, mean_nll


@register("trainer", "preference")
class PreferenceTrainer(Trainer):
    """The single offline preference trainer (DPO / IPO / SimPO / ORPO / KTO).

    The preference *algorithm* is now the ``loss:`` seam, not the trainer
    identity: pick it with ``loss: {name: dpo|ipo|simpo|orpo|kto, ...}``. This
    trainer does the shared double forward (chosen + rejected), enriches the
    batch with per-sample log-probs / NLL / reference log-probs, then hands off
    to ``ctx.loss_fn`` via the RL update rule. It never overwrites a
    recipe-provided loss.
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
        ref_namespace: str = "ref",
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
        self.val_every = int(val_every)
        self.ckpt_every = int(ckpt_every)
        self.log_every = max(1, int(log_every))
        self.ref_namespace = str(ref_namespace)
        self.ignore_index = int(ignore_index)

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

        self._stop_requested = False
        self._rl_rule = RLUpdateRule(grad_clip=grad_clip)

    # ------------------------------------------------------------------ fit

    def fit(self, *, steps: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}.fit: model is not set.")
        if self.optimizer is None:
            raise RuntimeError(f"{type(self).__name__}.fit: optimizer is not set.")
        if self.ctx.loss_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit: no preference loss configured. "
                "Set `loss: {name: dpo|ipo|simpo|orpo|kto, ...}` in the recipe."
            )

        target = int(steps) if steps is not None else self.max_steps
        loader = self.data_module.train_loader()
        iterator = iter(loader)

        self.bus.dispatch("on_train_start", trainer=self, ctx=self.ctx)
        self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)

        last_metrics: dict[str, Any] = {}
        last_batch: Any = None
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
                last_batch = batch

                self.bus.dispatch(
                    "on_train_batch_start", step=self.ctx.step, batch=batch, ctx=self.ctx
                )
                step_out = self.train_step(batch)
                metrics = step_out.metrics

                loss_sig_int = int(self.ctx.extras.get("loss_signal", 0))
                if loss_sig_int == int(Signal.STOP_TRAINING):
                    self._stop_requested = True

                sig = self.bus.dispatch(
                    "on_train_batch_end",
                    step=self.ctx.step,
                    batch=batch,
                    metrics=metrics,
                    ctx=self.ctx,
                )
                if sig == Signal.STOP_TRAINING:
                    self._stop_requested = True

                self.ctx.step += 1
                self.ctx.global_step = self.ctx.step
                last_metrics = dict(metrics)

                self._maybe_log(metrics)
                self._maybe_eval()
                self._maybe_save(metrics)

        except BaseException as exc:  # noqa: BLE001
            try:
                self.bus.dispatch(
                    "on_exception", trainer=self, exception=exc,
                    step=self.ctx.step, batch=last_batch,
                )
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed secondary exception in on_exception dispatch", exc_info=True)
            raise
        finally:
            try:
                self.bus.dispatch(
                    "on_train_end", trainer=self, ctx=self.ctx, metrics=last_metrics
                )
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed exception in on_train_end dispatch", exc_info=True)
            if self.logger is not None:
                try:
                    self.logger.flush()
                except Exception:  # noqa: BLE001
                    _log.warning("Suppressed exception in logger.flush", exc_info=True)

        return last_metrics

    # ------------------------------------------------------------------ core step

    def _preference_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute preference loss and update parameters."""
        validate_batch(batch, [
            "chosen_input_ids", "chosen_labels",
            "rejected_input_ids", "rejected_labels",
        ], "PreferenceTrainer")
        self.model.train()

        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch.get("chosen_attention_mask")
        chosen_labels = batch["chosen_labels"]
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch.get("rejected_attention_mask")
        rejected_labels = batch["rejected_labels"]

        # TODO: Wrap this forward pass in accelerator.autocast() or torch.autocast() in the future for AMP memory efficiency.
        # Forward on chosen
        chosen_logps, chosen_nll = _seq_logps_and_nll(
            self.model, chosen_ids, chosen_mask, chosen_labels,
            ignore_index=self.ignore_index,
        )
        # Forward on rejected
        rejected_logps, _ = _seq_logps_and_nll(
            self.model, rejected_ids, rejected_mask, rejected_labels,
            ignore_index=self.ignore_index,
        )

        # Read reference log-probs from artifact batch (if present)
        ref_chosen_key = f"aux.{self.ref_namespace}.chosen_logprobs"
        ref_rejected_key = f"aux.{self.ref_namespace}.rejected_logprobs"
        enriched = dict(batch)
        enriched["chosen_logps"] = chosen_logps
        enriched["rejected_logps"] = rejected_logps
        enriched["chosen_nll_loss"] = chosen_nll
        if ref_chosen_key in batch:
            enriched["ref_chosen_logps"] = batch[ref_chosen_key].to(chosen_logps.device)
        if ref_rejected_key in batch:
            enriched["ref_rejected_logps"] = batch[ref_rejected_key].to(chosen_logps.device)

        # Populate ctx for RLUpdateRule (backward/callbacks delegated below).
        # enriched (not batch) is passed so preference losses can read
        # chosen/rejected logps. The loss is the recipe-provided ``ctx.loss_fn``
        # (the ``loss:`` seam) — never overwritten here.
        if self.ctx.loss_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}: no preference loss configured. "
                "Set `loss: {name: dpo|ipo|simpo|orpo|kto, ...}` in the recipe."
            )
        self.ctx.extras["model"] = self.model
        self.ctx.model = self.model
        return self._rl_rule.step(self.model, enriched, self.ctx)

    def _step(self, batch: dict[str, Any]) -> StepOutput:  # type: ignore[override]
        """Bridge to _preference_step() for the unified train_step() protocol.

        Clears ``ctx.extras["loss_signal"]`` before the preference step so that
        stale signals from the previous iteration do not persist.
        """
        self.ctx.extras.pop("loss_signal", None)
        raw = self._preference_step(batch)
        return StepOutput(loss=raw.get("loss"), metrics=dict(raw))

    # ------------------------------------------------------------------ helpers

    def eval(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        return []

    def _maybe_log(self, metrics: Mapping[str, Any]) -> None:
        if not self._is_main():
            return
        if self.logger is None or not metrics:
            return
        if self.ctx.step % self.log_every != 0:
            return
        scalar_only = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }
        if scalar_only:
            self.logger.log_dict(scalar_only, step=self.ctx.step)

    def _maybe_eval(self) -> None:
        if self.val_every <= 0:
            return
        if self.ctx.step % self.val_every != 0:
            return
        self.eval()

    def _maybe_save(self, metrics: Mapping[str, Any]) -> None:
        if self.ckpt_manager is None or self.ckpt_every <= 0:
            return
        if self.ctx.step % self.ckpt_every != 0:
            return
        self.ckpt_manager.save(
            step=self.ctx.step,
            state={"model": self.model.state_dict(), "trainer": self.state_dict()},
            kind="step",
            extras={"metrics": dict(metrics)},
            parallel_ctx=self._pctx,
        )

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["ref_namespace"] = self.ref_namespace
        return sd


__all__ = ["PreferenceTrainer"]
