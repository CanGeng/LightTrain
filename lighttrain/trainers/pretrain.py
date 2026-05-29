"""PretrainTrainer — the canonical causal-LM pretraining loop.

Flow::

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
flags ``should_save`` after a step/eval, the trainer triggers a ``best``
save.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import torch

from ..callbacks.base import Signal
from ..protocols import StepOutput
from ..registry import register
from ..utils.seed import restore_rng_state, rng_state
from ._utils import _device_of, _move_batch
from .base import Trainer

_log = logging.getLogger(__name__)


@register("trainer", "pretrain")
class PretrainTrainer(Trainer):
    """Single-GPU causal-LM pretraining trainer."""

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
        arch_profile: Any | None = None,
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
        # optional ArchitectureProfile for stateful architectures (RWKV/Mamba)
        self.arch_profile = arch_profile

    # ---- lifecycle --------------------------------------------------------

    def fit(self, *, steps: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        if self.model is None:
            raise RuntimeError("PretrainTrainer.fit: model is not set.")
        if self.optimizer is None:
            raise RuntimeError("PretrainTrainer.fit: optimizer is not set.")

        target_steps = int(steps) if steps is not None else self.max_steps

        loader = self.data_module.train_loader()
        iterator = iter(loader)

        self.bus.dispatch("on_train_start", trainer=self, ctx=self.ctx)
        self.bus.dispatch("on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx)

        last_metrics: dict[str, Any] = {}
        last_batch: Any = None
        try:
            while self.ctx.step < target_steps and not self._stop_requested:
                try:
                    raw_batch = next(iterator)
                except StopIteration:
                    self.bus.dispatch("on_epoch_end", epoch=self.ctx.epoch, ctx=self.ctx)
                    self.ctx.epoch += 1
                    iterator = iter(loader)
                    self.bus.dispatch(
                        "on_epoch_begin", epoch=self.ctx.epoch, ctx=self.ctx
                    )
                    raw_batch = next(iterator)

                batch = _move_batch(raw_batch, self.device)
                last_batch = batch

                # Stateful architecture support (RWKV / Mamba):
                # On document boundary (batch["_doc_boundary"] == True), reset
                # the recurrent state so the model doesn't bleed context across
                # documents.  The model stores state internally; the batch flag
                # propagates the reset signal without changing the UpdateRule.
                if (
                    self.arch_profile is not None
                    and getattr(self.arch_profile, "state_mode", "stateless") == "stateful"
                    and batch.get("_doc_boundary", False)
                    and self.arch_profile.reset_state_fn is not None
                    and self.model is not None
                ):
                    self.arch_profile.reset_state_fn(self.model)
                    batch["_reset_state"] = True

                self.bus.dispatch(
                    "on_train_batch_start",
                    step=self.ctx.step,
                    batch=batch,
                    ctx=self.ctx,
                )

                step_out = self.train_step(batch)
                metrics = step_out.metrics

                # Honor signals raised inside engine.step (post-review fix):
                # STOP_TRAINING from ``on_loss_computed`` must stop the loop,
                # not silently collapse into a skipped step.
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
                self._maybe_save_best(metrics)

            self._final_save(last_metrics)
        except BaseException as exc:  # noqa: BLE001 — top-level crash hook
            # Any unhandled exception ⇒ dispatch ``on_exception`` so callbacks
            # (lineage_recorder, frozen_step) can react, write a crash bundle,
            # then re-raise so the user / CI sees the original error.
            try:
                self.bus.dispatch(
                    "on_exception",
                    trainer=self,
                    exception=exc,
                    step=self.ctx.step,
                    batch=last_batch,
                )
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed secondary exception in on_exception dispatch", exc_info=True)
            self._write_crash_bundle(exc, last_batch, last_metrics)
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
            # Always emit the run failure-entry page. Soft —
            # never let index generation interfere with crash propagation.
            try:
                from ..diagnostics.index_page import write_index_page

                rd = getattr(self, "_run_dir", None)
                if rd is not None:
                    write_index_page(rd, bus=self.bus)
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed exception in write_index_page", exc_info=True)

        return last_metrics

    def _step(self, batch: dict[str, Any]) -> StepOutput:  # type: ignore[override]
        """Delegate one training step to the engine.

        Clears ``ctx.extras["loss_signal"]`` before calling the engine so that
        stale signals from the previous step do not linger.  StandardUpdateRule
        writes to this slot inside ``on_loss_computed``; resetting it here keeps
        the slot semantics correct when ``train_step()`` is called from fit().
        """
        self.ctx.extras.pop("loss_signal", None)
        raw = self.engine.step(batch, self.ctx)
        return StepOutput(loss=raw.get("loss"), metrics=dict(raw))

    def _write_crash_bundle(
        self,
        exc: BaseException,
        last_batch: Any,
        last_metrics: Mapping[str, Any],
    ) -> None:
        """Drop a crash bundle under ``<run_dir>/diagnostics/crash_<ts>/``;
        also drop an OOM report if the exception looks like a CUDA OOM."""
        if not self._is_main():
            return
        rd = getattr(self, "_run_dir", None)
        if rd is None:
            return
        try:
            from ..diagnostics.crash_bundle import write_crash_bundle

            tokenizer = None
            if self.data_module is not None:
                tokenizer = getattr(self.data_module, "tokenizer", None)
            write_crash_bundle(
                rd,
                exception=exc,
                step=int(self.ctx.step),
                model=self.model,
                batch=last_batch if isinstance(last_batch, dict) else None,
                optimizer=self.optimizer,
                metrics=last_metrics,
                tokenizer=tokenizer,
            )
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed exception in write_crash_bundle", exc_info=True)
        # OOM-specific report — non-fatal.
        try:
            from ..diagnostics.oom_report import is_oom_exception, write_oom_report

            if is_oom_exception(exc):
                write_oom_report(rd, exception=exc)
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed exception in write_oom_report", exc_info=True)

    # ---- evaluation -------------------------------------------------------

    def eval(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        _ = (args, kwargs)
        if self.model is None:
            raise RuntimeError("PretrainTrainer.eval: model is not set.")
        val_loader = self.data_module.val_loader() if self.data_module is not None else None
        if val_loader is None:
            return {}

        loss_fn = self.ctx.loss_fn
        if loss_fn is None:
            return {}

        from ..protocols import LossContext, ModelOutput

        self.bus.dispatch("on_eval_begin", ctx=self.ctx)
        self.model.eval()
        total_loss = 0.0
        n = 0
        try:
            with torch.no_grad():
                for raw in val_loader:
                    batch = _move_batch(raw, self.device)
                    self.bus.dispatch("on_eval_batch_start", batch=batch, ctx=self.ctx)
                    out = self.model(**batch)
                    if not isinstance(out, ModelOutput):
                        out = ModelOutput(
                            outputs=dict(out)
                            if isinstance(out, Mapping)
                            else {"logits": out}
                        )
                    loss_dict = loss_fn(
                        out, batch, LossContext(step=self.ctx.step, epoch=self.ctx.epoch)
                    )
                    val = loss_dict.get("loss")
                    if isinstance(val, torch.Tensor):
                        total_loss += float(val.detach().item())
                        n += 1
                    self.bus.dispatch("on_eval_batch_end", batch=batch, ctx=self.ctx)
        finally:
            self.model.train()

        metrics = {"val_loss": (total_loss / n) if n else math.nan}
        self.ctx.metrics.update(metrics)
        self.bus.dispatch("on_eval_end", metrics=metrics, ctx=self.ctx)
        if self.logger is not None:
            self.logger.log_dict(metrics, step=self.ctx.step)
        return metrics

    def predict(
        self,
        *,
        loader: Any | None = None,
        return_outputs: bool = True,
    ) -> list[dict[str, Any]]:  # type: ignore[override]
        """Run model.forward in eval mode over a predict_loader.

        Implements the ``BaseTrainer`` surface promise of ``fit / eval / predict``.
        Returns a list of per-batch ``ModelOutput.outputs`` dicts (each batch
        becomes one dict of CPU tensors). No new callback events are introduced.
        """
        if self.model is None:
            raise RuntimeError("PretrainTrainer.predict: model is not set.")
        if loader is None:
            if self.data_module is None:
                raise RuntimeError(
                    "PretrainTrainer.predict: no loader and no data_module."
                )
            loader = self.data_module.predict_loader()
        if loader is None:
            raise RuntimeError(
                "PretrainTrainer.predict: data_module.predict_loader() returned None."
            )

        from ..protocols import ModelOutput

        self.model.eval()
        results: list[dict[str, Any]] = []
        try:
            with torch.no_grad():
                for raw in loader:
                    batch = _move_batch(raw, self.device)
                    out = self.model(**batch)
                    if not isinstance(out, ModelOutput):
                        out = ModelOutput(
                            outputs=dict(out)
                            if isinstance(out, Mapping)
                            else {"logits": out}
                        )
                    if return_outputs:
                        cpu_outputs = {
                            k: v.detach().cpu()
                            for k, v in out.outputs.items()
                            if isinstance(v, torch.Tensor)
                        }
                        results.append(cpu_outputs)
        finally:
            self.model.train()
        return results

    # ---- helpers ----------------------------------------------------------

    def _maybe_log(self, metrics: Mapping[str, Any]) -> None:
        if not self._is_main():
            return
        if self.logger is None or not metrics:
            return
        if self.ctx.step % self.log_every != 0:
            return
        scalar_only = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        }
        if scalar_only:
            self.logger.log_dict(scalar_only, step=self.ctx.step)

    def _maybe_eval(self) -> None:
        # ``ctx.extras["force_eval"]`` is set by FileSignalsCallback
        # when ``<run_dir>/control/eval_now`` was touched.
        forced = bool(self.ctx.extras.pop("force_eval", False))
        if not forced:
            if self.val_every <= 0:
                return
            if self.ctx.step % self.val_every != 0:
                return
        self.eval()

    def _save_with_events(
        self,
        *,
        kind: str,
        extras: Mapping[str, Any] | None = None,
    ) -> Any:
        """Save a checkpoint and dispatch on_save_checkpoint_{pre,post}.

        These events are how LineageRecorder and other callbacks learn about
        new checkpoint directories. ``path`` is the directory returned by
        CheckpointManager.save().
        """
        if self.ckpt_manager is None:
            return None
        self.bus.dispatch(
            "on_save_checkpoint_pre",
            trainer=self,
            step=self.ctx.step,
            kind=kind,
        )
        path = self.ckpt_manager.save(
            step=self.ctx.step,
            state=self._collect_state(),
            kind=kind,
            extras=dict(extras or {}),
            parallel_ctx=self._pctx,
        )
        # Re-read manifest so callbacks can index by content without doing IO.
        manifest: dict[str, Any] | None = None
        try:
            import json

            from pathlib import Path as _Path

            mf = _Path(str(path)) / "manifest.json"
            if mf.exists():
                manifest = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _log.warning("Failed to re-read manifest.json after save", exc_info=True)
            manifest = None
        self.bus.dispatch(
            "on_save_checkpoint_post",
            trainer=self,
            step=self.ctx.step,
            kind=kind,
            path=path,
            manifest=manifest,
        )
        return path

    def _maybe_save(self, metrics: Mapping[str, Any]) -> None:
        if self.ckpt_manager is None or self.ckpt_every <= 0:
            return
        if self.ctx.step % self.ckpt_every != 0:
            return
        self._save_with_events(kind="step", extras={"metrics": dict(metrics)})

    def _maybe_save_best(self, metrics: Mapping[str, Any]) -> None:
        if self.ckpt_manager is None:
            return
        for cb in self.callbacks:
            if not getattr(cb, "should_save", False):
                continue
            self._save_with_events(
                kind="best",
                extras={
                    "monitor": getattr(cb, "monitor", None),
                    "value": getattr(cb, "last_value", None),
                    "metrics": dict(metrics),
                },
            )
            cb.should_save = False

    def _final_save(self, metrics: Mapping[str, Any]) -> None:
        if self.ckpt_manager is None:
            return
        # Only write a step ckpt if we wouldn't otherwise have one at this step.
        if self.ckpt_every > 0 and self.ctx.step % self.ckpt_every == 0:
            return
        self._save_with_events(
            kind="step", extras={"final": True, "metrics": dict(metrics)}
        )

    def _collect_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"trainer": self.state_dict()}
        if self.model is not None:
            state["model"] = self.model.state_dict()
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler"] = self.scheduler.state_dict()
        # Full RNG capture (python/numpy/torch/cuda).
        try:
            state["rng"] = rng_state()
        except Exception:  # noqa: BLE001
            _log.warning("Failed to capture RNG state for checkpoint", exc_info=True)
        if self.data_module is not None and hasattr(self.data_module, "state_dict"):
            try:
                state["data_module"] = self.data_module.state_dict()
            except Exception:  # noqa: BLE001
                _log.warning("Failed to capture data_module state for checkpoint", exc_info=True)
        return state

    # ---- resume -----------------------------------------------------------

    def load_checkpoint(self, path: Any) -> None:
        if self.ckpt_manager is None:
            raise RuntimeError("PretrainTrainer.load_checkpoint: no ckpt_manager set.")
        self.bus.dispatch("on_load_checkpoint_pre", trainer=self, path=path)
        ckpt = self.ckpt_manager.load(path)
        if "model" in ckpt and self.model is not None:
            self.model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt and self.optimizer is not None and hasattr(
            self.optimizer, "load_state_dict"
        ):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if (
            "scheduler" in ckpt
            and self.scheduler is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "trainer" in ckpt:
            self.load_state_dict(ckpt["trainer"])
        if "data_module" in ckpt and self.data_module is not None and hasattr(
            self.data_module, "load_state_dict"
        ):
            try:
                self.data_module.load_state_dict(ckpt["data_module"])
            except Exception:  # noqa: BLE001
                _log.warning("Failed to restore data_module state from checkpoint", exc_info=True)
        rng = ckpt.get("rng")
        if rng:
            try:
                restore_rng_state(rng)
            except Exception:  # noqa: BLE001
                _log.warning("Failed to restore RNG state from checkpoint", exc_info=True)
        self.bus.dispatch("on_load_checkpoint_post", trainer=self, path=path)


__all__ = ["PretrainTrainer"]
