"""Trainer base — the flat trainer: owns shared state AND composes the loop.

``base.Trainer`` owns the wiring every paradigm shares (EventBus,
CheckpointManager, LoggerBus, StepContext) and now also a **concrete** ``fit``
that composes the public primitives (``run_train_loop`` + ``apply_update``).
The 90% case (causal-LM pretraining) needs no subclass body at all; a new
paradigm overrides the two seams — ``produce_batch`` (what a batch is) and
``forward_loss`` (the forward + loss) — or, for full control, overrides
``fit`` and calls the same primitives directly.

Override points::

    produce_batch(raw)   # default: move-to-device. RL/OPD: rollout.
    forward_loss(batch)  # default: None → route through engine.step (the
                         #   StandardUpdateRule forward+loss+backward, kept a
                         #   numerical no-op for pretrain). Custom paradigms
                         #   return (loss, metrics) and ride apply_update.
    before_step(batch)   # optional: GAE / group-advantage precompute.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..callbacks.base import EventBus
from ..distributed._context import ParallelContext
from ..engine._context import StepContext
from ..engine.update_rules._primitives import MicroState, apply_update
from ..protocols import LossContext, ModelOutput, StepOutput
from ..utils.seed import restore_rng_state, rng_state
from ._primitives import run_train_loop
from ._utils import _device_of, _move_batch

if TYPE_CHECKING:  # pragma: no cover
    pass

_log = logging.getLogger(__name__)

# Keys that describe the *recipe*, not the run progress. They are written to
# state_dict() for audit, but load_state_dict() never restores them — resuming
# from a step-5 checkpoint with a recipe asking for max_steps=10 must keep
# max_steps=10, otherwise fit() returns immediately with no training (Issue #8).
_RECIPE_CONTROLLED_KEYS: tuple[str, ...] = ("max_steps", "max_epochs")


class Trainer:
    """Flat trainer: shared scaffolding + the composed training loop."""

    # ---- objective-seam contract (class-level declarations) ----------------
    # Whether this trainer drives loss through the canonical ``objective`` seam
    # (``ctx.loss_fn`` = ``self.objective``). Inline-algorithm trainers
    # (reward-model BT, online-distill REINFORCE) compute loss themselves and
    # declare ``consumes_objective = False`` — the runtime then rejects a recipe
    # that hands them a ``loss:``/``objective:``.
    consumes_objective: bool = True
    # Whether this trainer runs ``objective.prepare_batch`` before the forward
    # (the default ``produce_batch`` path does). RL/preference trainers consume
    # the objective as a loss but bring their own batches, so they set this
    # False; the runtime then rejects a *real* ``objective:`` (with a non-trivial
    # prepare) given to them (a plain ``loss:`` is always fine).
    consumes_objective_prepare: bool = True
    # Consuming trainers with no sensible built-in default (preference) set this
    # True; the runtime then errors when a recipe omits both loss and objective.
    requires_objective: bool = False

    def default_objective(self) -> Any:
        """The objective used when a consuming trainer's recipe omits loss/obj.

        Called by the runtime *after* construction (so subclasses can wrap a
        surrogate loss they built in ``__init__``). The abstract base supplies
        **no** concrete default — core does not know any specific loss (DESIGN
        §3.3). A consuming trainer that has a sensible default (e.g.
        ``PretrainTrainer`` → next-token cross-entropy) overrides this; a trainer
        that requires an explicit objective sets ``requires_objective = True`` so
        the runtime errors loudly instead of falling back to ``None``.
        """
        return None

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
        models: dict[str, Any] | None = None,
        optimizers: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.max_steps = int(max_steps)
        self.val_every = int(val_every)
        self.ckpt_every = int(ckpt_every)
        self.log_every = max(1, int(log_every))

        self.callbacks = list(callbacks or [])
        self.bus = EventBus(self.callbacks)
        self.ctx = StepContext()
        self.ctx.bus = self.bus
        self.ctx.optimizer = optimizer
        self.ctx.scheduler = scheduler
        self.ctx.logger = logger

        self.model = model
        if model is not None:
            self.ctx.model = model
        if device is not None:
            self.device: torch.device | None = torch.device(device)
            if model is not None:
                self.model = model.to(self.device)
                self.ctx.model = self.model
        else:
            self.device = _device_of(self.model) if self.model is not None else None

        self._stop_requested = False
        # The canonical training objective. Left None here; the runtime binds it
        # post-construction (recipe-provided or ``default_objective()``) via
        # ``_wire_objective``. Inline-algorithm trainers keep it None.
        self.objective: Any | None = None
        # optional ArchitectureProfile for stateful architectures (RWKV/Mamba)
        self.arch_profile = arch_profile
        # The named model set (Axis A/B). Single-model recipes get
        # ``{"main": model}``; multi-model paradigms (OPD/KD/GAN) read e.g.
        # ``self.models["teacher"]``. Falls back to wrapping the primary model.
        if models:
            self.models = dict(models)
        elif self.model is not None:
            self.models = {"main": self.model}
        else:
            self.models = {}
        # The named optimizer set (Axis B). Single-optimizer recipes get
        # ``{"main": optimizer}``; multi-optimizer paradigms (GAN/actor-critic)
        # read e.g. ``self.optimizers["disc"]`` and drive each via apply_update.
        if optimizers:
            self.optimizers = dict(optimizers)
        elif self.optimizer is not None:
            self.optimizers = {"main": self.optimizer}
        else:
            self.optimizers = {}
        # Gradient-accumulation cursor for the custom forward_loss + apply_update
        # path (the default path routes through engine.step which owns its own).
        self._micro = MicroState()

    # ---- lifecycle (concrete) ---------------------------------------------

    def fit(self, *, steps: int | None = None) -> dict[str, Any]:
        target_steps = int(steps) if steps is not None else self.max_steps
        return run_train_loop(self, target_steps=target_steps)

    # ---- override points --------------------------------------------------

    def _prepare_with_objective(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run ``objective.prepare_batch`` if an objective is attached.

        The single place both ``produce_batch`` (train) and ``eval`` route
        through, so the objective's batch transform (diffusion noise, JEPA patch
        sampling, …) is never duplicated. Presence-based: inline trainers leave
        ``self.objective`` None → identity; ``LossOnlyObjective`` is identity too.
        """
        if self.objective is None:
            return batch
        return self.objective.prepare_batch(
            batch, step=self.ctx.step, device=self.device
        )

    def produce_batch(self, raw: Any) -> dict[str, Any]:
        """Default: move a raw loader batch to the device, reset recurrent state
        at document boundaries, then run the objective's batch preparation.

        Order: move → arch-profile state reset (RWKV/Mamba, on ``_doc_boundary``)
        → ``objective.prepare_batch``. RL/OPD trainers override this with rollout.
        """
        batch = _move_batch(raw, self.device)
        if (
            self.arch_profile is not None
            and getattr(self.arch_profile, "state_mode", "stateless") == "stateful"
            and batch.get("_doc_boundary", False)
            and self.arch_profile.reset_state_fn is not None
            and self.model is not None
        ):
            self.arch_profile.reset_state_fn(self.model)
            batch["_reset_state"] = True
        # ``_doc_boundary`` is trainer-only metadata — drop it so it never reaches
        # ``model(**batch)`` (no reliance on the model tolerating unknown kwargs).
        batch.pop("_doc_boundary", None)
        return self._prepare_with_objective(batch)

    def forward_loss(self, batch: dict[str, Any]) -> Any:
        """Default: ``None`` ⇒ route the whole step through ``engine.step``.

        Returning ``None`` keeps the pretrain path numerically identical to the
        pre-refactor ``StandardUpdateRule.step`` (forward + loss + backward, with
        RETRY_STEP / SKIP_STEP / RNG replay all intact). A custom paradigm
        overrides this to return either a ``loss`` tensor, a ``(loss, metrics)``
        tuple, or a dict containing ``"loss"``; the base then drives the shared
        ``apply_update`` backward half.
        """
        return None

    def before_step(self, batch: dict[str, Any]) -> None:  # noqa: ARG002
        """Optional pre-step precompute hook (e.g. GAE / group advantages)."""
        return None

    # ---- step -------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> StepOutput:
        """Public entry point for a single training step.

        Loops must call this rather than the internal ``_step`` so future
        hook/callback extensions live in one place. The stale ``loss_signal``
        slot is cleared inside the step body (``_run_step`` / a legacy ``_step``
        override).
        """
        return self._normalize_step_output(self._step(batch))

    def _step(self, batch: dict[str, Any]) -> StepOutput | dict[str, Any]:
        """Execute one gradient update and return metrics.

        Default (the flat trainer): defer to ``_run_step`` — ``forward_loss``
        decides whether to route through the engine (pretrain no-op) or the
        custom ``apply_update`` path. Legacy paradigm trainers (ppo/grpo/
        preference/rm, migrated in steps 2–3) override this directly.
        """
        return self._run_step(batch)

    def _run_step(self, batch: dict[str, Any]) -> StepOutput | dict[str, Any]:
        # Clear stale loss_signal from the previous step before the engine /
        # algorithm call (StandardUpdateRule writes it inside on_loss_computed).
        self.ctx.extras.pop("loss_signal", None)
        result = self.forward_loss(batch)
        if result is None:
            # Default path: engine owns forward + loss + backward (no-op for
            # pretrain vs. the pre-refactor _step).
            raw = self.engine.step(batch, self.ctx)
            return StepOutput(loss=raw.get("loss"), metrics=dict(raw))

        # Custom forward path: drive the shared backward half.
        self.before_step(batch)
        loss, metrics = self._split_forward_result(result)
        grad_norm = apply_update(
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
            ctx=self.ctx,
            micro_state=self._micro,
            scheduler=self.scheduler,
            accelerator=getattr(self.ctx, "accelerator", None),
            grad_clip=float(getattr(self, "grad_clip", 1.0)),
            accumulate_grad_batches=int(getattr(self, "accumulate", 1)),
            bus=self.bus,
        )
        metrics = dict(metrics)
        metrics.setdefault("grad_norm", grad_norm)
        return StepOutput(loss=loss, metrics=metrics)

    @staticmethod
    def _split_forward_result(result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            loss, metrics = result
            return loss, dict(metrics or {})
        if isinstance(result, Mapping):
            return result.get("loss"), dict(result)
        # bare loss tensor / scalar
        return result, {"loss": result}

    def _normalize_step_output(self, result: Any) -> StepOutput:
        if isinstance(result, StepOutput):
            return result
        if isinstance(result, dict):
            return StepOutput(loss=result.get("loss"), metrics=dict(result))
        raise TypeError(
            f"_run_step() must return StepOutput or dict, got {type(result).__name__}"
        )

    # ---- evaluation -------------------------------------------------------

    def eval(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        _ = (args, kwargs)
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}.eval: model is not set.")
        val_loader = self.data_module.val_loader() if self.data_module is not None else None
        if val_loader is None:
            return {}

        loss_fn = self.ctx.loss_fn
        if loss_fn is None:
            return {}

        self.bus.dispatch("on_eval_begin", ctx=self.ctx)
        self.model.eval()
        total_loss = 0.0
        n = 0
        try:
            with torch.no_grad():
                for raw in val_loader:
                    batch = _move_batch(raw, self.device)
                    # Trainer-only metadata — never feed it to ``model(**batch)``.
                    batch.pop("_doc_boundary", None)
                    # Same objective batch-prep as training (diffusion noise,
                    # JEPA patches, …) so objective recipes don't KeyError in eval.
                    # Not via produce_batch — that would trigger RL rollout overrides.
                    batch = self._prepare_with_objective(batch)
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
    ) -> list[dict[str, Any]]:
        """Run model.forward in eval mode over a predict_loader.

        Returns a list of per-batch ``ModelOutput.outputs`` dicts (each batch
        becomes one dict of CPU tensors).
        """
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}.predict: model is not set.")
        if loader is None:
            if self.data_module is None:
                raise RuntimeError(
                    f"{type(self).__name__}.predict: no loader and no data_module."
                )
            loader = self.data_module.predict_loader()
        if loader is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict: data_module.predict_loader() returned None."
            )

        self.model.eval()
        results: list[dict[str, Any]] = []
        try:
            with torch.no_grad():
                for raw in loader:
                    batch = _move_batch(raw, self.device)
                    batch.pop("_doc_boundary", None)  # trainer-only metadata
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

    # ---- periodic hooks (driven by run_train_loop) ------------------------

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
        # ``ctx.extras["force_eval"]`` is set by FileSignalsCallback when
        # ``<run_dir>/control/eval_now`` was touched.
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
        """Save a checkpoint and dispatch on_save_checkpoint_{pre,post}."""
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
            mf = Path(str(path)) / "manifest.json"
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
            from ..observability.diagnostics.crash_bundle import write_crash_bundle

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
            from ..observability.diagnostics.oom_report import (
                is_oom_exception,
                write_oom_report,
            )

            if is_oom_exception(exc):
                write_oom_report(rd, exception=exc)
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed exception in write_oom_report", exc_info=True)

    # ---- distributed helpers ----------------------------------------------

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
            "batch_in_epoch": int(self.ctx.batch_in_epoch),
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
        # Absent in pre-v0.1.9 checkpoints → 0 = resume at epoch start (the old
        # epoch-granularity behavior), never worse.
        self.ctx.batch_in_epoch = int(sd.get("batch_in_epoch", 0))

    # ---- resume -----------------------------------------------------------

    def load_checkpoint(self, path: Any) -> None:
        if self.ckpt_manager is None:
            raise RuntimeError(f"{type(self).__name__}.load_checkpoint: no ckpt_manager set.")
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
        # Authoritative mid-epoch seek (BUG-1): position the sampler from the
        # trainer's consumed-batch count, overriding any prefetch-skewed
        # yield-time position. Runs AFTER load_state_dict so it wins.
        if self.data_module is not None and hasattr(self.data_module, "seek"):
            try:
                self.data_module.seek(self.ctx.epoch, self.ctx.batch_in_epoch)
            except Exception:  # noqa: BLE001
                _log.warning("Failed to seek data position on resume", exc_info=True)
        rng = ckpt.get("rng")
        if rng:
            try:
                restore_rng_state(rng)
            except Exception:  # noqa: BLE001
                _log.warning("Failed to restore RNG state from checkpoint", exc_info=True)
        self.bus.dispatch("on_load_checkpoint_post", trainer=self, path=path)


__all__ = ["StepOutput", "Trainer"]
