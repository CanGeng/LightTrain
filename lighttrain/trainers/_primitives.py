"""Public training-loop primitive — the flat loop, re-usable without a Trainer.

``run_train_loop`` is the generic body lifted verbatim from
``PretrainTrainer.fit``: epoch rollover, ``batch_in_epoch`` resume accounting
(BUG-1), the ``on_train_*`` lifecycle events, ``Signal.STOP_TRAINING``
handling, periodic log/eval/checkpoint, the crash bundle, and the ``finally``
teardown (``on_train_end`` / logger flush / index page).

It drives a *trainer object* through a small protocol (``produce_batch`` →
``train_step`` plus the ``_maybe_*`` periodic hooks), so both the built-in flat
``Trainer`` and a user's custom registered trainer call the same code. The
backward half (``apply_update``) lives in ``..update_rules._primitives``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..callbacks.base import Signal

_log = logging.getLogger(__name__)


def forward_with_activations(
    model: Any,
    batch: Any,
    *,
    layers: list[int] | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    """Forward ``model`` returning ``(output, per-layer hidden_states)``.

    The layer-granularity capture primitive for Axis-C (e.g. greedy layer-wise
    distillation): it asks the model for ``output_hidden_states`` and returns
    the tuple of intermediate activations so a paradigm can read teacher targets
    at layer L+1 and the frozen-prefix input at layer L without reaching into
    the model's internals. ``layers`` optionally selects a subset by index.

    The model must accept ``output_hidden_states=True`` and populate
    ``ModelOutput.hidden_states`` (the existing distillation contract).
    """
    out = model(**{**dict(batch), "output_hidden_states": True})
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise ValueError(
            f"{type(model).__name__} returned no hidden_states; "
            "forward_with_activations needs a model that honours "
            "output_hidden_states=True."
        )
    hs = tuple(hs)
    if layers is not None:
        hs = tuple(hs[i] for i in layers)
    return out, hs


def run_train_loop(trainer: Any, *, target_steps: int) -> dict[str, Any]:
    """Run the flat training loop against ``trainer`` for ``target_steps`` steps.

    The trainer must expose: ``bus``, ``ctx``, ``data_module``, ``model``,
    ``optimizer``, ``logger``; the methods ``produce_batch(raw)`` and
    ``train_step(batch)``; and the periodic hooks ``_maybe_log`` / ``_maybe_eval``
    / ``_maybe_save`` / ``_maybe_save_best`` / ``_final_save`` /
    ``_write_crash_bundle``. ``trainer._stop_requested`` is honoured for early
    stop.
    """
    if trainer.model is None:
        raise RuntimeError(f"{type(trainer).__name__}.fit: model is not set.")
    if trainer.optimizer is None:
        raise RuntimeError(f"{type(trainer).__name__}.fit: optimizer is not set.")

    loader = trainer.data_module.train_loader()
    iterator = iter(loader)

    trainer.bus.dispatch("on_train_start", trainer=trainer, ctx=trainer.ctx)
    trainer.bus.dispatch("on_epoch_begin", epoch=trainer.ctx.epoch, ctx=trainer.ctx)

    last_metrics: dict[str, Any] = {}
    last_batch: Any = None
    try:
        while trainer.ctx.step < target_steps and not trainer._stop_requested:
            try:
                raw_batch = next(iterator)
            except StopIteration:
                trainer.bus.dispatch("on_epoch_end", epoch=trainer.ctx.epoch, ctx=trainer.ctx)
                trainer.ctx.epoch += 1
                trainer.ctx.batch_in_epoch = 0  # new epoch → reset data position
                iterator = iter(loader)
                trainer.bus.dispatch(
                    "on_epoch_begin", epoch=trainer.ctx.epoch, ctx=trainer.ctx
                )
                raw_batch = next(iterator)

            # One batch consumed from the loader this epoch (authoritative,
            # prefetch-independent — counts loop `next()`s, not sampler yields).
            # Drives step-exact mid-epoch resume (BUG-1).
            trainer.ctx.batch_in_epoch += 1

            batch = trainer.produce_batch(raw_batch)
            last_batch = batch

            trainer.bus.dispatch(
                "on_train_batch_start",
                step=trainer.ctx.step,
                batch=batch,
                ctx=trainer.ctx,
            )

            step_out = trainer.train_step(batch)
            metrics = step_out.metrics

            # Honor signals raised inside the step: STOP_TRAINING from
            # ``on_loss_computed`` must stop the loop, not silently collapse
            # into a skipped step.
            loss_sig_int = int(trainer.ctx.extras.get("loss_signal", 0))
            if loss_sig_int == int(Signal.STOP_TRAINING):
                trainer._stop_requested = True

            sig = trainer.bus.dispatch(
                "on_train_batch_end",
                step=trainer.ctx.step,
                batch=batch,
                metrics=metrics,
                ctx=trainer.ctx,
            )
            if sig == Signal.STOP_TRAINING:
                trainer._stop_requested = True

            trainer.ctx.step += 1
            trainer.ctx.global_step = trainer.ctx.step
            last_metrics = dict(metrics)

            trainer._maybe_log(metrics)
            trainer._maybe_eval()
            trainer._maybe_save(metrics)
            trainer._maybe_save_best(metrics)

        trainer._final_save(last_metrics)
    except BaseException as exc:  # noqa: BLE001 — top-level crash hook
        # Any unhandled exception ⇒ dispatch ``on_exception`` so callbacks
        # (lineage_recorder, frozen_step) can react, write a crash bundle,
        # then re-raise so the user / CI sees the original error.
        try:
            trainer.bus.dispatch(
                "on_exception",
                trainer=trainer,
                exception=exc,
                step=trainer.ctx.step,
                batch=last_batch,
            )
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed secondary exception in on_exception dispatch", exc_info=True)
        trainer._write_crash_bundle(exc, last_batch, last_metrics)
        raise
    finally:
        try:
            trainer.bus.dispatch(
                "on_train_end", trainer=trainer, ctx=trainer.ctx, metrics=last_metrics
            )
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed exception in on_train_end dispatch", exc_info=True)
        if trainer.logger is not None:
            try:
                trainer.logger.flush()
            except Exception:  # noqa: BLE001
                _log.warning("Suppressed exception in logger.flush", exc_info=True)
        # Always emit the run failure-entry page. Soft — never let index
        # generation interfere with crash propagation.
        try:
            from ..observability.diagnostics.index_page import write_index_page

            rd = getattr(trainer, "_run_dir", None)
            if rd is not None:
                write_index_page(rd, bus=trainer.bus)
        except Exception:  # noqa: BLE001
            _log.warning("Suppressed exception in write_index_page", exc_info=True)

    return last_metrics


__all__ = ["forward_with_activations", "run_train_loop"]
