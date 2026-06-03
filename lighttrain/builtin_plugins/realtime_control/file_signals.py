"""FileSignalsCallback.

Polls ``<run_dir>/control/`` every ``poll_every`` steps for four
filenames:

* ``lr.json``    — ``{"scale": <float>}`` multiplies every optimizer
                   param-group lr by the scale (one-shot, file removed
                   after read).
* ``stop``       — its presence requests ``Signal.STOP_TRAINING`` on the
                   next step boundary.
* ``eval_now``   — its presence sets ``ctx.extras["force_eval"] = True``;
                   the trainer's ``_maybe_eval`` honors it.
* ``inject.py``  — its presence runs ``exec(code)`` in a small namespace
                   ``{trainer, model, ctx}`` (lab-only).

Every triggered action is recorded under
``ctx.diagnostics["realtime_events"]`` so :func:`write_index_page` can
list them and ``LineageRecorderCallback`` can optionally persist them.
TCP / socket variants are not yet implemented.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lighttrain.callbacks.base import Signal
from lighttrain.registry import register


_FILES = ("lr.json", "stop", "eval_now", "inject.py")


@register("callback", "file_signals")
class FileSignalsCallback:
    """File-based runtime control knobs."""

    def __init__(
        self,
        *,
        control_dir: str | Path | None = None,
        poll_every: int = 10,
        allow_inject: bool = True,
    ) -> None:
        self.control_dir = Path(control_dir) if control_dir is not None else None
        self.poll_every = max(1, int(poll_every))
        self.allow_inject = bool(allow_inject)
        self._trainer: Any = None
        self._ctx: Any = None

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        self._trainer = trainer
        self._ctx = ctx
        if self.control_dir is None:
            rd = getattr(ctx, "run_dir", None) if ctx is not None else None
            if rd is None and trainer is not None:
                rd = getattr(trainer, "_run_dir", None)
            if rd is not None:
                self.control_dir = Path(rd) / "control"
        if self.control_dir is not None:
            self.control_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(
        self,
        *,
        step: int = 0,
        **_: Any,
    ) -> Signal:
        if self.control_dir is None:
            return Signal.CONTINUE
        if int(step) % self.poll_every != 0:
            return Signal.CONTINUE

        events: list[dict[str, Any]] = []
        signal = Signal.CONTINUE

        # lr.json — scale.
        lr_path = self.control_dir / "lr.json"
        if lr_path.exists():
            try:
                payload = json.loads(lr_path.read_text(encoding="utf-8"))
                scale = float(payload.get("scale", 1.0))
                self._apply_lr_scale(scale)
                events.append(
                    {"event": "lr_scale", "scale": scale, "step": int(step), "ts": time.time()}
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                lr_path.unlink()
            except FileNotFoundError:
                pass

        # stop — soft stop.
        stop_path = self.control_dir / "stop"
        if stop_path.exists():
            events.append({"event": "stop", "step": int(step), "ts": time.time()})
            signal = Signal.STOP_TRAINING
            try:
                stop_path.unlink()
            except FileNotFoundError:
                pass

        # eval_now — set flag.
        eval_now = self.control_dir / "eval_now"
        if eval_now.exists():
            events.append(
                {"event": "eval_now", "step": int(step), "ts": time.time()}
            )
            if self._ctx is not None:
                self._ctx.extras["force_eval"] = True
            try:
                eval_now.unlink()
            except FileNotFoundError:
                pass

        # inject.py — exec in tiny namespace.
        inject = self.control_dir / "inject.py"
        if inject.exists() and self.allow_inject:
            try:
                code = inject.read_text(encoding="utf-8")
                ns = {
                    "trainer": self._trainer,
                    "model": getattr(self._trainer, "model", None),
                    "ctx": self._ctx,
                }
                exec(code, ns, ns)  # noqa: S102 — explicit lab tool
                events.append(
                    {"event": "inject", "step": int(step), "ts": time.time()}
                )
            except Exception as exc:  # noqa: BLE001 — never kill on inject error
                events.append(
                    {
                        "event": "inject_error",
                        "step": int(step),
                        "error": str(exc),
                    }
                )
            try:
                inject.unlink()
            except FileNotFoundError:
                pass

        if events and self._ctx is not None:
            log = self._ctx.diagnostics.setdefault("realtime_events", [])
            log.extend(events)
            # Optional lineage write — single bookkeeping update.
            store = getattr(self._ctx, "lineage_store", None)
            run_node = getattr(self._trainer, "_run_node_id", None)
            if store is not None and run_node is not None and hasattr(store, "update_node_payload"):
                try:
                    store.update_node_payload(int(run_node), {"realtime_events": log})
                except Exception:  # noqa: BLE001
                    pass

        return signal

    # ----- internals -------------------------------------------------------

    def _apply_lr_scale(self, scale: float) -> None:
        if not (scale > 0):
            return
        optimizer = getattr(self._ctx, "optimizer", None) if self._ctx else None
        if optimizer is None and self._trainer is not None:
            optimizer = getattr(self._trainer, "optimizer", None)
        if optimizer is None:
            return
        inner = getattr(optimizer, "optimizer", optimizer)
        groups = getattr(inner, "param_groups", None)
        if groups:
            for g in groups:
                if "lr" in g:
                    g["lr"] = float(g["lr"]) * float(scale)
        scheduler = getattr(self._ctx, "scheduler", None) if self._ctx else None
        if scheduler is None and self._trainer is not None:
            scheduler = getattr(self._trainer, "scheduler", None)
        # Schedulers cache base_lrs (LinearScheduler / WarmupCosine etc.).
        base_lrs = getattr(scheduler, "base_lrs", None)
        if isinstance(base_lrs, list):
            scheduler.base_lrs = [float(lr) * float(scale) for lr in base_lrs]


__all__ = ["FileSignalsCallback"]
