"""FrozenStepCallback — bridges Trainer events to FrozenStepWriter.

Lifecycle::

    on_train_start  → instantiate FrozenStepWriter under <run_dir>/frozen_steps
                      and attach it to ctx.frozen_step_writer so the
                      StandardUpdateRule's RETRY_STEP path can restore from it.
    on_step_begin   → writer.snapshot(step, ctx, batch, model, optimizer)
    on_step_end     → if step % every == 0 → writer.commit(reason="scheduled")
    on_exception    → writer.commit(reason="exception") if a snapshot exists

This callback is intentionally *non*-critical: a failure to snapshot
must not kill the training run. The cost is silent — checked in
``diagnostics/index.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...diagnostics.frozen_step import FrozenStepWriter
from ...registry import register


@register("callback", "frozen_step")
class FrozenStepCallback:
    """Scheduled frozen step snapshots."""

    def __init__(self, *, every: int = 1000, reason: str = "scheduled") -> None:
        self.every = max(1, int(every))
        self.reason = reason
        self._writer: FrozenStepWriter | None = None
        self._config_yaml: str = ""

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        rd = getattr(ctx, "run_dir", None) if ctx is not None else None
        if rd is None and trainer is not None:
            rd = getattr(trainer, "_run_dir", None)
        if rd is None:
            return
        mode = str(getattr(ctx, "mode", "lab") or "lab")
        lineage = getattr(ctx, "lineage_store", None) if ctx is not None else None
        run_id = getattr(ctx, "run_id", None) if ctx is not None else None
        run_node_id = None
        # If LineageRecorderCallback is also attached we can read its node id
        # so frozen_step nodes hang off the *same* run node.
        if trainer is not None:
            for cb in getattr(trainer, "callbacks", []) or []:
                rn = getattr(cb, "_run_node_id", None)
                if isinstance(rn, int):
                    run_node_id = rn
                    break
        self._writer = FrozenStepWriter(
            Path(rd),
            mode=mode,
            lineage_store=lineage,
            run_node_id=run_node_id,
            run_id=run_id,
        )
        # Expose to ctx so StandardUpdateRule's RETRY_STEP can borrow it.
        if ctx is not None:
            ctx.frozen_step_writer = self._writer
        # Best-effort grab of the resolved YAML if the trainer stashed one.
        self._config_yaml = str(getattr(trainer, "_resolved_yaml", "") or "")

    def on_step_begin(
        self,
        *,
        step: int = 0,
        batch: Any = None,
        ctx: Any = None,
        **_: Any,
    ) -> None:
        if self._writer is None or not isinstance(batch, dict):
            return
        model = getattr(ctx, "model", None)
        optimizer = getattr(ctx, "optimizer", None)
        if model is None or optimizer is None:
            return
        try:
            self._writer.snapshot(
                step=int(step),
                ctx=ctx,
                batch=batch,
                model=model,
                optimizer=optimizer,
                config_resolved_yaml=self._config_yaml,
            )
        except Exception:  # noqa: BLE001
            pass

    def on_step_end(self, *, step: int = 0, **_: Any) -> None:
        if self._writer is None:
            return
        if step <= 0 or int(step) % self.every != 0:
            return
        try:
            self._writer.commit(reason=self.reason)
        except Exception:  # noqa: BLE001
            pass

    def on_exception(self, **_: Any) -> None:
        if self._writer is None:
            return
        try:
            self._writer.commit(reason="exception")
        except Exception:  # noqa: BLE001
            pass


__all__ = ["FrozenStepCallback"]
