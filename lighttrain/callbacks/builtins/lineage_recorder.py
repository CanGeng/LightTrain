"""LineageRecorderCallback.

Translates Trainer / engine events into LineageStore writes so the SQLite
graph reflects checkpoints / artifacts / run metadata produced during a run.

Designed to be a **critical** callback (see :class:`EventBus`): if the SQLite
file can't be opened the user wants to know immediately, not after 12 hours.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...lineage.dag import apply_cycle_policy, cycle_check
from ...lineage.store import LineageStore
from ...registry import register


@register("callback", "lineage_recorder")
class LineageRecorderCallback:
    """Bridge between Trainer events and :class:`LineageStore`.

    Reads its store + run-id off the :class:`StepContext` at train start. If
    ``ctx.lineage_store`` is absent (e.g. user disabled it) the callback
    silently no-ops — lineage is a soft dependency.
    """

    critical: bool = True

    def __init__(
        self,
        *,
        cycle_policy: str = "warn",  # allowed | warn | forbid
        cycle_depth: int = 4,
        require_external_signal: bool = False,
    ) -> None:
        self.cycle_policy = cycle_policy
        self.cycle_depth = int(cycle_depth)
        self.require_external_signal = bool(require_external_signal)
        self._store: LineageStore | None = None
        self._run_node_id: int | None = None
        self._run_id: str | None = None

    # ----- lifecycle -------------------------------------------------------

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        store: LineageStore | None = getattr(ctx, "lineage_store", None)
        if store is None:
            return
        self._store = store
        self._run_id = str(getattr(ctx, "run_id", "") or "unknown")
        # Use the stable ``run_id`` itself as version so on_train_end can
        # update the SAME row instead of creating a parallel run node with
        # a different (timestamp-based) version key.
        self._run_node_id = store.upsert_node(
            kind="run",
            name=self._run_id,
            version=self._run_id,
            run_id=self._run_id,
            schema_kind="run_meta",
            schema_version="0.4",
            payload_path=str(getattr(trainer, "_run_dir", "") or ""),
            payload={"started_ts": time.time()},
        )

    def on_train_end(self, *, ctx: Any = None, metrics: Any = None, **_: Any) -> None:
        if self._store is None or self._run_node_id is None:
            return
        # Update the existing node by id (merging payload) — do NOT upsert
        # with a different version, which would create a parallel row.
        self._store.update_node_payload(
            self._run_node_id,
            {"ended_ts": time.time(), "final_metrics": _safe_metrics(metrics)},
        )

    # ----- checkpoint --------------------------------------------------

    def on_save_checkpoint_post(
        self,
        *,
        step: int | None = None,
        path: Any = None,
        manifest: Any = None,
        **_: Any,
    ) -> None:
        if self._store is None or path is None:
            return
        # Do not swallow lineage errors — EventBus critical semantics will
        # take the trainer down if this callback is critical, which is the
        # correct behaviour for the registered lineage_recorder.
        ckpt_node = self._store.upsert_node(
            kind="checkpoint",
            name=str(self._run_id),
            version=f"step_{step}" if step is not None else None,
            run_id=self._run_id,
            step=int(step) if step is not None else None,
            schema_kind="checkpoint_manifest",
            schema_version="0.4",
            payload_path=str(path),
            payload=dict(manifest) if isinstance(manifest, dict) else None,
        )
        if self._run_node_id is not None:
            self._store.add_edge(
                self._run_node_id, ckpt_node, "produced_by", {"step": step}
            )

    # ----- artifact --------------------------------------------------------

    def on_artifact_finalized(
        self,
        *,
        path: Any = None,
        step: int | None = None,
        artifact_node: int | None = None,
        **_: Any,
    ) -> None:
        if self._store is None:
            return
        # Most producers handle their own upsert; this is a fallback path used
        # when an external producer fires the event without doing the write.
        if artifact_node is None and path is not None:
            name = Path(str(path)).parent.name
            artifact_node = self._store.upsert_node(
                kind="artifact",
                name=name,
                version=f"step_{step}" if step is not None else None,
                payload_path=str(path),
            )
        if artifact_node is not None and self._run_node_id is not None:
            self._store.add_edge(
                self._run_node_id, int(artifact_node), "produced_by", {"step": step}
            )
            # Cycle check: does this artifact ancestry loop back?
            if self._run_id is not None:
                hits = cycle_check(
                    self._store,
                    int(artifact_node),
                    current_run_id=self._run_id,
                    k=self.cycle_depth,
                )
                apply_cycle_policy(
                    hits,
                    self_feeding=self.cycle_policy,
                    require_external_signal=self.require_external_signal,
                    external_signal_present=False,
                )

    def on_artifact_new_version(self, *, path: Any = None, step: int | None = None, **_: Any) -> None:
        # Forward to the finalize handler — same lineage write.
        self.on_artifact_finalized(path=path, step=step)

    # ----- exception ------------------------------------------------------

    def on_exception(
        self,
        *,
        trainer: Any = None,
        exception: BaseException | None = None,
        step: int | None = None,
        **_: Any,
    ) -> None:
        """Record an unhandled training exception as a ``frozen_step`` node.

        Why ``frozen_step`` and not a new ``crash`` kind: the SQL schema has
        five node kinds plus a reserved ``frozen_step``. A crash bundle *is*
        an unscheduled frozen step with extra ``traceback.txt`` / ``env.json``
        payloads, so we reuse the kind to stay inside the reverse-compat
        schema envelope.
        """
        if self._store is None or self._run_node_id is None:
            return
        try:
            crash_node = self._store.upsert_node(
                kind="frozen_step",
                name=str(self._run_id),
                version=f"crash_step_{int(step) if step is not None else 0}",
                run_id=str(self._run_id),
                step=int(step) if step is not None else None,
                schema_kind="frozen_step",
                schema_version="0.4",
                payload={
                    "reason": "exception",
                    "exc_type": type(exception).__name__ if exception else "Unknown",
                    "exc_str": str(exception) if exception else "",
                    "ts": time.time(),
                },
            )
            if crash_node:
                self._store.add_edge(
                    self._run_node_id, int(crash_node), "produced_by",
                    {"reason": "exception", "step": step},
                )
        except Exception:  # noqa: BLE001 — never mask the original crash
            pass


def _safe_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            try:
                out[str(k)] = str(v)
            except Exception:
                continue
    return out


__all__ = ["LineageRecorderCallback"]
