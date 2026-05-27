"""Dynamic artifact producer callback.

Triggered on training events (typically ``on_step_end``), this callback enqueues
``(model_snapshot, batch_view, version_tag)`` to a background worker thread
that runs the producer's ``produce()`` + ``finalize()`` without blocking the
training step.

Back-pressure: when the queue is full, the new submission is **dropped** and
``ctx.metrics["dynamic_artifact.dropped"]`` is incremented. Tune
``output.queue_size`` to balance memory vs. throughput.

**Known limitations**:
  * Only ``async_mode='thread'`` is supported; ``'process'`` is accepted but
    raises :class:`NotImplementedError`.
  * On-policy version pinning (``pin_model_version=True``) deep-copies the model
    state dict at each trigger. When disabled, ``$self`` forwards a live
    reference, which may race with concurrent training steps.
  * A new artifact version fires ``on_artifact_new_version`` on the bus;
    :class:`ArtifactJoinedDataset` callers must observe it and call
    ``.reload()`` to swap in the updated store.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from ..registry import register
from .producer import ModelForwardProducer


@register("callback", "dynamic_artifact")
class DynamicArtifactCallback:
    """Schedule artifact production from the training loop.

    Parameters
    ----------
    producer : dict
        Spec used to instantiate the producer (typically ``{name: model_forward,
        extras: [...], ...}``). The string ``"$self"`` in ``model`` is replaced
        with ``ctx.model`` at attach time.
    trigger : dict
        ``{event: on_step_end, every_n_steps: 500, condition?: "<expr>"}``.
        ``condition`` is evaluated against ``ctx.metrics`` via ``eval(...)``.
    output : dict
        ``{name: <artifact-name>, version: auto, queue_size: 4,
        async_mode: thread, root: <path>}``.
    """

    critical: bool = False

    def __init__(
        self,
        *,
        producer: Mapping[str, Any],
        trigger: Mapping[str, Any],
        output: Mapping[str, Any] | None = None,
    ) -> None:
        self.producer_spec = dict(producer)
        self.trigger = dict(trigger)
        self.output = dict(output or {})
        self.async_mode = str(self.output.get("async_mode", "thread"))
        if self.async_mode != "thread":
            raise NotImplementedError(
                "dynamic_artifact only supports async_mode='thread'; "
                "'process' mode is not yet implemented."
            )
        self._q: queue.Queue[Any] = queue.Queue(maxsize=int(self.output.get("queue_size", 4)))
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._dropped = 0
        self._produced = 0
        self._last_step = -1
        self._ctx_ref: Any = None

    # ----- lifecycle hooks -------------------------------------------------

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        self._ctx_ref = ctx
        self._worker = threading.Thread(target=self._loop, daemon=True, name="dynamic-artifact")
        self._worker.start()

    def on_train_end(self, *, ctx: Any = None, **_: Any) -> None:
        self._stop.set()
        if self._worker is not None:
            try:
                self._q.put_nowait(None)  # non-blocking poison pill
            except queue.Full:
                pass  # worker will still see _stop on next get() timeout
            self._worker.join(timeout=10.0)
            self._worker = None

    def on_step_end(
        self,
        *,
        step: int,
        batch: Any = None,
        ctx: Any = None,
        metrics: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if self.trigger.get("event", "on_step_end") != "on_step_end":
            return
        every = int(self.trigger.get("every_n_steps", 1))
        if every <= 0 or step % every != 0 or step == self._last_step:
            return
        self._last_step = step
        cond = self.trigger.get("condition")
        if cond:
            try:
                if not eval(  # noqa: S307 — guarded eval, user-supplied
                    str(cond),
                    {"__builtins__": {}},
                    {"metrics": dict(metrics or {}), "step": step},
                ):
                    return
            except Exception:
                return
        submission = {
            "step": int(step),
            "batch": batch,
            "ctx": ctx,
            "version_tag": self._next_version_tag(step),
        }
        try:
            self._q.put_nowait(submission)
        except queue.Full:
            self._dropped += 1
            if ctx is not None and hasattr(ctx, "metrics"):
                ctx.metrics["dynamic_artifact.dropped"] = float(self._dropped)

    # ----- worker loop -----------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                self._produce_one(item)
            except Exception as exc:  # pragma: no cover — logging only
                # Worker faults must not kill training.
                if self._ctx_ref is not None and getattr(self._ctx_ref, "logger", None):
                    try:
                        self._ctx_ref.logger.log_text(
                            f"[dynamic_artifact] worker error: {exc!r}",
                            int(item.get("step", -1)),
                        )
                    except Exception:
                        pass
            finally:
                self._q.task_done()

    def _produce_one(self, item: Mapping[str, Any]) -> None:
        ctx = item.get("ctx")
        step = int(item.get("step", 0))
        spec = self._resolve_spec(ctx)
        spec.setdefault("artifact_version", item.get("version_tag"))
        spec.setdefault("artifact_name", self.output.get("name") or "dynamic_artifact")
        if "store" not in spec:
            base_root = self.output.get("root") or "./runs/dynamic_artifacts"
            spec["store"] = {
                "name": self.output.get("store_backend", "safetensors-shards"),
                "root": str(Path(base_root) / f"{spec['artifact_name']}_{item['version_tag']}"),
            }
        from ..config._resolver import resolve as _resolve
        producer: ModelForwardProducer = _resolve(spec, category="artifact_producer")
        producer.prepare({"lineage_store": getattr(ctx, "lineage_store", None)})
        batch = item.get("batch") or {}
        if isinstance(batch, Mapping):
            samples = _explode_batch(batch)
            for s in samples:
                producer.produce(s)
        manifest = producer.finalize()
        self._produced += 1
        if ctx is not None and hasattr(ctx, "metrics"):
            ctx.metrics["dynamic_artifact.produced"] = float(self._produced)
        bus = getattr(ctx, "bus", None)
        if bus is not None and hasattr(bus, "dispatch"):
            bus.dispatch("on_artifact_new_version", path=str(manifest), step=step)
            bus.dispatch("on_artifact_finalized", path=str(manifest), step=step)

    def _resolve_spec(self, ctx: Any) -> dict[str, Any]:
        spec = dict(self.producer_spec)
        if str(spec.get("model")) == "$self":
            live_model = getattr(ctx, "model", None)
            if live_model is not None and self.output.get("pin_model_version", False):
                # Deep-copy the model state so the worker holds a snapshot
                # and concurrent training steps don't race the producer.
                import copy
                snapshot_model = copy.deepcopy(live_model)
                snapshot_model.eval()
                for p in snapshot_model.parameters():
                    p.requires_grad_(False)
                spec["model"] = snapshot_model
            else:
                spec["model"] = live_model
        return spec

    def _next_version_tag(self, step: int) -> str:
        tag = self.output.get("version", "auto")
        if tag == "auto":
            return f"v{int(time.time())}_step{step}"
        return str(tag)


def _explode_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn a batch dict into a list of per-sample dicts (best-effort)."""
    keys = [k for k in ("input_ids", "attention_mask", "labels") if k in batch]
    if not keys:
        return []
    n = batch[keys[0]].shape[0]
    out: list[dict[str, Any]] = []
    for i in range(int(n)):
        row: dict[str, Any] = {}
        for k in keys:
            t = batch[k]
            row[k] = t[i].detach().cpu()
        row.setdefault("id", f"dyn_{i}")
        out.append(row)
    return out


__all__ = ["DynamicArtifactCallback"]
