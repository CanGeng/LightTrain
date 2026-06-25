"""Coverage tests for DynamicArtifactCallback (dynamic_producer.py).

Pins and exercises every previously-uncovered branch:
  * Line 111  — on_step_end skips when trigger event != 'on_step_end'
  * Line 114  — on_step_end skips when every<=0, step not a multiple, or same step
  * Lines 125,132 — condition evaluates to False → skip; condition raises → warning + skip
  * Lines 152-153 — worker _loop handles queue.Empty (timeout) via continue
  * Line 155   — worker _loop exits on poison-pill (None item)
  * Lines 186-187 — _produce_one builds default 'store' when not in spec
  * Lines 205-206 — bus.dispatch called after produce_one succeeds
  * Lines 215-220 — _resolve_spec pin_model_version=True deep-copies & freezes model
  * Line 229   — _next_version_tag returns 'auto' formatted tag
  * Line 236   — _next_version_tag returns fixed string when version != 'auto'
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.data.artifacts.dynamic_producer import (
    DynamicArtifactCallback,
    _explode_batch,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal 2-layer net; deterministic weights via seed."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.lin = nn.Linear(4, 4)

    def forward(self, **_kw: Any) -> Any:
        from lighttrain.protocols import ModelOutput
        return ModelOutput(outputs={"logits": torch.zeros(1, 4)})


class _FakeBus:
    """Records dispatch calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def dispatch(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


class _FakeCtx:
    """Minimal training context stub."""

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self.metrics: dict[str, Any] = {}
        self.bus: _FakeBus | None = None
        self.lineage_store: Any = None


def _make_cb(
    tmp_path: Path,
    *,
    queue_size: int = 4,
    every_n_steps: int = 1,
    event: str = "on_step_end",
    condition: str | None = None,
    pin_model_version: bool = False,
    version: str = "auto",
) -> DynamicArtifactCallback:
    spec: dict[str, Any] = {
        "name": "model_forward",
        "model": "$self",
        "store": {"name": "safetensors-shards", "root": str(tmp_path / "store"), "shard_size": 4},
    }
    trigger: dict[str, Any] = {"event": event, "every_n_steps": every_n_steps}
    if condition is not None:
        trigger["condition"] = condition
    output: dict[str, Any] = {
        "name": "dyn",
        "queue_size": queue_size,
        "root": str(tmp_path),
        "pin_model_version": pin_model_version,
        "version": version,
    }
    return DynamicArtifactCallback(producer=spec, trigger=trigger, output=output)


# ---------------------------------------------------------------------------
# Line 111 — wrong event
# ---------------------------------------------------------------------------

def test_invariant_wrong_event_does_not_enqueue(tmp_path: Path) -> None:
    """on_step_end returns immediately when trigger.event != 'on_step_end'
    (line 111 `return`)."""
    cb = _make_cb(tmp_path, event="on_epoch_end")
    ctx = _FakeCtx(_TinyModel())
    cb.on_train_start(ctx=ctx)
    cb.on_step_end(step=1, batch=None, ctx=ctx)
    # Queue must be empty — nothing was submitted
    assert cb._q.qsize() == 0
    cb.on_train_end(ctx=ctx)


# ---------------------------------------------------------------------------
# Line 113-114 — every_n_steps gate / dedup gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "every, step, description",
    [
        (0, 5, "every<=0 disables trigger"),
        (-1, 5, "negative every disables trigger"),
        (3, 1, "step not a multiple of every"),
        (1, 3, "same step repeated"),  # needs _last_step pre-set
    ],
)
def test_invariant_step_gate_skips_submission(
    tmp_path: Path, every: int, step: int, description: str
) -> None:
    """on_step_end skips when every<=0 / wrong multiple / duplicate step (line 113-114)."""
    cb = _make_cb(tmp_path, every_n_steps=every)
    ctx = _FakeCtx(_TinyModel())
    cb.on_train_start(ctx=ctx)
    if description == "same step repeated":
        # pre-set _last_step so second call at same step is deduped
        cb._last_step = step
    cb.on_step_end(step=step, batch=None, ctx=ctx)
    assert cb._q.qsize() == 0
    cb.on_train_end(ctx=ctx)


# ---------------------------------------------------------------------------
# Lines 125,132 — condition False / condition exception
# ---------------------------------------------------------------------------

def test_invariant_condition_false_does_not_enqueue(tmp_path: Path) -> None:
    """When the trigger condition evaluates to False, no item is queued (line 125 `return`)."""
    cb = _make_cb(tmp_path, condition="metrics.get('loss', 1.0) < 0.0")
    ctx = _FakeCtx(_TinyModel())
    cb.on_train_start(ctx=ctx)
    cb.on_step_end(step=1, batch=None, ctx=ctx, metrics={"loss": 9999.0})
    assert cb._q.qsize() == 0
    cb.on_train_end(ctx=ctx)


def test_pin_current_behavior_condition_exception_logs_warning_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When the trigger condition raises, a WARNING is logged and nothing is queued
    (lines 126-132).

    Note: pinning current behavior — if the eval guard changes this may need updating.
    """
    import logging

    # Use an expression that will NameError at eval time
    cb = _make_cb(tmp_path, condition="undefined_name_xyz + step")
    ctx = _FakeCtx(_TinyModel())
    cb.on_train_start(ctx=ctx)
    with caplog.at_level(logging.WARNING, logger="lighttrain.builtin_plugins.data.artifacts.dynamic_producer"):
        cb.on_step_end(step=1, batch=None, ctx=ctx, metrics={})
    assert cb._q.qsize() == 0
    assert any("trigger condition" in r.message for r in caplog.records)
    cb.on_train_end(ctx=ctx)


# ---------------------------------------------------------------------------
# Lines 152-153 — _loop: queue.Empty timeout → continue
# ---------------------------------------------------------------------------

def test_invariant_worker_loop_tolerates_empty_queue_timeout() -> None:
    """The _loop must continue spinning when queue.get() times out (lines 152-153).

    Strategy: start worker, wait briefly so it times out at least once, then
    send poison pill and confirm it exits cleanly.
    """
    cb = DynamicArtifactCallback(
        producer={"name": "model_forward", "model": None,
                  "store": {"name": "safetensors-shards", "root": "/tmp/x"}},
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "dyn", "queue_size": 4},
    )
    # Start worker
    cb._worker = threading.Thread(target=cb._loop, daemon=True, name="dyn-test")
    cb._worker.start()
    # Let the worker spin and hit at least one queue.Empty timeout (0.5s each)
    time.sleep(0.6)
    # Signal stop and join
    cb._stop.set()
    cb._q.put_nowait(None)  # poison pill
    cb._worker.join(timeout=5.0)
    assert not cb._worker.is_alive()


# ---------------------------------------------------------------------------
# Line 155 — _loop: poison pill (None) causes return
# ---------------------------------------------------------------------------

def test_invariant_worker_exits_on_poison_pill() -> None:
    """The worker thread exits promptly when it dequeues None (line 155 `return`)."""
    cb = DynamicArtifactCallback(
        producer={"name": "model_forward", "model": None,
                  "store": {"name": "safetensors-shards", "root": "/tmp/x"}},
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "dyn", "queue_size": 2},
    )
    cb._worker = threading.Thread(target=cb._loop, daemon=True, name="dyn-test2")
    cb._worker.start()
    # Send poison pill immediately; worker should exit before next get() timeout
    cb._q.put(None)
    cb._worker.join(timeout=3.0)
    assert not cb._worker.is_alive()


# ---------------------------------------------------------------------------
# Lines 186-187 — _produce_one builds default store when not in spec
# ---------------------------------------------------------------------------

def test_invariant_produce_one_builds_default_store_when_missing(tmp_path: Path) -> None:
    """When 'store' is absent from the resolved spec, _produce_one constructs it from
    output.root + artifact_name (lines 186-187)."""
    model = _TinyModel()
    ctx = _FakeCtx(model)
    # Deliberately omit 'store' from the producer spec
    cb = DynamicArtifactCallback(
        producer={"name": "model_forward", "model": "$self"},
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "myart", "root": str(tmp_path / "out"), "queue_size": 1},
    )
    # Call _produce_one directly (bypass queue / threading)
    item = {
        "step": 1,
        "batch": {},
        "ctx": ctx,
        "version_tag": "v_test",
    }
    cb._produce_one(item)
    # A directory inside the output root must have been created
    root_out = tmp_path / "out"
    assert root_out.exists()
    # Some subdirectory should have been written (artifact store)
    children = list(root_out.iterdir())
    assert len(children) >= 1


# ---------------------------------------------------------------------------
# Lines 205-206 — bus.dispatch called after successful produce
# ---------------------------------------------------------------------------

def test_invariant_bus_dispatch_called_after_produce(tmp_path: Path) -> None:
    """After _produce_one completes, both on_artifact_new_version and
    on_artifact_finalized are dispatched to ctx.bus (lines 205-206)."""
    model = _TinyModel()
    ctx = _FakeCtx(model)
    bus = _FakeBus()
    ctx.bus = bus

    cb = DynamicArtifactCallback(
        producer={
            "name": "model_forward",
            "model": "$self",
            "store": {"name": "safetensors-shards", "root": str(tmp_path / "store"), "shard_size": 4},
        },
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "dyn", "queue_size": 4, "root": str(tmp_path)},
    )
    item = {"step": 10, "batch": {}, "ctx": ctx, "version_tag": "v_dispatch"}
    cb._produce_one(item)

    dispatched_events = [e for e, _ in bus.calls]
    assert "on_artifact_new_version" in dispatched_events
    assert "on_artifact_finalized" in dispatched_events


# ---------------------------------------------------------------------------
# Lines 215-220 — _resolve_spec pin_model_version deep-copies model
# ---------------------------------------------------------------------------

def test_invariant_pin_model_version_deep_copies_and_freezes(tmp_path: Path) -> None:
    """When pin_model_version=True, _resolve_spec deep-copies the model and
    calls eval() + requires_grad_(False) on all parameters (lines 215-220)."""
    model = _TinyModel()
    # Ensure at least one parameter has grad enabled
    for p in model.parameters():
        p.requires_grad_(True)

    cb = _make_cb(tmp_path, pin_model_version=True)
    ctx = _FakeCtx(model)
    spec = cb._resolve_spec(ctx)

    snapshot = spec["model"]
    # Must be a distinct object (deep copy)
    assert snapshot is not model
    # Must be the same nn.Module type
    assert isinstance(snapshot, nn.Module)
    # All parameters must have requires_grad=False
    for p in snapshot.parameters():
        assert not p.requires_grad, "snapshot param still requires grad"


def test_invariant_no_pin_uses_live_model_reference(tmp_path: Path) -> None:
    """When pin_model_version=False (default), _resolve_spec returns the live model
    (the '$self' branch without deep copy)."""
    model = _TinyModel()
    cb = _make_cb(tmp_path, pin_model_version=False)
    ctx = _FakeCtx(model)
    spec = cb._resolve_spec(ctx)
    # Without pinning, the live reference is forwarded
    assert spec["model"] is model


# ---------------------------------------------------------------------------
# Line 229 — _next_version_tag "auto" path
# ---------------------------------------------------------------------------

def test_invariant_auto_version_tag_contains_step(tmp_path: Path) -> None:
    """When output.version='auto', _next_version_tag returns 'v<timestamp>_step<N>'
    (line 229)."""
    cb = _make_cb(tmp_path, version="auto")
    tag = cb._next_version_tag(42)
    assert tag.endswith("_step42"), f"unexpected tag: {tag!r}"
    assert tag.startswith("v"), f"tag should start with 'v': {tag!r}"


# ---------------------------------------------------------------------------
# Line 236 — _next_version_tag fixed tag path
# ---------------------------------------------------------------------------

def test_invariant_fixed_version_tag_returned_verbatim(tmp_path: Path) -> None:
    """When output.version is a fixed string, _next_version_tag returns it as-is
    (line 236)."""
    cb = _make_cb(tmp_path, version="myrelease-1.2.3")
    tag = cb._next_version_tag(0)
    assert tag == "myrelease-1.2.3"


# ---------------------------------------------------------------------------
# _explode_batch edge cases
# ---------------------------------------------------------------------------

def test_invariant_explode_batch_returns_empty_for_no_known_keys() -> None:
    """_explode_batch returns [] when the batch has no input_ids/attention_mask/labels
    (line 236 of _explode_batch)."""
    result = _explode_batch({"unknown_key": torch.zeros(2, 4)})
    assert result == []


def test_invariant_explode_batch_per_sample_rows() -> None:
    """_explode_batch yields one dict per row in the batch."""
    batch = {
        "input_ids": torch.zeros(3, 4, dtype=torch.long),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
    }
    rows = _explode_batch(batch)
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert "input_ids" in row
        assert "attention_mask" in row
        assert row.get("id") == f"dyn_{i}"


# ---------------------------------------------------------------------------
# on_train_end: full lifecycle (start → submit → end) without crash
# ---------------------------------------------------------------------------

def test_invariant_full_lifecycle_does_not_crash(tmp_path: Path) -> None:
    """on_train_start / on_step_end / on_train_end completes without exception when
    the worker has no submissions to process."""
    cb = _make_cb(tmp_path, every_n_steps=100)  # step=1 never matches
    ctx = _FakeCtx(_TinyModel())
    cb.on_train_start(ctx=ctx)
    cb.on_step_end(step=1, batch=None, ctx=ctx)
    cb.on_train_end(ctx=ctx)
    # Worker should be cleaned up
    assert cb._worker is None


# ---------------------------------------------------------------------------
# on_train_end: poison pill path when queue is full
# ---------------------------------------------------------------------------

def test_invariant_on_train_end_handles_full_queue() -> None:
    """on_train_end with a full queue must not raise (uses try/except queue.Full)."""
    cb = DynamicArtifactCallback(
        producer={"name": "model_forward", "model": None,
                  "store": {"name": "safetensors-shards", "root": "/tmp/x"}},
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "dyn", "queue_size": 1},
    )
    # Manually fill the queue so put_nowait in on_train_end raises Full
    cb._q.put_nowait({"step": 1})
    # Start a worker that will get stuck (nothing to produce), then stop it
    cb._stop = threading.Event()
    # Manually stop the worker via the stop event before joining
    # We don't start the real worker here; instead test the _stop path directly
    cb._worker = None
    # on_train_end with a full queue should swallow queue.Full gracefully
    cb._stop.set()
    cb.on_train_end()   # _worker is None → no join


# ---------------------------------------------------------------------------
# _resolve_spec: model = None with $self and pin_model_version=True
# ---------------------------------------------------------------------------

def test_pin_current_behavior_pin_model_version_with_none_model(tmp_path: Path) -> None:
    """When ctx.model is None and pin_model_version=True, the spec['model'] becomes
    None (no deep-copy attempted on None).

    Pins current behavior — the code path `if live_model is not None and ...`.
    """
    cb = _make_cb(tmp_path, pin_model_version=True)
    ctx = _FakeCtx(model=None)
    spec = cb._resolve_spec(ctx)
    assert spec["model"] is None
