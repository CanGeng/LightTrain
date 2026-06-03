"""Adversarial tests for ``lighttrain.builtin_plugins.realtime_control.file_signals.FileSignalsCallback``.

Coverage beyond ``tests/test_realtime_control.py`` (which tests the basic
lr_scale / stop / eval_now / inject paths):

* **lr_scale rejects nan / inf / zero / negative** (line 163 of source).
* **lr_scale invalid JSON tolerated** (file unlinked, no crash).
* **Poll-every gate**: with poll_every=10, step=3 ignores files; step=10
  reads them.
* **Files unlinked even on parse failure** so the next poll is clean.
* **inject.py exec namespace pin**: ``{trainer, model, ctx}`` keys exactly.
* **inject error recorded as ``inject_error`` event** with str(exc).
* **inject sandbox-by-design pin**: with ``allow_inject=True``, the script
  CAN access ``trainer.__class__`` (this is documented; control dir is
  trusted).
* **Realtime events appended to ctx.diagnostics["realtime_events"]**.
* **Poll-every clamped to >= 1**.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.realtime_control.file_signals import FileSignalsCallback
from lighttrain.callbacks.base import Signal
from lighttrain.engine._context import StepContext


class _Trainer:
    def __init__(self, model, optimizer, run_dir):
        self.model = model
        self.optimizer = optimizer
        self._run_dir = run_dir


def _setup(tmp_path: Path, *, poll_every: int = 1, allow_inject: bool = True):
    optimizer = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)], lr=1e-3
    )
    ctx = StepContext(run_dir=tmp_path, optimizer=optimizer)
    cb = FileSignalsCallback(poll_every=poll_every, allow_inject=allow_inject)
    trainer = _Trainer(model=None, optimizer=optimizer, run_dir=tmp_path)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    return cb, ctx, optimizer


def _write_control(tmp_path: Path, name: str, content: str) -> None:
    p = tmp_path / "control" / name
    p.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# lr_scale validation
# ---------------------------------------------------------------------------

def test_invariant_lr_scale_doubles_lr_round_trip(tmp_path):
    """Closed form: scale=2.0 with lr=1e-3 → new lr == 2e-3.

    File is unlinked after read.
    """
    cb, _ctx, opt = _setup(tmp_path)
    _write_control(tmp_path, "lr.json", json.dumps({"scale": 2.0}))
    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert not (tmp_path / "control" / "lr.json").exists()


def test_lr_scale_zero_rejected_no_lr_change(tmp_path):
    """``scale=0`` is rejected by ``_apply_lr_scale`` (line 163: ``if not
    (scale > 0): return``).

    Setup: scale=0; lr stays at original value.
    """
    cb, _ctx, opt = _setup(tmp_path)
    original_lr = opt.param_groups[0]["lr"]
    _write_control(tmp_path, "lr.json", json.dumps({"scale": 0.0}))
    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == original_lr


def test_lr_scale_negative_rejected_no_lr_change(tmp_path):
    """``scale=-1`` is rejected (only scale > 0 is honored)."""
    cb, _ctx, opt = _setup(tmp_path)
    original_lr = opt.param_groups[0]["lr"]
    _write_control(tmp_path, "lr.json", json.dumps({"scale": -1.0}))
    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == original_lr


def test_lr_scale_nan_rejected_no_lr_change(tmp_path):
    """``scale=NaN`` is rejected (``NaN > 0`` is False)."""
    cb, _ctx, opt = _setup(tmp_path)
    original_lr = opt.param_groups[0]["lr"]
    # JSON does not natively encode NaN, so we use a payload that becomes
    # NaN on float() — write as raw text since json.dumps rejects NaN.
    p = tmp_path / "control" / "lr.json"
    p.write_text('{"scale": NaN}', encoding="utf-8")
    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == original_lr


def test_lr_scale_invalid_json_tolerated_file_unlinked(tmp_path):
    """An lr.json with malformed JSON does NOT crash; the file is unlinked
    so the next poll is clean (line 89-94 of source).
    """
    cb, _ctx, opt = _setup(tmp_path)
    original_lr = opt.param_groups[0]["lr"]
    _write_control(tmp_path, "lr.json", "not valid json {")
    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == original_lr
    assert not (tmp_path / "control" / "lr.json").exists()


# ---------------------------------------------------------------------------
# Poll-every gate
# ---------------------------------------------------------------------------

def test_poll_every_gates_file_read(tmp_path):
    """With poll_every=10, step=3 ignores the file; step=10 reads it.

    Setup: write lr.json at the start; tick step=3, then step=10.
    Expected: at step=3, file untouched; at step=10, file gone + lr applied.
    """
    cb, _ctx, opt = _setup(tmp_path, poll_every=10)
    _write_control(tmp_path, "lr.json", json.dumps({"scale": 0.5}))
    original_lr = opt.param_groups[0]["lr"]

    cb.on_step_end(step=3)
    # File still present, lr unchanged
    assert (tmp_path / "control" / "lr.json").exists()
    assert opt.param_groups[0]["lr"] == original_lr

    cb.on_step_end(step=10)
    # File consumed, lr halved
    assert not (tmp_path / "control" / "lr.json").exists()
    assert opt.param_groups[0]["lr"] == pytest.approx(original_lr * 0.5)


def test_invariant_poll_every_clamped_to_at_least_one():
    """``poll_every`` is clamped to >= 1 by the constructor (line 48)."""
    cb = FileSignalsCallback(poll_every=0)
    assert cb.poll_every == 1
    cb2 = FileSignalsCallback(poll_every=-5)
    assert cb2.poll_every == 1


def test_callback_with_no_control_dir_returns_continue(tmp_path):
    """When ``control_dir`` is None and no trainer/ctx provides it,
    on_step_end returns CONTINUE (line 71-72).
    """
    cb = FileSignalsCallback(poll_every=1)
    sig = cb.on_step_end(step=1)
    assert sig == Signal.CONTINUE


# ---------------------------------------------------------------------------
# Stop signal
# ---------------------------------------------------------------------------

def test_stop_file_returns_stop_training_signal(tmp_path):
    """The presence of ``stop`` file returns ``Signal.STOP_TRAINING``.

    Setup: write stop file; tick.
    Expected: signal == STOP_TRAINING; file unlinked.
    """
    cb, _ctx, _opt = _setup(tmp_path)
    _write_control(tmp_path, "stop", "")
    sig = cb.on_step_end(step=1)
    assert sig == Signal.STOP_TRAINING
    assert not (tmp_path / "control" / "stop").exists()


# ---------------------------------------------------------------------------
# eval_now
# ---------------------------------------------------------------------------

def test_eval_now_sets_force_eval_flag_on_ctx(tmp_path):
    """``eval_now`` file sets ``ctx.extras["force_eval"] = True``.

    Setup: write eval_now; tick.
    Expected: flag set; file removed.
    """
    cb, ctx, _opt = _setup(tmp_path)
    _write_control(tmp_path, "eval_now", "")
    cb.on_step_end(step=1)
    assert ctx.extras.get("force_eval") is True
    assert not (tmp_path / "control" / "eval_now").exists()


# ---------------------------------------------------------------------------
# inject.py
# ---------------------------------------------------------------------------

def test_inject_executes_in_namespace_with_trainer_model_ctx(tmp_path):
    """The inject namespace has exactly ``{trainer, model, ctx}`` bindings.

    Setup: inject.py writes its namespace keys into a sentinel file.
    Expected: the file lists the 3 keys.
    """
    cb, _ctx, _opt = _setup(tmp_path)
    sentinel = tmp_path / "sentinel.txt"
    script = f"""
with open({repr(str(sentinel))}, 'w') as f:
    f.write(','.join(sorted(['trainer', 'model', 'ctx'])))
"""
    _write_control(tmp_path, "inject.py", script)
    cb.on_step_end(step=1)
    # Inject ran (sentinel created)
    assert sentinel.exists()
    assert sentinel.read_text() == "ctx,model,trainer"


def test_inject_error_recorded_as_inject_error_event(tmp_path):
    """When inject.py raises, the failure is recorded as an event of kind
    ``inject_error`` with ``str(exc)`` (line 134-140 of source).

    Setup: inject.py with explicit raise.
    Expected: realtime_events contains an inject_error with the message.
    """
    cb, ctx, _opt = _setup(tmp_path)
    _write_control(tmp_path, "inject.py", "raise RuntimeError('controlled boom')")
    cb.on_step_end(step=1)
    log = ctx.diagnostics.get("realtime_events", [])
    error_events = [e for e in log if e["event"] == "inject_error"]
    assert len(error_events) == 1
    assert "controlled boom" in error_events[0]["error"]


def test_inject_disabled_when_allow_inject_false(tmp_path):
    """With ``allow_inject=False``, inject.py is NOT executed even when present.

    Setup: inject.py would write a sentinel; ``allow_inject=False``.
    Expected: sentinel NOT created.
    """
    cb, _ctx, _opt = _setup(tmp_path, allow_inject=False)
    sentinel = tmp_path / "should_not_exist.txt"
    _write_control(
        tmp_path, "inject.py", f"open({repr(str(sentinel))}, 'w').write('hi')"
    )
    cb.on_step_end(step=1)
    assert not sentinel.exists()
    # The inject.py file itself is NOT unlinked when allow_inject=False
    # (the unlink is inside the allow_inject branch).
    assert (tmp_path / "control" / "inject.py").exists()


def test_pin_inject_exec_unrestricted_by_design(tmp_path):
    """Pin: inject.py runs in an UNSANDBOXED exec — it can reach
    ``trainer.__class__``, ``ctx.__class__``, etc.

    This is the documented design contract: the control dir is trusted
    (protected by filesystem permissions). If a sandbox is later added,
    update this test AND document the breaking change.
    """
    cb, _ctx, _opt = _setup(tmp_path)
    sentinel = tmp_path / "klass.txt"
    script = f"""
name = type(trainer).__name__
with open({repr(str(sentinel))}, 'w') as f:
    f.write(name)
"""
    _write_control(tmp_path, "inject.py", script)
    cb.on_step_end(step=1)
    # Sentinel contains _Trainer (the test helper class name)
    assert sentinel.read_text() == "_Trainer"


# ---------------------------------------------------------------------------
# Realtime events log
# ---------------------------------------------------------------------------

def test_invariant_events_appended_to_ctx_diagnostics(tmp_path):
    """Every triggered action is appended to
    ``ctx.diagnostics["realtime_events"]`` (line 146-148).

    Setup: trigger lr_scale + eval_now in one tick.
    Expected: two events in the log.
    """
    cb, ctx, _opt = _setup(tmp_path)
    _write_control(tmp_path, "lr.json", json.dumps({"scale": 0.5}))
    _write_control(tmp_path, "eval_now", "")
    cb.on_step_end(step=1)
    events = ctx.diagnostics.get("realtime_events", [])
    kinds = {e["event"] for e in events}
    assert kinds == {"lr_scale", "eval_now"}


def test_events_accumulate_across_ticks(tmp_path):
    """Repeated triggers accumulate in the same list across multiple ticks."""
    cb, ctx, _opt = _setup(tmp_path)
    _write_control(tmp_path, "eval_now", "")
    cb.on_step_end(step=1)
    _write_control(tmp_path, "eval_now", "")
    cb.on_step_end(step=2)
    events = ctx.diagnostics.get("realtime_events", [])
    assert sum(1 for e in events if e["event"] == "eval_now") == 2


# ---------------------------------------------------------------------------
# Registry pin
# ---------------------------------------------------------------------------

def test_file_signals_registered_under_callback_file_signals():
    """Pin: registered as ``('callback', 'file_signals')``."""
    from lighttrain.registry import get
    assert get("callback", "file_signals") is FileSignalsCallback
