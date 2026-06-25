"""Coverage tests for ``lighttrain.builtin_plugins.callbacks.realtime_control.file_signals``.

Pins the uncovered branches not yet exercised by tests/realtime_control/test_file_signals.py:

* **line 61** — control_dir resolved from ``trainer._run_dir`` when ctx.run_dir is None.
* **lines 95–96** — lr.json FileNotFoundError on unlink is silently swallowed.
* **lines 105–106** — stop FileNotFoundError on unlink is silently swallowed.
* **lines 118–119** — eval_now FileNotFoundError on unlink is silently swallowed.
* **lines 146–147** — inject.py FileNotFoundError on unlink is silently swallowed.
* **lines 156–159** — lineage store update_node_payload called when store + run_node present;
                       exception from update_node_payload is silently swallowed.
* **line 170** — optimizer falls back to trainer.optimizer when ctx has no optimizer.
* **line 172** — _apply_lr_scale returns early when optimizer is None from both sources.
* **line 185** — scheduler.base_lrs list is rescaled together with param_group lrs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lighttrain.builtin_plugins.callbacks.realtime_control.file_signals import (
    FileSignalsCallback,
)
from lighttrain.callbacks.base import Signal
from lighttrain.engine._context import StepContext

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeOpt:
    """Minimal optimizer stub with a single param group."""

    def __init__(self, lr: float = 1e-3) -> None:
        self.param_groups = [{"lr": lr}]


class _FakeTrainer:
    """Minimal trainer stub."""

    def __init__(
        self,
        run_dir: Path | None = None,
        optimizer=None,
        scheduler=None,
        run_node_id: int | None = None,
    ) -> None:
        if run_dir is not None:
            self._run_dir = run_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        if run_node_id is not None:
            self._run_node_id = run_node_id


def _control_dir(tmp_path: Path) -> Path:
    d = tmp_path / "control"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# line 61: resolve control_dir from trainer._run_dir when ctx has no run_dir
# ---------------------------------------------------------------------------

def test_invariant_control_dir_resolved_from_trainer_run_dir(tmp_path: Path) -> None:
    """When ctx.run_dir is None the callback falls back to trainer._run_dir
    to derive control_dir (source line 61).

    A stop file in trainer._run_dir/control/ must be picked up.
    """
    trainer = _FakeTrainer(run_dir=tmp_path)
    # ctx deliberately has no run_dir
    ctx = StepContext()
    cb = FileSignalsCallback(poll_every=1)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    # control dir was resolved
    assert cb.control_dir == tmp_path / "control"
    # write a stop file and verify it is read
    (tmp_path / "control" / "stop").write_text("", encoding="utf-8")
    sig = cb.on_step_end(step=1)
    assert sig == Signal.STOP_TRAINING


def test_pin_no_control_dir_when_both_trainer_and_ctx_lack_run_dir() -> None:
    """Pin: when neither ctx nor trainer provide a run_dir, control_dir
    remains None and on_step_end returns CONTINUE without error.
    """
    trainer = _FakeTrainer()  # no _run_dir attribute
    ctx = StepContext()       # run_dir=None
    cb = FileSignalsCallback(poll_every=1)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    assert cb.control_dir is None
    assert cb.on_step_end(step=1) == Signal.CONTINUE


# ---------------------------------------------------------------------------
# lines 95–96: lr_path.unlink() FileNotFoundError is silently swallowed
# ---------------------------------------------------------------------------

def test_pin_current_behavior_lr_json_unlink_file_not_found_is_swallowed(
    tmp_path: Path,
) -> None:
    """Pin: if lr.json disappears between the exists() check and unlink()
    (race condition), the FileNotFoundError on line 94–96 is swallowed and
    the callback continues normally returning CONTINUE.

    Achieved by patching Path.unlink to raise FileNotFoundError only for
    lr.json.
    """
    opt = _FakeOpt()
    ctx = StepContext(optimizer=opt)
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = None
    _control_dir(tmp_path)
    lr_path = tmp_path / "control" / "lr.json"
    lr_path.write_text(json.dumps({"scale": 2.0}), encoding="utf-8")

    original_unlink = Path.unlink

    def _patched_unlink(self, missing_ok=False):  # noqa: ANN001
        if self.name == "lr.json":
            raise FileNotFoundError("gone")
        return original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", _patched_unlink):
        sig = cb.on_step_end(step=1)

    assert sig == Signal.CONTINUE
    # lr was still applied before the failed unlink
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)


# ---------------------------------------------------------------------------
# lines 105–106: stop unlink FileNotFoundError is silently swallowed
# ---------------------------------------------------------------------------

def test_pin_current_behavior_stop_unlink_file_not_found_is_swallowed(
    tmp_path: Path,
) -> None:
    """Pin: FileNotFoundError when unlinking 'stop' (lines 105–106) is
    swallowed; the STOP_TRAINING signal is still returned.
    """
    ctx = StepContext()
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = None
    _control_dir(tmp_path)
    (tmp_path / "control" / "stop").write_text("", encoding="utf-8")

    original_unlink = Path.unlink

    def _patched_unlink(self, missing_ok=False):  # noqa: ANN001
        if self.name == "stop":
            raise FileNotFoundError("gone")
        return original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", _patched_unlink):
        sig = cb.on_step_end(step=1)

    assert sig == Signal.STOP_TRAINING


# ---------------------------------------------------------------------------
# lines 118–119: eval_now unlink FileNotFoundError is silently swallowed
# ---------------------------------------------------------------------------

def test_pin_current_behavior_eval_now_unlink_file_not_found_is_swallowed(
    tmp_path: Path,
) -> None:
    """Pin: FileNotFoundError when unlinking 'eval_now' (lines 118–119) is
    swallowed; force_eval is still set on ctx.extras.
    """
    ctx = StepContext()
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = None
    _control_dir(tmp_path)
    (tmp_path / "control" / "eval_now").write_text("", encoding="utf-8")

    original_unlink = Path.unlink

    def _patched_unlink(self, missing_ok=False):  # noqa: ANN001
        if self.name == "eval_now":
            raise FileNotFoundError("gone")
        return original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", _patched_unlink):
        cb.on_step_end(step=1)

    assert ctx.extras.get("force_eval") is True


# ---------------------------------------------------------------------------
# lines 146–147: inject.py unlink FileNotFoundError is silently swallowed
# ---------------------------------------------------------------------------

def test_pin_current_behavior_inject_unlink_file_not_found_is_swallowed(
    tmp_path: Path,
) -> None:
    """Pin: FileNotFoundError when unlinking 'inject.py' (lines 146–147) is
    swallowed; the inject event is still recorded.
    """
    ctx = StepContext()
    cb = FileSignalsCallback(
        control_dir=tmp_path / "control", poll_every=1, allow_inject=True
    )
    cb._ctx = ctx
    cb._trainer = None
    _control_dir(tmp_path)
    (tmp_path / "control" / "inject.py").write_text("x = 1", encoding="utf-8")

    original_unlink = Path.unlink

    def _patched_unlink(self, missing_ok=False):  # noqa: ANN001
        if self.name == "inject.py":
            raise FileNotFoundError("gone")
        return original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", _patched_unlink):
        cb.on_step_end(step=1)

    events = ctx.diagnostics.get("realtime_events", [])
    assert any(e["event"] == "inject" for e in events)


# ---------------------------------------------------------------------------
# lines 156–159: lineage store update path
# ---------------------------------------------------------------------------

def test_invariant_lineage_store_update_node_payload_called(tmp_path: Path) -> None:
    """When ctx.lineage_store and trainer._run_node_id are both set,
    update_node_payload is called with the accumulated realtime_events
    (source lines 155–157).
    """
    store = MagicMock()
    ctx = StepContext(lineage_store=store)

    trainer = _FakeTrainer(run_node_id=42)
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "stop").write_text("", encoding="utf-8")

    cb.on_step_end(step=1)

    store.update_node_payload.assert_called_once()
    call_args = store.update_node_payload.call_args
    assert call_args[0][0] == 42
    realtime_events = call_args[0][1]["realtime_events"]
    assert any(e["event"] == "stop" for e in realtime_events)


def test_pin_current_behavior_lineage_update_exception_swallowed(
    tmp_path: Path,
) -> None:
    """Pin: if update_node_payload raises, the exception is swallowed
    (source lines 156–159) and the signal is still returned correctly.
    """
    store = MagicMock()
    store.update_node_payload.side_effect = RuntimeError("db gone")
    ctx = StepContext(lineage_store=store)

    trainer = _FakeTrainer(run_node_id=7)
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "stop").write_text("", encoding="utf-8")

    sig = cb.on_step_end(step=1)
    assert sig == Signal.STOP_TRAINING


def test_pin_current_behavior_lineage_not_called_when_store_none(
    tmp_path: Path,
) -> None:
    """Pin: when lineage_store is None, update_node_payload is never invoked
    (no AttributeError and events are still recorded locally).
    """
    ctx = StepContext()  # lineage_store=None
    trainer = _FakeTrainer(run_node_id=1)
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "eval_now").write_text("", encoding="utf-8")

    cb.on_step_end(step=1)  # must not raise
    events = ctx.diagnostics.get("realtime_events", [])
    assert any(e["event"] == "eval_now" for e in events)


def test_pin_current_behavior_lineage_not_called_when_run_node_none(
    tmp_path: Path,
) -> None:
    """Pin: when trainer has no _run_node_id, update_node_payload is never
    invoked even if store is set.
    """
    store = MagicMock()
    ctx = StepContext(lineage_store=store)
    trainer = _FakeTrainer()  # no _run_node_id attribute
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "eval_now").write_text("", encoding="utf-8")

    cb.on_step_end(step=1)

    store.update_node_payload.assert_not_called()


# ---------------------------------------------------------------------------
# line 170: optimizer from trainer.optimizer when ctx.optimizer is None
# ---------------------------------------------------------------------------

def test_invariant_lr_scale_uses_trainer_optimizer_when_ctx_optimizer_none(
    tmp_path: Path,
) -> None:
    """When ctx has no optimizer, _apply_lr_scale falls back to
    trainer.optimizer (source line 170).

    Setup: ctx.optimizer=None, trainer.optimizer=_FakeOpt, scale=3.0.
    Expected: trainer.optimizer param group lr tripled.
    """
    opt = _FakeOpt(lr=1e-3)
    trainer = _FakeTrainer(optimizer=opt)
    ctx = StepContext()  # no optimizer

    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 3.0}), encoding="utf-8"
    )

    cb.on_step_end(step=1)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-3)


# ---------------------------------------------------------------------------
# line 172: _apply_lr_scale returns early when optimizer is None everywhere
# ---------------------------------------------------------------------------

def test_pin_current_behavior_lr_scale_no_op_when_no_optimizer(
    tmp_path: Path,
) -> None:
    """Pin: when both ctx.optimizer and trainer.optimizer are None,
    _apply_lr_scale returns early (line 172) without error.

    A valid scale still produces an event (lr_scale) in diagnostics because
    the apply failure is silent — only the actual lr modification is skipped.
    """
    trainer = _FakeTrainer()  # optimizer=None
    ctx = StepContext()       # optimizer=None

    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 2.0}), encoding="utf-8"
    )

    sig = cb.on_step_end(step=1)  # must not raise
    assert sig == Signal.CONTINUE
    # event still recorded
    events = ctx.diagnostics.get("realtime_events", [])
    assert any(e["event"] == "lr_scale" for e in events)


# ---------------------------------------------------------------------------
# line 185: scheduler.base_lrs list is rescaled
# ---------------------------------------------------------------------------

def test_invariant_scheduler_base_lrs_rescaled_with_lr_scale(tmp_path: Path) -> None:
    """When the scheduler has a base_lrs list, _apply_lr_scale multiplies
    every element (source line 185).

    Setup: fake scheduler with base_lrs=[1e-3, 2e-3], scale=2.0.
    Expected: base_lrs == [2e-3, 4e-3].
    """
    opt = _FakeOpt(lr=1e-3)

    class _FakeScheduler:
        def __init__(self) -> None:
            self.base_lrs = [1e-3, 2e-3]

    scheduler = _FakeScheduler()
    ctx = StepContext(optimizer=opt, scheduler=scheduler)
    trainer = _FakeTrainer()  # scheduler also not on trainer

    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 2.0}), encoding="utf-8"
    )

    cb.on_step_end(step=1)

    assert scheduler.base_lrs == pytest.approx([2e-3, 4e-3])
    # param group lr also scaled
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)


def test_invariant_scheduler_from_trainer_when_ctx_scheduler_none(
    tmp_path: Path,
) -> None:
    """When ctx.scheduler is None, _apply_lr_scale falls back to
    trainer.scheduler for base_lrs rescaling (line 181 + 185).
    """
    opt = _FakeOpt(lr=1e-3)

    class _FakeScheduler:
        def __init__(self) -> None:
            self.base_lrs = [5e-4]

    scheduler = _FakeScheduler()
    trainer = _FakeTrainer(optimizer=opt, scheduler=scheduler)
    ctx = StepContext()  # no scheduler, no optimizer

    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 4.0}), encoding="utf-8"
    )

    cb.on_step_end(step=1)

    assert scheduler.base_lrs == pytest.approx([2e-3])
    assert opt.param_groups[0]["lr"] == pytest.approx(4e-3)


def test_pin_current_behavior_scheduler_with_non_list_base_lrs_not_rescaled(
    tmp_path: Path,
) -> None:
    """Pin: if scheduler.base_lrs is NOT a list, it is left untouched
    (isinstance guard on line 184 prevents iteration).

    This pins existing behavior; if base_lrs=None or a tuple, the guard
    returns silently.
    """
    opt = _FakeOpt(lr=1e-3)

    class _FakeScheduler:
        base_lrs: object = None  # not a list

    scheduler = _FakeScheduler()
    ctx = StepContext(optimizer=opt, scheduler=scheduler)
    trainer = _FakeTrainer()

    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = trainer
    _control_dir(tmp_path)
    (tmp_path / "control" / "lr.json").write_text(
        json.dumps({"scale": 2.0}), encoding="utf-8"
    )

    cb.on_step_end(step=1)
    # base_lrs stays None — no error
    assert scheduler.base_lrs is None
    # param group lr was still scaled
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)


# ---------------------------------------------------------------------------
# Parametrize: combined signal priorities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "files,expected_signal",
    [
        (["stop"], Signal.STOP_TRAINING),
        (["eval_now"], Signal.CONTINUE),
        (["stop", "eval_now"], Signal.STOP_TRAINING),
    ],
)
def test_invariant_signal_priority_stop_dominates(
    tmp_path: Path,
    files: list[str],
    expected_signal: Signal,
) -> None:
    """STOP_TRAINING is returned whenever 'stop' is present; eval_now alone
    returns CONTINUE (it only sets a flag, not a signal).
    """
    ctx = StepContext()
    cb = FileSignalsCallback(control_dir=tmp_path / "control", poll_every=1)
    cb._ctx = ctx
    cb._trainer = None
    _control_dir(tmp_path)
    for name in files:
        (tmp_path / "control" / name).write_text("", encoding="utf-8")
    sig = cb.on_step_end(step=1)
    assert sig == expected_signal


# ---------------------------------------------------------------------------
# on_train_start: explicit control_dir skips run_dir derivation
# ---------------------------------------------------------------------------

def test_pin_explicit_control_dir_not_overridden_by_ctx_run_dir(
    tmp_path: Path,
) -> None:
    """Pin: when control_dir is given explicitly, on_train_start DOES NOT
    override it from ctx.run_dir or trainer._run_dir.
    """
    explicit = tmp_path / "myctl"
    ctx = StepContext(run_dir=tmp_path / "other_run")
    trainer = _FakeTrainer(run_dir=tmp_path / "trainer_run")
    cb = FileSignalsCallback(control_dir=explicit, poll_every=1)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    assert cb.control_dir == explicit


# ---------------------------------------------------------------------------
# on_train_start: ctx is None edge case (no crash)
# ---------------------------------------------------------------------------

def test_pin_on_train_start_ctx_none_does_not_crash(tmp_path: Path) -> None:
    """Pin: on_train_start with ctx=None should not raise even when
    deriving control_dir from trainer._run_dir.
    """
    trainer = _FakeTrainer(run_dir=tmp_path)
    cb = FileSignalsCallback(poll_every=1)
    cb.on_train_start(trainer=trainer, ctx=None)
    assert cb.control_dir == tmp_path / "control"
    # and a tick should not crash
    sig = cb.on_step_end(step=1)
    assert sig == Signal.CONTINUE
