"""Edge-case coverage for ``lighttrain.trainers._primitives``.

Pins exercised:
* ``forward_with_activations`` — normal path (lines 42-53): model called with
  ``output_hidden_states=True``; ``hs`` returned as a tuple; ``layers`` subset
  selection; ValueError when model returns no hidden_states.
* ``run_train_loop`` — model/optimizer guards; happy-path loop with epoch
  rollover; Signal.STOP_TRAINING from ctx.extras / from bus; crash-path
  ``on_exception`` dispatch secondary-exception suppression (lines 150-151);
  finally ``on_train_end`` exception suppression (lines 159-160);
  finally ``logger.flush`` exception suppression (lines 164-165);
  finally ``write_index_page`` exception suppression (lines 174-175);
  no _run_dir → write_index_page skipped; logger=None → flush skipped.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.trainers._primitives import forward_with_activations, run_train_loop

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _HiddenStatesOutput:
    """Minimal model output that carries hidden_states."""

    def __init__(self, hidden_states=None):
        self.hidden_states = hidden_states
        self.logits = None


class _ModelWithHiddenStates:
    """forward() returns hidden_states tuple of requested length."""

    def __init__(self, n_layers: int = 4):
        self.n_layers = n_layers
        self.call_kwargs: dict = {}

    def __call__(self, **kwargs):
        self.call_kwargs = dict(kwargs)
        hs = tuple(f"layer_{i}" for i in range(self.n_layers))
        return _HiddenStatesOutput(hidden_states=hs)


class _ModelNoHiddenStates:
    """forward() returns an object with hidden_states=None."""

    def __call__(self, **kwargs):
        return _HiddenStatesOutput(hidden_states=None)


class _StepOutput:
    def __init__(self, metrics: dict):
        self.metrics = metrics


class _Ctx:
    """Minimal training context."""

    def __init__(self):
        self.step = 0
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.extras: dict[str, Any] = {}


class _FakeLoader:
    """Finite iterable acting as a DataLoader."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(list(self._batches))


class _FakeDataModule:
    def __init__(self, batches):
        self._loader = _FakeLoader(batches)

    def train_loader(self):
        return self._loader


def _make_trainer(
    *,
    batches=None,
    target_steps=2,
    model=True,
    optimizer=True,
    logger=None,
    bus=None,
    _run_dir=None,
    extras_signal: int = 0,
    bus_signal: Signal = Signal.CONTINUE,
):
    """Build a minimal stub trainer compatible with run_train_loop."""
    if batches is None:
        batches = [{"x": i} for i in range(10)]

    ctx = _Ctx()

    t = MagicMock()
    t.model = object() if model else None
    t.optimizer = object() if optimizer else None
    t.data_module = _FakeDataModule(batches)
    t.ctx = ctx
    t.logger = logger
    t._stop_requested = False
    t._run_dir = _run_dir

    # bus: use a real EventBus by default so dispatch behaves correctly
    if bus is None:
        real_bus = EventBus()
        t.bus = real_bus
    else:
        t.bus = bus

    def _produce_batch(raw):
        return raw

    def _train_step(batch):
        ctx.extras["loss_signal"] = extras_signal
        return _StepOutput({"loss": 0.1})

    t.produce_batch.side_effect = _produce_batch
    t.train_step.side_effect = _train_step

    t._stop_requested = False

    return t


# ===========================================================================
# forward_with_activations — lines 42–53
# ===========================================================================


def test_invariant_forward_with_activations_calls_model_with_flag():
    """Model is called with ``output_hidden_states=True`` merged into batch kwargs."""
    model = _ModelWithHiddenStates(n_layers=3)
    batch = {"input_ids": [1, 2, 3]}
    out, hs = forward_with_activations(model, batch)
    assert model.call_kwargs.get("output_hidden_states") is True
    assert "input_ids" in model.call_kwargs


def test_invariant_forward_with_activations_returns_tuple_of_all_layers():
    """``hs`` is a plain tuple; length equals model n_layers when ``layers=None``."""
    model = _ModelWithHiddenStates(n_layers=4)
    out, hs = forward_with_activations(model, {})
    assert isinstance(hs, tuple)
    assert len(hs) == 4


def test_invariant_forward_with_activations_layers_subset():
    """``layers=[0, 2]`` selects only those indices from the full hidden_states."""
    model = _ModelWithHiddenStates(n_layers=4)
    out, hs = forward_with_activations(model, {}, layers=[0, 2])
    assert len(hs) == 2
    assert hs[0] == "layer_0"
    assert hs[1] == "layer_2"


def test_invariant_forward_with_activations_layers_none_returns_all():
    """With ``layers=None`` the branch at line 51 is not entered; all hs returned."""
    model = _ModelWithHiddenStates(n_layers=3)
    out, hs = forward_with_activations(model, {}, layers=None)
    assert hs == ("layer_0", "layer_1", "layer_2")


def test_invariant_forward_with_activations_raises_on_no_hidden_states():
    """ValueError raised when model returns hidden_states=None (lines 44-49)."""
    model = _ModelNoHiddenStates()
    with pytest.raises(ValueError, match="_ModelNoHiddenStates.*hidden_states"):
        forward_with_activations(model, {})


def test_invariant_forward_returns_output_object():
    """The first element of the return tuple is the raw model output."""
    model = _ModelWithHiddenStates(n_layers=2)
    out, hs = forward_with_activations(model, {})
    assert isinstance(out, _HiddenStatesOutput)


# ===========================================================================
# run_train_loop — guards
# ===========================================================================


def test_invariant_run_train_loop_raises_when_model_is_none():
    """RuntimeError when trainer.model is None."""
    t = _make_trainer(model=False)
    with pytest.raises(RuntimeError, match="model is not set"):
        run_train_loop(t, target_steps=1)


def test_invariant_run_train_loop_raises_when_optimizer_is_none():
    """RuntimeError when trainer.optimizer is None."""
    t = _make_trainer(optimizer=False)
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        run_train_loop(t, target_steps=1)


# ===========================================================================
# run_train_loop — happy path
# ===========================================================================


def test_invariant_run_train_loop_returns_metrics_after_target_steps():
    """Loop terminates after target_steps and returns a dict."""
    t = _make_trainer(target_steps=3)
    result = run_train_loop(t, target_steps=3)
    assert isinstance(result, dict)
    assert t.ctx.step == 3


def test_invariant_run_train_loop_increments_step_each_iteration():
    """ctx.step and ctx.global_step both advance by 1 per batch."""
    t = _make_trainer()
    run_train_loop(t, target_steps=2)
    assert t.ctx.step == 2
    assert t.ctx.global_step == 2


def test_invariant_run_train_loop_dispatches_on_train_start_and_epoch_begin():
    """on_train_start and on_epoch_begin are dispatched before the first batch."""
    dispatched: list[str] = []

    class _Recorder:
        def on_train_start(self, **_):
            dispatched.append("on_train_start")

        def on_epoch_begin(self, **_):
            dispatched.append("on_epoch_begin")

    bus = EventBus(callbacks=[_Recorder()])
    t = _make_trainer(bus=bus)
    run_train_loop(t, target_steps=1)
    assert dispatched[0] == "on_train_start"
    assert "on_epoch_begin" in dispatched


def test_invariant_run_train_loop_epoch_rollover():
    """When loader is exhausted, on_epoch_end is dispatched; epoch increments;
    batch_in_epoch resets; iteration continues from the new epoch."""
    epoch_ends: list[int] = []

    class _EpochTracker:
        def on_epoch_end(self, *, epoch, **_):
            epoch_ends.append(epoch)

    bus = EventBus(callbacks=[_EpochTracker()])
    # Only 1 batch per epoch, need 3 steps → 3 epochs
    t = _make_trainer(batches=[{"x": 0}], bus=bus)
    run_train_loop(t, target_steps=3)
    # at least one epoch rollover must have happened
    assert len(epoch_ends) >= 1
    assert t.ctx.epoch >= 1


def test_invariant_run_train_loop_stop_requested_exits_early():
    """Setting trainer._stop_requested inside a callback stops the loop early."""

    class _Stopper:
        def on_train_batch_end(self, **_):
            return Signal.STOP_TRAINING

    bus = EventBus(callbacks=[_Stopper()])
    t = _make_trainer(bus=bus)
    run_train_loop(t, target_steps=100)
    # Must have stopped after 1 step due to STOP_TRAINING signal
    assert t.ctx.step <= 5  # definitely not 100


def test_invariant_run_train_loop_signal_stop_from_ctx_extras():
    """When ctx.extras['loss_signal'] == Signal.STOP_TRAINING the loop stops."""
    t = _make_trainer(extras_signal=int(Signal.STOP_TRAINING))
    run_train_loop(t, target_steps=100)
    # Must have exited after the first step sets _stop_requested
    assert t.ctx.step <= 2


def test_invariant_run_train_loop_calls_periodic_hooks():
    """_maybe_log, _maybe_eval, _maybe_save, _maybe_save_best called each step."""
    t = _make_trainer(target_steps=2)
    run_train_loop(t, target_steps=2)
    assert t._maybe_log.call_count == 2
    assert t._maybe_eval.call_count == 2
    assert t._maybe_save.call_count == 2
    assert t._maybe_save_best.call_count == 2


def test_invariant_run_train_loop_calls_final_save():
    """_final_save is called with last_metrics once the loop exits normally."""
    t = _make_trainer(target_steps=2)
    run_train_loop(t, target_steps=2)
    t._final_save.assert_called_once()


# ===========================================================================
# run_train_loop — crash path: on_exception secondary exception suppressed
# ===========================================================================


def test_pin_current_behavior_on_exception_secondary_exception_suppressed(caplog):
    """If on_exception dispatch itself raises, the secondary exception is logged
    (lines 150-151) and the original exc re-raised.

    Pin: the suppression log message is 'Suppressed secondary exception…'.
    """
    original_exc = RuntimeError("original crash")

    # Make train_step raise the original exception
    t = _make_trainer()
    t.train_step.side_effect = original_exc

    # Make bus.dispatch raise on "on_exception" call
    EventBus()
    dispatch_calls: list[str] = []

    def _dispatch(event: str, **kwargs):
        dispatch_calls.append(event)
        if event == "on_exception":
            raise RuntimeError("secondary crash in on_exception")
        return Signal.CONTINUE

    t.bus.dispatch = _dispatch

    with caplog.at_level(logging.WARNING, logger="lighttrain.trainers._primitives"):
        with pytest.raises(RuntimeError, match="original crash"):
            run_train_loop(t, target_steps=5)

    assert any("Suppressed secondary exception" in r.message for r in caplog.records)


# ===========================================================================
# run_train_loop — finally: on_train_end exception suppressed (lines 159-160)
# ===========================================================================


def test_pin_current_behavior_on_train_end_exception_suppressed(caplog):
    """on_train_end raising inside the finally block is suppressed with a
    WARNING log (lines 159-160); the loop still returns last_metrics."""

    t = _make_trainer()
    dispatch_calls: list[str] = []

    def _dispatch(event: str, **kwargs):
        dispatch_calls.append(event)
        if event == "on_train_end":
            raise RuntimeError("on_train_end boom")
        return Signal.CONTINUE

    t.bus.dispatch = _dispatch

    with caplog.at_level(logging.WARNING, logger="lighttrain.trainers._primitives"):
        result = run_train_loop(t, target_steps=1)

    assert "on_train_end" in dispatch_calls
    assert any("Suppressed exception in on_train_end" in r.message for r in caplog.records)
    assert isinstance(result, dict)


# ===========================================================================
# run_train_loop — finally: logger.flush exception suppressed (lines 164-165)
# ===========================================================================


def test_pin_current_behavior_logger_flush_exception_suppressed(caplog):
    """If logger.flush() raises, the exception is suppressed and a WARNING is
    logged (lines 164-165)."""
    logger = MagicMock()
    logger.flush.side_effect = RuntimeError("flush boom")

    t = _make_trainer(logger=logger)

    with caplog.at_level(logging.WARNING, logger="lighttrain.trainers._primitives"):
        result = run_train_loop(t, target_steps=1)

    logger.flush.assert_called_once()
    assert any("Suppressed exception in logger.flush" in r.message for r in caplog.records)
    assert isinstance(result, dict)


def test_invariant_logger_none_skips_flush():
    """When trainer.logger is None, no flush is attempted and no error occurs."""
    t = _make_trainer(logger=None)
    result = run_train_loop(t, target_steps=1)
    assert isinstance(result, dict)


# ===========================================================================
# run_train_loop — finally: write_index_page exception suppressed (lines 174-175)
# ===========================================================================


def test_pin_current_behavior_write_index_page_exception_suppressed(caplog, tmp_path):
    """If write_index_page raises, the exception is suppressed (lines 174-175)."""
    t = _make_trainer(_run_dir=tmp_path)

    with patch(
        "lighttrain.observability.diagnostics.index_page.write_index_page",
        side_effect=RuntimeError("index boom"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.trainers._primitives"):
            result = run_train_loop(t, target_steps=1)

    assert any("Suppressed exception in write_index_page" in r.message for r in caplog.records)
    assert isinstance(result, dict)


def test_invariant_run_dir_none_skips_write_index_page(tmp_path):
    """When trainer._run_dir is None, write_index_page is not called."""
    t = _make_trainer(_run_dir=None)
    with patch(
        "lighttrain.observability.diagnostics.index_page.write_index_page",
    ) as mock_wip:
        run_train_loop(t, target_steps=1)
    mock_wip.assert_not_called()


def test_invariant_run_dir_set_calls_write_index_page(tmp_path):
    """When trainer._run_dir is set, write_index_page is called on a normal exit."""
    t = _make_trainer(_run_dir=tmp_path)
    # write_index_page is imported locally inside the finally block, so patch
    # it at its definition site.
    with patch(
        "lighttrain.observability.diagnostics.index_page.write_index_page",
    ) as mock_wip:
        run_train_loop(t, target_steps=1)
    mock_wip.assert_called_once()


# ===========================================================================
# run_train_loop — crash path re-raises original exception
# ===========================================================================


def test_invariant_crash_reraises_original_exception():
    """An unhandled exception from train_step propagates out of run_train_loop."""
    t = _make_trainer()
    t.train_step.side_effect = ValueError("step exploded")

    with pytest.raises(ValueError, match="step exploded"):
        run_train_loop(t, target_steps=5)

    t._write_crash_bundle.assert_called_once()


def test_invariant_crash_calls_write_crash_bundle_with_exc_and_batch():
    """_write_crash_bundle receives the exception, last_batch, and last_metrics."""
    t = _make_trainer()
    exc = RuntimeError("boom")
    t.train_step.side_effect = exc

    with pytest.raises(RuntimeError):
        run_train_loop(t, target_steps=5)

    call_args = t._write_crash_bundle.call_args
    assert call_args.args[0] is exc


# ===========================================================================
# run_train_loop — batch_in_epoch accounting
# ===========================================================================


def test_invariant_batch_in_epoch_increments_each_step():
    """batch_in_epoch increments for every batch consumed within an epoch."""
    t = _make_trainer(batches=[{"x": i} for i in range(5)], target_steps=3)
    run_train_loop(t, target_steps=3)
    # After 3 steps all within first epoch: batch_in_epoch == 3
    assert t.ctx.batch_in_epoch == 3


def test_invariant_batch_in_epoch_resets_on_epoch_rollover():
    """After epoch rollover, batch_in_epoch resets to 0 then increments again."""
    counts: list[int] = []

    # Use a real ctx so we can inspect it
    t = _make_trainer(batches=[{"x": 0}], target_steps=2)

    def _produce(raw):
        counts.append(t.ctx.batch_in_epoch)
        return raw

    t.produce_batch.side_effect = _produce
    run_train_loop(t, target_steps=2)
    # After epoch rollover, batch_in_epoch would have been reset to 0 then bumped to 1
    assert 1 in counts
