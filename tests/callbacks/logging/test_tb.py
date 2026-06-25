"""Adversarial tests for ``lighttrain.builtin_plugins.callbacks.logging.tb.TensorBoardLogger``.

Coverage:
* **No log_dir and no run_dir raises ValueError** (lines 25-27).
* **run_dir mode sets log_dir to <run_dir>/logs** (line 28).
* **log_dir is stored as Path and created on init** (lines 29-30).
* **SummaryWriter lazy-imported on __init__** (lines 32-34).
* **log_scalars iterates all keys, calls add_scalar** (lines 37-39).
* **log_scalars skips unconvertible values via continue** (line 40-41).
* **log_text calls add_text with "text" tag** (line 44).
* **log_artifact calls add_text with artifact path and name** (line 47).
* **flush calls writer.flush()** (line 50).
* **close flushes then closes writer** (lines 53-55).
* **close exception path logs warning** (lines 53-57).
* **Registered under both ('logger', 'tensorboard') and ('logger', 'tb')**.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lighttrain.builtin_plugins.callbacks.logging.tb import TensorBoardLogger

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_TB_IMPORT_PATH = "torch.utils.tensorboard.SummaryWriter"


def _patched_logger(log_dir: Path) -> tuple[TensorBoardLogger, MagicMock]:
    """Return (logger, mock_writer) by patching SummaryWriter at the source level.

    Because the import is lazy (inside __init__), we patch the torch module
    attribute that the ``from torch.utils.tensorboard import SummaryWriter``
    statement resolves.
    """
    mock_writer = MagicMock()
    with patch(_TB_IMPORT_PATH, return_value=mock_writer):
        logger = TensorBoardLogger(log_dir=log_dir)
    return logger, mock_writer


# ---------------------------------------------------------------------------
# __init__ — error conditions (lines 25-30)
# ---------------------------------------------------------------------------


def test_invariant_no_log_dir_no_run_dir_raises_value_error():
    """TensorBoardLogger() with neither log_dir nor run_dir must raise ValueError.

    Pins lines 25-27: ``if log_dir is None: if run_dir is None: raise ValueError``.
    """
    with pytest.raises(ValueError, match="(?i)log_dir|run_dir"):
        TensorBoardLogger()


def test_invariant_run_dir_sets_log_dir_to_logs_subdir(tmp_path: Path):
    """When only run_dir is given, log_dir resolves to <run_dir>/logs (line 28).

    Pins: ``log_dir = Path(run_dir) / "logs"``
    """
    logger = TensorBoardLogger(run_dir=tmp_path)
    assert logger.log_dir == tmp_path / "logs"
    logger.close()


def test_invariant_log_dir_stored_as_path(tmp_path: Path):
    """self.log_dir is a Path object regardless of whether a str was passed (line 29).

    Pins: ``self.log_dir = Path(log_dir)``
    """
    log_dir = tmp_path / "mylogs"
    logger = TensorBoardLogger(log_dir=str(log_dir))
    assert isinstance(logger.log_dir, Path)
    logger.close()


def test_invariant_log_dir_created_on_init(tmp_path: Path):
    """The log directory is created by __init__ even if it did not exist (line 30).

    Pins: ``self.log_dir.mkdir(parents=True, exist_ok=True)``
    """
    nested = tmp_path / "a" / "b" / "c" / "logs"
    assert not nested.exists()
    logger = TensorBoardLogger(log_dir=nested)
    assert nested.is_dir()
    logger.close()


def test_invariant_explicit_log_dir_not_overridden_by_run_dir(tmp_path: Path):
    """When both log_dir and run_dir are provided, log_dir wins (lines 25-26).

    Pins: ``if log_dir is None`` — so a non-None log_dir is used as-is.
    """
    explicit = tmp_path / "explicit_logs"
    logger = TensorBoardLogger(log_dir=explicit, run_dir=tmp_path)
    assert logger.log_dir == explicit
    logger.close()


# ---------------------------------------------------------------------------
# SummaryWriter lazy-import (lines 32-34)
# ---------------------------------------------------------------------------


def test_invariant_summary_writer_instantiated_in_init(tmp_path: Path):
    """SummaryWriter is constructed once during __init__ with the log_dir string.

    Pins lines 32-34:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
    """
    log_dir = tmp_path / "logs"
    mock_instance = MagicMock()
    with patch(_TB_IMPORT_PATH, return_value=mock_instance) as MockSW:
        logger = TensorBoardLogger(log_dir=log_dir)
    MockSW.assert_called_once_with(log_dir=str(log_dir))
    assert logger._writer is mock_instance


# ---------------------------------------------------------------------------
# log_scalars (lines 37-41)
# ---------------------------------------------------------------------------


def test_invariant_log_scalars_calls_add_scalar_per_key(tmp_path: Path):
    """log_scalars iterates all keys and calls add_scalar for each (lines 37-39).

    Setup: scalars = {"loss": 0.5, "acc": 0.9}, step=10.
    Expected: add_scalar called twice with correct args.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_scalars({"loss": 0.5, "acc": 0.9}, step=10)
    assert mock_writer.add_scalar.call_count == 2
    mock_writer.add_scalar.assert_any_call("loss", 0.5, 10)
    mock_writer.add_scalar.assert_any_call("acc", 0.9, 10)


def test_invariant_log_scalars_coerces_value_to_float(tmp_path: Path):
    """add_scalar receives float(v) not raw v (line 39 — explicit ``float(v)`` cast).

    Setup: pass integer 1; expected: add_scalar called with float 1.0.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_scalars({"step_int": 1}, step=3)
    mock_writer.add_scalar.assert_called_once_with("step_int", 1.0, 3)
    _, positional, _ = mock_writer.add_scalar.mock_calls[0]
    assert isinstance(positional[1], float)


def test_invariant_log_scalars_coerces_step_to_int(tmp_path: Path):
    """add_scalar receives int(step) (line 39 — explicit ``int(step)`` cast).

    Setup: pass float step=3.7; expected: add_scalar called with step=3.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_scalars({"loss": 0.5}, step=3.7)  # type: ignore[arg-type]
    mock_writer.add_scalar.assert_called_once()
    _name, _val, step_arg = mock_writer.add_scalar.call_args[0]
    assert step_arg == 3
    assert isinstance(step_arg, int)


def test_invariant_log_scalars_skips_unconvertible_value_continues(tmp_path: Path):
    """Unconvertible values (TypeError) are silently skipped via continue (line 40-41).

    Setup: {"bad": <raises TypeError on float()>, "loss": 0.3}.
    Expected: add_scalar called exactly once (for "loss" only).
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")

    class _BadFloat:
        def __float__(self):
            raise TypeError("cannot convert")

    logger.log_scalars({"bad": _BadFloat(), "loss": 0.3}, step=1)  # type: ignore[dict-item]
    assert mock_writer.add_scalar.call_count == 1
    mock_writer.add_scalar.assert_called_once_with("loss", 0.3, 1)


def test_invariant_log_scalars_skips_value_error_continues(tmp_path: Path):
    """ValueError is also caught and the loop continues (line 40: ``except (TypeError, ValueError)``).

    Setup: {"skip": <raises ValueError>, "keep": 0.7}.
    Expected: add_scalar called once for "keep" only.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")

    class _BadValue:
        def __float__(self):
            raise ValueError("bad value")

    logger.log_scalars({"skip": _BadValue(), "keep": 0.7}, step=5)  # type: ignore[dict-item]
    assert mock_writer.add_scalar.call_count == 1
    mock_writer.add_scalar.assert_called_once_with("keep", 0.7, 5)


def test_invariant_log_scalars_empty_dict_no_op(tmp_path: Path):
    """log_scalars({}, step=1) does not call add_scalar at all (empty iteration)."""
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_scalars({}, step=1)
    mock_writer.add_scalar.assert_not_called()


# ---------------------------------------------------------------------------
# log_text (line 44)
# ---------------------------------------------------------------------------


def test_invariant_log_text_calls_add_text_with_text_tag(tmp_path: Path):
    """log_text calls add_text('text', text, int(step)) (line 44).

    Setup: log_text("hello world", step=7).
    Expected: add_text called with tag='text', text='hello world', step=7.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_text("hello world", step=7)
    mock_writer.add_text.assert_called_once_with("text", "hello world", 7)


def test_invariant_log_text_coerces_step_to_int(tmp_path: Path):
    """add_text receives int(step) (line 44 explicit int(step) cast).

    Setup: step=2.9 (float).
    Expected: add_text called with step=2 (int).
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_text("msg", step=2.9)  # type: ignore[arg-type]
    _tag, _text, step_arg = mock_writer.add_text.call_args[0]
    assert step_arg == 2
    assert isinstance(step_arg, int)


def test_invariant_log_text_unicode_payload(tmp_path: Path):
    """log_text passes unicode payload through unchanged.

    Setup: Chinese + emoji payload.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_text("你好 🚀", step=1)
    mock_writer.add_text.assert_called_once_with("text", "你好 🚀", 1)


# ---------------------------------------------------------------------------
# log_artifact (line 47)
# ---------------------------------------------------------------------------


def test_invariant_log_artifact_calls_add_text_with_name_and_path(tmp_path: Path):
    """log_artifact(path, name=...) calls add_text('artifact', '<name> -> <path>') (line 47).

    No step argument — add_text is called with two positional args.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_artifact("/tmp/model.pt", name="weights")
    mock_writer.add_text.assert_called_once_with("artifact", "weights -> /tmp/model.pt")


def test_invariant_log_artifact_none_name_uses_empty_string(tmp_path: Path):
    """When name=None (default), the prefix is empty string (line 47: ``name or ''``).

    Setup: log_artifact("/tmp/x.bin") with no name.
    Expected: add_text called with ' -> /tmp/x.bin'.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_artifact("/tmp/x.bin")
    mock_writer.add_text.assert_called_once_with("artifact", " -> /tmp/x.bin")


def test_invariant_log_artifact_explicit_none_name_uses_empty_string(tmp_path: Path):
    """name=None explicitly → same as omitting name: prefix is ''.

    Setup: log_artifact("/tmp/x.bin", name=None).
    Expected: text is ' -> /tmp/x.bin'.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.log_artifact("/tmp/x.bin", name=None)
    mock_writer.add_text.assert_called_once_with("artifact", " -> /tmp/x.bin")


# ---------------------------------------------------------------------------
# flush (line 50)
# ---------------------------------------------------------------------------


def test_invariant_flush_calls_writer_flush(tmp_path: Path):
    """flush() delegates to self._writer.flush() (line 50)."""
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    logger.flush()
    mock_writer.flush.assert_called_once()


# ---------------------------------------------------------------------------
# close (lines 52-57)
# ---------------------------------------------------------------------------


def test_invariant_close_calls_flush_then_close(tmp_path: Path):
    """close() calls writer.flush() then writer.close() in order (lines 53-55).

    Pins: the normal close path.
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    call_order: list[str] = []
    mock_writer.flush.side_effect = lambda: call_order.append("flush")
    mock_writer.close.side_effect = lambda: call_order.append("close")

    logger.close()

    mock_writer.flush.assert_called_once()
    mock_writer.close.assert_called_once()
    assert call_order == ["flush", "close"], f"Expected flush then close; got {call_order}"


def test_invariant_close_exception_swallowed(tmp_path: Path):
    """When writer.flush raises, close() does not propagate the exception.

    Pins the ``except Exception`` guard on line 56 (noted pragma: no cover
    in source, but still reachable via a mock).
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    mock_writer.flush.side_effect = OSError("disk full")

    # Must not raise
    logger.close()


def test_invariant_close_logs_warning_on_exception(tmp_path: Path, caplog):
    """close() swallows exceptions from writer.flush/close and logs a warning.

    Pins the ``_log.warning(...)`` call on line 57.
    """
    import logging

    logger, mock_writer = _patched_logger(tmp_path / "logs")
    mock_writer.flush.side_effect = OSError("disk full")

    with caplog.at_level(
        logging.WARNING,
        logger="lighttrain.builtin_plugins.callbacks.logging.tb",
    ):
        logger.close()

    assert len(caplog.records) >= 1, "Expected at least one WARNING log record"
    messages = " ".join(r.message for r in caplog.records)
    assert "tensorboard logger" in messages.lower() or "close" in messages.lower()


def test_invariant_close_writer_close_exception_swallowed(tmp_path: Path):
    """When writer.close() raises (after flush succeeds), the exception is swallowed.

    Pins: the except clause covers the whole try block including writer.close().
    """
    logger, mock_writer = _patched_logger(tmp_path / "logs")
    mock_writer.close.side_effect = RuntimeError("writer already closed")

    logger.close()  # must not raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_invariant_registered_as_logger_tensorboard():
    """TensorBoardLogger is registered under ('logger', 'tensorboard')."""
    from lighttrain.registry import get

    cls = get("logger", "tensorboard")
    assert cls is TensorBoardLogger


def test_invariant_registered_as_logger_tb():
    """TensorBoardLogger is also registered under ('logger', 'tb') (line 15)."""
    from lighttrain.registry import get

    cls = get("logger", "tb")
    assert cls is TensorBoardLogger


# ---------------------------------------------------------------------------
# Real integration: round-trip with actual SummaryWriter
# ---------------------------------------------------------------------------


def test_integration_log_scalars_writes_event_file(tmp_path: Path):
    """Smoke test: a real TensorBoardLogger writes event files to log_dir.

    This exercises the actual SummaryWriter path (no mock) so that lines
    25-34 are covered by a real instantiation.
    """
    log_dir = tmp_path / "logs"
    logger = TensorBoardLogger(log_dir=log_dir)
    logger.log_scalars({"loss": 0.5, "acc": 0.8}, step=1)
    logger.log_text("epoch done", step=1)
    logger.log_artifact("/tmp/model.pt", name="ckpt")
    logger.flush()
    logger.close()

    events = list(log_dir.glob("events.out.tfevents.*"))
    assert len(events) >= 1, f"No tfevents file found in {log_dir}"


def test_integration_run_dir_creates_logs_subdir(tmp_path: Path):
    """Using run_dir creates the <run_dir>/logs directory and writes there."""
    logger = TensorBoardLogger(run_dir=tmp_path)
    logger.log_scalars({"loss": 1.0}, step=0)
    logger.close()

    logs_dir = tmp_path / "logs"
    assert logs_dir.is_dir()
    events = list(logs_dir.glob("events.out.tfevents.*"))
    assert len(events) >= 1


def test_integration_log_scalars_multiple_steps(tmp_path: Path):
    """log_scalars over many steps does not raise (stress the iteration loop)."""
    logger = TensorBoardLogger(log_dir=tmp_path / "logs")
    for step in range(10):
        logger.log_scalars(
            {"train/loss": float(step), "train/acc": 1.0 / (step + 1)}, step=step
        )
    logger.close()
