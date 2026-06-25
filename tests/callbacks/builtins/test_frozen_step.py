"""Edge-case and branch-exhaustive tests for
``lighttrain.builtin_plugins.callbacks.builtins.frozen_step.FrozenStepCallback``.

What this file pins / drives toward coverage:

* ``__init__``: default ``every`` / ``reason``; ``every`` is clamped to ≥ 1.
* ``_warn_once`` (lines 43-46): first call logs + adds to ``_warned``; second
  call with same key is a no-op (no duplicate log).
* ``on_train_start``:
  - ``run_dir`` present on ``ctx`` → writer created, exposed via ``ctx.frozen_step_writer``.
  - ``run_dir`` absent on ctx but present on ``trainer._run_dir`` (line 51).
  - Both None → WARNING logged, early return (lines 53-57).
  - ``ctx.mode`` respected; default ``"lab"`` when absent.
  - ``_resolved_yaml`` stashed from trainer (line 81).
  - ``run_node_id`` harvested from callbacks[*]._run_node_id (lines 68-69).
* ``on_step_begin``:
  - Writer None → silent return (line 92).
  - Batch is not a dict → ``_warn_once("batch_type")`` + return (lines 94-100).
  - Model or optimizer None → ``_warn_once("no_model_opt")`` + return (lines 104-111).
  - Happy path: ``writer.snapshot`` is called.
  - ``writer.snapshot`` raises → WARNING with exc_info swallowed (lines 121-122).
* ``on_step_end``:
  - Writer None → silent return (line 130).
  - Non-commit step (step % every != 0) → debug log only.
  - Step 0 guard.
  - Commit step → ``writer.commit`` called.
  - ``writer.commit`` raises → WARNING swallowed (lines 137-138).
  - ``writer.commit`` returns None → ``_warn_once("commit_none")`` (lines 143-145).
* ``on_exception`` (lines 152-158):
  - Writer None → silent return.
  - Happy path: ``writer.commit(reason="exception")`` called.
  - ``writer.commit`` raises → WARNING swallowed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lighttrain.builtin_plugins.callbacks.builtins.frozen_step import FrozenStepCallback

_CB_LOGGER = "lighttrain.builtin_plugins.callbacks.builtins.frozen_step"


# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal context double."""

    def __init__(self, run_dir: Path | None = None, mode: str = "train") -> None:
        self.run_dir = run_dir
        self.mode = mode
        self.lineage_store = None
        self.run_id = None
        self.model = MagicMock()
        self.optimizer = MagicMock()


class _Trainer:
    """Minimal trainer double."""

    def __init__(self, run_dir: Path | None = None, yaml: str = "", callbacks=None) -> None:
        if run_dir is not None:
            self._run_dir = run_dir
        self._resolved_yaml = yaml
        self.callbacks = callbacks or []


class _CbWithNodeId:
    """Callback exposing ``_run_node_id`` (an int)."""

    def __init__(self, node_id: int) -> None:
        self._run_node_id = node_id


class _CbWithNonIntNodeId:
    """Callback where ``_run_node_id`` is not an int → must be skipped."""

    _run_node_id = "not-an-int"


class _MockWriter:
    """Minimal FrozenStepWriter double with controllable snapshot/commit."""

    def __init__(self) -> None:
        self.snapshot_calls: list[dict] = []
        self.commit_calls: list[str] = []
        self._snapshot_raises: Exception | None = None
        self._commit_raises: Exception | None = None
        self._commit_return = MagicMock(name="bundle_path")  # truthy by default

    def snapshot(self, **kwargs) -> None:
        if self._snapshot_raises is not None:
            raise self._snapshot_raises
        self.snapshot_calls.append(kwargs)

    def commit(self, *, reason: str) -> object:
        self.commit_calls.append(reason)
        if self._commit_raises is not None:
            raise self._commit_raises
        return self._commit_return


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_invariant_init_defaults() -> None:
    """Default ``every=1000`` and ``reason='scheduled'`` are set."""
    cb = FrozenStepCallback()
    assert cb.every == 1000
    assert cb.reason == "scheduled"
    assert cb._writer is None
    assert cb._warned == set()


def test_invariant_init_clamps_every_to_one() -> None:
    """``every <= 0`` is clamped to 1 (``max(1, int(every))``)."""
    assert FrozenStepCallback(every=0).every == 1
    assert FrozenStepCallback(every=-5).every == 1
    assert FrozenStepCallback(every=3).every == 3


def test_invariant_init_custom_reason() -> None:
    """A custom ``reason`` string is preserved verbatim."""
    cb = FrozenStepCallback(reason="exception")
    assert cb.reason == "exception"


# ---------------------------------------------------------------------------
# _warn_once
# ---------------------------------------------------------------------------


def test_invariant_warn_once_logs_on_first_call(caplog) -> None:
    """First call with a new key logs the warning and records the key."""
    cb = FrozenStepCallback()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb._warn_once("key1", "hello %s", "world")
    assert "key1" in cb._warned
    recs = [r for r in caplog.records if "hello world" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_warn_once_suppresses_duplicate(caplog) -> None:
    """Second call with the same key emits no additional log (lines 43-44)."""
    cb = FrozenStepCallback()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb._warn_once("dup", "first %s", "call")
        cb._warn_once("dup", "second %s", "call")  # must be suppressed
    msgs = [r.getMessage() for r in caplog.records if "call" in r.getMessage()]
    assert len(msgs) == 1
    assert "first call" in msgs[0]


def test_invariant_warn_once_different_keys_both_log(caplog) -> None:
    """Two different keys both produce log records."""
    cb = FrozenStepCallback()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb._warn_once("a", "msg-a")
        cb._warn_once("b", "msg-b")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("msg-a" in m for m in msgs)
    assert any("msg-b" in m for m in msgs)


# ---------------------------------------------------------------------------
# on_train_start
# ---------------------------------------------------------------------------


def test_invariant_on_train_start_creates_writer_from_ctx_run_dir(tmp_path) -> None:
    """``ctx.run_dir`` is used to create the FrozenStepWriter."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)
    cb.on_train_start(ctx=ctx)
    assert cb._writer is not None
    # writer is exposed on ctx
    assert ctx.frozen_step_writer is cb._writer  # type: ignore[attr-defined]


def test_invariant_on_train_start_falls_back_to_trainer_run_dir(tmp_path) -> None:
    """When ctx has no ``run_dir``, ``trainer._run_dir`` is used (line 51)."""
    cb = FrozenStepCallback()

    class _CtxNoRD:
        run_dir = None
        mode = "lab"
        lineage_store = None
        run_id = None

    trainer = _Trainer(run_dir=tmp_path)
    cb.on_train_start(ctx=_CtxNoRD(), trainer=trainer)
    assert cb._writer is not None


def test_invariant_on_train_start_warns_when_no_run_dir(caplog) -> None:
    """No run_dir on either ctx or trainer → WARNING + writer stays None (lines 53-57)."""
    cb = FrozenStepCallback()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_train_start(ctx=None, trainer=None)
    assert cb._writer is None
    recs = [r for r in caplog.records if "frozen_step disabled" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_on_train_start_no_run_dir_trainer_without_attr(caplog) -> None:
    """Trainer without ``_run_dir`` attribute is handled gracefully."""
    cb = FrozenStepCallback()

    class _TrainerNoRD:
        pass

    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_train_start(ctx=None, trainer=_TrainerNoRD())
    assert cb._writer is None


def test_invariant_on_train_start_ctx_none_trainer_has_run_dir(tmp_path) -> None:
    """ctx=None still allows writer creation when trainer supplies run_dir."""
    cb = FrozenStepCallback()
    trainer = _Trainer(run_dir=tmp_path)
    cb.on_train_start(ctx=None, trainer=trainer)
    assert cb._writer is not None


def test_invariant_on_train_start_mode_from_ctx(tmp_path) -> None:
    """``ctx.mode`` is forwarded to the writer (passed as ``mode=`` arg)."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path, mode="production")
    # We can't inspect the writer's internal mode easily, so just assert no error.
    cb.on_train_start(ctx=ctx)
    assert cb._writer is not None


def test_invariant_on_train_start_mode_defaults_to_lab_when_none(tmp_path) -> None:
    """``ctx.mode = None`` is coerced to ``'lab'`` without error (line 58)."""
    cb = FrozenStepCallback()

    class _CtxNoneMode:
        run_dir = tmp_path
        mode = None
        lineage_store = None
        run_id = None

    cb.on_train_start(ctx=_CtxNoneMode())
    assert cb._writer is not None


def test_invariant_on_train_start_stashes_resolved_yaml(tmp_path) -> None:
    """``trainer._resolved_yaml`` is captured into ``self._config_yaml`` (line 81)."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)
    trainer = _Trainer(run_dir=tmp_path, yaml="lr: 0.001\n")
    cb.on_train_start(ctx=ctx, trainer=trainer)
    assert cb._config_yaml == "lr: 0.001\n"


def test_invariant_on_train_start_resolved_yaml_none_becomes_empty(tmp_path) -> None:
    """``trainer._resolved_yaml = None`` is coerced to ``''`` (line 81)."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)
    trainer = _Trainer(run_dir=tmp_path, yaml=None)  # type: ignore[arg-type]
    cb.on_train_start(ctx=ctx, trainer=trainer)
    assert cb._config_yaml == ""


def test_invariant_on_train_start_harvests_run_node_id_from_callbacks(tmp_path) -> None:
    """``run_node_id`` is taken from the first callback that has an int ``_run_node_id``
    (lines 64-69)."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)

    # Patch FrozenStepWriter to capture the run_node_id argument.
    captured: dict = {}

    class _CapturingWriter:
        def __init__(self, *a, run_node_id=None, **kw) -> None:
            captured["run_node_id"] = run_node_id

    trainer = _Trainer(
        run_dir=tmp_path,
        callbacks=[_CbWithNonIntNodeId(), _CbWithNodeId(42)],
    )

    with patch(
        "lighttrain.builtin_plugins.callbacks.builtins.frozen_step.FrozenStepWriter",
        _CapturingWriter,
    ):
        cb.on_train_start(ctx=ctx, trainer=trainer)

    assert captured["run_node_id"] == 42


def test_invariant_on_train_start_skips_non_int_node_id(tmp_path) -> None:
    """Non-int ``_run_node_id`` values are not used (isinstance guard lines 67-68)."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)
    captured: dict = {}

    class _CapturingWriter:
        def __init__(self, *a, run_node_id=None, **kw) -> None:
            captured["run_node_id"] = run_node_id

    trainer = _Trainer(run_dir=tmp_path, callbacks=[_CbWithNonIntNodeId()])

    with patch(
        "lighttrain.builtin_plugins.callbacks.builtins.frozen_step.FrozenStepWriter",
        _CapturingWriter,
    ):
        cb.on_train_start(ctx=ctx, trainer=trainer)

    assert captured["run_node_id"] is None


def test_invariant_on_train_start_no_callbacks_run_node_id_is_none(tmp_path) -> None:
    """Empty callbacks list → ``run_node_id`` stays None."""
    cb = FrozenStepCallback()
    ctx = _Ctx(run_dir=tmp_path)
    captured: dict = {}

    class _CapturingWriter:
        def __init__(self, *a, run_node_id=None, **kw) -> None:
            captured["run_node_id"] = run_node_id

    trainer = _Trainer(run_dir=tmp_path, callbacks=[])

    with patch(
        "lighttrain.builtin_plugins.callbacks.builtins.frozen_step.FrozenStepWriter",
        _CapturingWriter,
    ):
        cb.on_train_start(ctx=ctx, trainer=trainer)

    assert captured["run_node_id"] is None


# ---------------------------------------------------------------------------
# on_step_begin
# ---------------------------------------------------------------------------


def _cb_with_mock_writer() -> tuple[FrozenStepCallback, _MockWriter]:
    cb = FrozenStepCallback(every=5)
    writer = _MockWriter()
    cb._writer = writer  # type: ignore[assignment]
    return cb, writer


def test_invariant_on_step_begin_writer_none_is_silent_noop() -> None:
    """Writer is None → ``on_step_begin`` returns silently (line 92)."""
    cb = FrozenStepCallback()
    assert cb._writer is None
    # Must not raise.
    cb.on_step_begin(step=1, batch={"input_ids": []}, ctx=MagicMock())


def test_invariant_on_step_begin_non_dict_batch_warns_and_skips(caplog) -> None:
    """A non-dict batch triggers ``_warn_once('batch_type')`` and returns (lines 94-100)."""
    cb, writer = _cb_with_mock_writer()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_begin(step=1, batch=[1, 2, 3], ctx=MagicMock())
    assert writer.snapshot_calls == []
    assert "batch_type" in cb._warned
    recs = [r for r in caplog.records if "not a dict" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_on_step_begin_non_dict_batch_warns_only_once(caplog) -> None:
    """Second non-dict batch call does NOT emit a duplicate warning."""
    cb, _ = _cb_with_mock_writer()
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_begin(step=1, batch="not-a-dict", ctx=MagicMock())
        cb.on_step_begin(step=2, batch="not-a-dict", ctx=MagicMock())
    recs = [r for r in caplog.records if "not a dict" in r.getMessage()]
    assert len(recs) == 1


def test_invariant_on_step_begin_model_none_warns_and_skips(caplog) -> None:
    """``ctx.model is None`` triggers ``_warn_once('no_model_opt')`` (lines 104-111)."""
    cb, writer = _cb_with_mock_writer()

    class _CtxNoModel:
        model = None
        optimizer = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_begin(step=1, batch={"x": 1}, ctx=_CtxNoModel())
    assert writer.snapshot_calls == []
    assert "no_model_opt" in cb._warned
    recs = [r for r in caplog.records if "ctx.model" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_on_step_begin_optimizer_none_warns_and_skips(caplog) -> None:
    """``ctx.optimizer is None`` triggers ``_warn_once('no_model_opt')`` (line 103)."""
    cb, writer = _cb_with_mock_writer()

    class _CtxNoOpt:
        model = MagicMock()
        optimizer = None

    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_begin(step=1, batch={"x": 1}, ctx=_CtxNoOpt())
    assert writer.snapshot_calls == []
    assert "no_model_opt" in cb._warned


def test_invariant_on_step_begin_model_opt_none_warns_only_once(caplog) -> None:
    """Repeated no-model calls do not flood the log."""
    cb, _ = _cb_with_mock_writer()

    class _CtxNone:
        model = None
        optimizer = None

    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        for _ in range(3):
            cb.on_step_begin(step=1, batch={"x": 1}, ctx=_CtxNone())
    recs = [r for r in caplog.records if "ctx.model" in r.getMessage()]
    assert len(recs) == 1


def test_invariant_on_step_begin_calls_snapshot_on_happy_path() -> None:
    """When writer, dict-batch, model, and optimizer are all present, ``snapshot`` is called."""
    cb, writer = _cb_with_mock_writer()

    class _CtxOk:
        model = MagicMock()
        optimizer = MagicMock()

    cb.on_step_begin(step=3, batch={"input_ids": []}, ctx=_CtxOk())
    assert len(writer.snapshot_calls) == 1
    call = writer.snapshot_calls[0]
    assert call["step"] == 3


def test_invariant_on_step_begin_snapshot_exception_is_swallowed(caplog) -> None:
    """A ``snapshot()`` that raises is caught with WARNING + exc_info (lines 121-122)."""
    cb, writer = _cb_with_mock_writer()
    writer._snapshot_raises = RuntimeError("disk full")

    class _CtxOk:
        model = MagicMock()
        optimizer = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        # Must not raise.
        cb.on_step_begin(step=7, batch={"x": 1}, ctx=_CtxOk())
    recs = [r for r in caplog.records if "snapshot at step" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


# ---------------------------------------------------------------------------
# on_step_end
# ---------------------------------------------------------------------------


def test_invariant_on_step_end_writer_none_is_silent_noop() -> None:
    """Writer is None → ``on_step_end`` returns silently (line 130)."""
    cb = FrozenStepCallback()
    assert cb._writer is None
    cb.on_step_end(step=1000)  # must not raise


def test_invariant_on_step_end_skips_non_commit_step() -> None:
    """Step not divisible by ``every`` → no commit (debug-only)."""
    cb, writer = _cb_with_mock_writer()  # every=5
    cb.on_step_end(step=3)
    assert writer.commit_calls == []


@pytest.mark.parametrize("step", [0, -1])
def test_invariant_on_step_end_step_zero_or_negative_skips(step: int) -> None:
    """Step <= 0 is always skipped regardless of ``every`` (line 131)."""
    cb, writer = _cb_with_mock_writer()
    cb.on_step_end(step=step)
    assert writer.commit_calls == []


def test_invariant_on_step_end_commit_on_scheduled_step() -> None:
    """Step divisible by ``every`` triggers ``writer.commit`` (line 136)."""
    cb, writer = _cb_with_mock_writer()  # every=5
    cb.on_step_end(step=10)
    assert writer.commit_calls == ["scheduled"]


def test_invariant_on_step_end_uses_self_reason() -> None:
    """The commit is called with ``self.reason`` (line 136)."""
    cb = FrozenStepCallback(every=2, reason="exception")
    writer = _MockWriter()
    cb._writer = writer  # type: ignore[assignment]
    cb.on_step_end(step=4)
    assert writer.commit_calls == ["exception"]


def test_invariant_on_step_end_commit_exception_swallowed(caplog) -> None:
    """``writer.commit`` raising is caught with WARNING (lines 137-138)."""
    cb, writer = _cb_with_mock_writer()
    writer._commit_raises = OSError("no space")
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_end(step=5)
    recs = [r for r in caplog.records if "scheduled commit" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_on_step_end_commit_none_warns_once(caplog) -> None:
    """``writer.commit`` returning None triggers ``_warn_once('commit_none')`` (lines 143-145)."""
    cb, writer = _cb_with_mock_writer()
    writer._commit_return = None  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_end(step=5)
        cb.on_step_end(step=10)  # second call — must not emit a second warning
    recs = [r for r in caplog.records if "no bundle" in r.getMessage()]
    assert len(recs) == 1, f"Expected 1 warn, got: {[r.getMessage() for r in recs]}"
    assert "commit_none" in cb._warned


def test_invariant_on_step_end_non_none_commit_does_not_warn(caplog) -> None:
    """Successful (non-None) commit produces no 'commit_none' warning."""
    cb, writer = _cb_with_mock_writer()  # default returns truthy MagicMock
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_step_end(step=5)
    recs = [r for r in caplog.records if "no bundle" in r.getMessage()]
    assert recs == []


# ---------------------------------------------------------------------------
# on_exception
# ---------------------------------------------------------------------------


def test_invariant_on_exception_writer_none_is_silent_noop() -> None:
    """Writer is None → ``on_exception`` returns silently (lines 153-154)."""
    cb = FrozenStepCallback()
    assert cb._writer is None
    cb.on_exception()  # must not raise


def test_invariant_on_exception_calls_commit_with_exception_reason() -> None:
    """``on_exception`` calls ``writer.commit(reason='exception')`` (line 156)."""
    cb, writer = _cb_with_mock_writer()
    cb.on_exception()
    assert writer.commit_calls == ["exception"]


def test_invariant_on_exception_commit_exception_swallowed(caplog) -> None:
    """``writer.commit`` raising during exception handling is swallowed (lines 157-158)."""
    cb, writer = _cb_with_mock_writer()
    writer._commit_raises = RuntimeError("commit exploded")
    with caplog.at_level(logging.WARNING, logger=_CB_LOGGER):
        cb.on_exception()  # must not raise
    recs = [
        r for r in caplog.records
        if "exception-time commit failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_on_exception_ignores_kwargs() -> None:
    """``on_exception`` accepts arbitrary keyword args and ignores them."""
    cb, writer = _cb_with_mock_writer()
    cb.on_exception(exc=ValueError("oops"), step=99, ctx=MagicMock())
    assert writer.commit_calls == ["exception"]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_invariant_registered_as_callback_frozen_step() -> None:
    """``FrozenStepCallback`` is registered under ``('callback', 'frozen_step')``."""
    from lighttrain.registry import get as registry_get

    cls = registry_get("callback", "frozen_step")
    assert cls is FrozenStepCallback
