"""Adversarial tests for ``lighttrain.logging._bus.LoggerBus``.

Layered on top of ``tests/test_logging_bus.py``. New coverage:

* **Per-call exception isolation** for every public method (log_text,
  log_artifact, flush, close), not just log_scalars.
* **Backend without a method is silently skipped** (line 33-35 of _bus.py).
* **log_dict empty input is a no-op** (line 56-57).
* **log_dict prefix=None uses the bare keys** (no spurious "None/" prefix).
* **log_scalar singleton wraps log_scalars**.
* **flush / close fan out to every backend**.
* **backends property is a copy**.
* **stderr message identifies the offending backend class name**.
"""

from __future__ import annotations

import pytest

from lighttrain.logging._bus import LoggerBus


class _Recorder:
    """Records every method call as a tuple (method, args, kwargs)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def log_scalars(self, scalars, step):
        self.calls.append(("log_scalars", (dict(scalars), int(step)), {}))

    def log_text(self, text, step):
        self.calls.append(("log_text", (str(text), int(step)), {}))

    def log_artifact(self, path, name=None):
        self.calls.append(("log_artifact", (str(path), name), {}))

    def flush(self):
        self.calls.append(("flush", (), {}))

    def close(self):
        self.calls.append(("close", (), {}))


class _RaisingBackend:
    """Backend that always raises in every method — used to verify isolation."""

    name = "_RaisingBackend"

    def log_scalars(self, scalars, step): raise RuntimeError("scalar boom")
    def log_text(self, text, step): raise RuntimeError("text boom")
    def log_artifact(self, path, name=None): raise RuntimeError("artifact boom")
    def flush(self): raise RuntimeError("flush boom")
    def close(self): raise RuntimeError("close boom")


class _PartialBackend:
    """Backend that ONLY has log_scalars; other methods are absent."""

    def __init__(self) -> None:
        self.calls = 0

    def log_scalars(self, scalars, step):
        self.calls += 1


# ---------------------------------------------------------------------------
# Fan-out + isolation
# ---------------------------------------------------------------------------

def test_invariant_fan_out_to_all_backends_log_scalars():
    """log_scalars goes to every backend in order."""
    a, b, c = _Recorder(), _Recorder(), _Recorder()
    bus = LoggerBus([a, b, c])
    bus.log_scalars({"loss": 0.5}, step=7)
    for r in (a, b, c):
        assert r.calls == [("log_scalars", ({"loss": 0.5}, 7), {})]


def test_invariant_fan_out_log_text():
    """log_text fans out to every backend."""
    a, b = _Recorder(), _Recorder()
    LoggerBus([a, b]).log_text("hello", step=3)
    for r in (a, b):
        assert r.calls == [("log_text", ("hello", 3), {})]


def test_invariant_fan_out_log_artifact():
    """log_artifact fans out to every backend with optional name=None."""
    rec = _Recorder()
    LoggerBus([rec]).log_artifact("/tmp/x.pt", name="model")
    assert rec.calls == [("log_artifact", ("/tmp/x.pt", "model"), {})]


def test_invariant_fan_out_flush():
    """flush fans out to every backend."""
    a, b = _Recorder(), _Recorder()
    LoggerBus([a, b]).flush()
    for r in (a, b):
        assert r.calls == [("flush", (), {})]


def test_invariant_fan_out_close():
    """close fans out to every backend."""
    a, b = _Recorder(), _Recorder()
    LoggerBus([a, b]).close()
    for r in (a, b):
        assert r.calls == [("close", (), {})]


@pytest.mark.parametrize(
    "method,args",
    [
        ("log_scalars", ({"loss": 1.0}, 1)),
        ("log_text", ("hi", 1)),
        ("log_artifact", ("/tmp/x", None)),
        ("flush", ()),
        ("close", ()),
    ],
)
def test_invariant_exception_isolation_for_every_public_method(method, args, capsys):
    """Per-call invariant: a raising backend does NOT prevent later backends
    from receiving the same call.

    Parametrized over every public LoggerBus method. The order is
    [raising, good] — verify the good backend gets through despite the
    raise.
    """
    good = _Recorder()
    bus = LoggerBus([_RaisingBackend(), good])
    getattr(bus, method)(*args)
    # The good backend's method got the call exactly once.
    assert len(good.calls) == 1
    assert good.calls[0][0] == method


def test_invariant_stderr_message_names_offending_backend_class(capsys):
    """The error message routes to stderr and contains the failing backend's
    class name plus the method name.

    Goal: pin debugging surface — without the class name the user can't tell
    which sink is broken.
    """
    bus = LoggerBus([_RaisingBackend(), _Recorder()])
    bus.log_scalars({"x": 1.0}, step=1)
    err = capsys.readouterr().err
    assert "_RaisingBackend" in err
    assert "log_scalars" in err


# ---------------------------------------------------------------------------
# Partial backend (missing methods)
# ---------------------------------------------------------------------------

def test_backend_without_method_silently_skipped():
    """If a backend lacks the method, ``_safe`` silently moves on
    (line 33-35 of _bus.py).

    Setup: a backend with ONLY log_scalars; call log_text and flush.
    Expected: no exception, log_text/flush effectively no-op for it; the
    second (full) backend still receives the calls.
    """
    partial = _PartialBackend()
    full = _Recorder()
    bus = LoggerBus([partial, full])
    # These methods don't exist on partial — must not crash.
    bus.log_text("ignored", step=1)
    bus.log_artifact("/tmp/x", name=None)
    bus.flush()
    bus.close()
    # Only log_scalars-capable methods invoked the partial backend.
    bus.log_scalars({"l": 1.0}, step=1)
    assert partial.calls == 1
    # The full backend got all 5 events.
    assert [c[0] for c in full.calls] == [
        "log_text", "log_artifact", "flush", "close", "log_scalars"
    ]


# ---------------------------------------------------------------------------
# log_dict
# ---------------------------------------------------------------------------

def test_log_dict_prefix_namespaces_keys():
    """``log_dict(d, step, prefix='train')`` adds the ``train/`` namespace
    to every key.

    Closed form: ``{"loss": 0.5}`` → backend sees ``{"train/loss": 0.5}``.
    """
    rec = _Recorder()
    LoggerBus([rec]).log_dict({"loss": 0.5}, step=3, prefix="train")
    assert rec.calls == [("log_scalars", ({"train/loss": 0.5}, 3), {})]


def test_invariant_log_dict_with_no_prefix_uses_bare_keys():
    """Pin: ``prefix=None`` (default) yields bare keys — no spurious
    ``None/`` prefix.

    Goal: catches a refactor that does ``f"{prefix}/{k}"`` unconditionally
    (which would produce ``"None/loss"``).
    """
    rec = _Recorder()
    LoggerBus([rec]).log_dict({"loss": 0.5}, step=1)
    assert rec.calls == [("log_scalars", ({"loss": 0.5}, 1), {})]


def test_invariant_log_dict_empty_input_is_no_op():
    """``log_dict({}, step)`` does NOT call any backend (line 56-57).

    Goal: catches a regression that emits an empty {} record (which would
    bloat the JSONL with no-data rows).
    """
    rec = _Recorder()
    LoggerBus([rec]).log_dict({}, step=1)
    assert rec.calls == []


def test_log_dict_coerces_values_to_float():
    """log_dict casts every value to float (line 55 of _bus.py).

    Setup: dict with int and float values.
    Expected: backend sees float values for every key.
    """
    rec = _Recorder()
    LoggerBus([rec]).log_dict({"a": 1, "b": 2.5}, step=1)
    received = rec.calls[0][1][0]
    assert all(isinstance(v, float) for v in received.values())


# ---------------------------------------------------------------------------
# log_scalar (singular) wraps log_scalars
# ---------------------------------------------------------------------------

def test_log_scalar_singular_wraps_log_scalars():
    """``log_scalar(tag, value, step)`` is implemented as
    ``log_scalars({tag: value}, step)``.
    """
    rec = _Recorder()
    LoggerBus([rec]).log_scalar("loss", 0.7, step=5)
    assert rec.calls == [("log_scalars", ({"loss": 0.7}, 5), {})]


# ---------------------------------------------------------------------------
# backends property
# ---------------------------------------------------------------------------

def test_invariant_backends_property_returns_copy_not_internal_list():
    """``bus.backends`` returns a copy; mutating it does not affect bus state."""
    rec = _Recorder()
    bus = LoggerBus([rec])
    snap = bus.backends
    snap.append("intruder")
    assert "intruder" not in bus._backends


def test_add_appends_backend_and_routes_subsequent_calls():
    """``bus.add(b)`` registers a new backend; subsequent calls reach it."""
    rec = _Recorder()
    bus = LoggerBus([])
    bus.add(rec)
    bus.log_scalars({"x": 0.0}, step=1)
    assert len(rec.calls) == 1


# ---------------------------------------------------------------------------
# Multiple raising backends do not derail order
# ---------------------------------------------------------------------------

def test_multiple_failures_do_not_stop_remaining_backends(capsys):
    """Two raising backends interleaved with two good backends: all four
    good calls succeed; all four exceptions land on stderr.
    """
    a = _Recorder()
    b = _Recorder()
    bus = LoggerBus([_RaisingBackend(), a, _RaisingBackend(), b])
    bus.log_scalars({"l": 0.1}, step=1)
    assert len(a.calls) == 1
    assert len(b.calls) == 1
    err = capsys.readouterr().err
    # Two failure messages (one per raising backend) appear on stderr.
    assert err.count("_RaisingBackend") >= 2
