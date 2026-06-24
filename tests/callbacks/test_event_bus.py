"""Adversarial tests for ``lighttrain.callbacks.base.EventBus``.

Layered on top of ``tests/test_callbacks_event_bus.py`` and
``tests/test_eventbus_critical.py``. New coverage:

* **Signal aggregation strict precedence**: STOP_TRAINING > RETRY_STEP >
  SKIP_STEP > CONTINUE — parametrize all 6 ordered pairs.
* **Quarantine after exactly max_consecutive_failures** — the failing
  callback is skipped on dispatch N+1.
* **Failure counter resets on success** (consecutive, not cumulative).
* **Critical class-name list** triggers immediate raise.
* **Critical instance attribute** also triggers immediate raise.
* **Custom on_error invoked** with (event, callback, exception).
* **_coerce** handles every input type (None / int / str / Signal).
* **callbacks property returns a copy** (not the internal list).
* **max_consecutive_failures clamped to >= 1**.
* **_DEFAULT_CRITICAL** name list pinned.
* **Main-thread-only concurrency pin** — observable signal: no
  ``threading.Lock``/``RLock`` token in source AND class docstring
  contains the literal "Thread-safety: NOT thread-safe".
"""

from __future__ import annotations

import inspect

import pytest

from lighttrain.callbacks.base import (
    _DEFAULT_CRITICAL,
    EventBus,
    Signal,
    _coerce,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _Counter:
    """Callback that increments ``calls`` and returns a configured signal."""

    def __init__(self, signal=None, *, critical_attr: bool = False) -> None:
        self.calls = 0
        self.signal = signal
        if critical_attr:
            self.critical = True

    def on_step_begin(self, **_):
        self.calls += 1
        return self.signal


class _RaisingTimes:
    """Callback that raises N times, then succeeds."""

    def __init__(self, raise_count: int) -> None:
        self.calls = 0
        self._remaining = raise_count

    def on_step_begin(self, **_):
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError(f"controlled failure {self.calls}")
        return None


class LineageRecorderCallback:
    """Same class name as the default critical entry — exception propagates."""

    def on_step_begin(self, **_):
        raise RuntimeError("critical-by-class-name")


class CheckpointCallback:
    """Another default-critical class name."""

    def on_step_begin(self, **_):
        raise RuntimeError("critical-checkpoint")


class _OnErrorRecorder:
    """Captures every on_error invocation as a tuple (event, cb_type, exc_type)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []

    def __call__(self, event, cb, exc):
        self.records.append((event, type(cb).__name__, type(exc).__name__))


# ---------------------------------------------------------------------------
# Signal aggregation strict precedence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "signals,expected",
    [
        # (signals from callbacks in order, expected aggregated result)
        ([Signal.CONTINUE, Signal.SKIP_STEP], Signal.SKIP_STEP),
        ([Signal.SKIP_STEP, Signal.CONTINUE], Signal.SKIP_STEP),
        ([Signal.SKIP_STEP, Signal.RETRY_STEP], Signal.RETRY_STEP),
        ([Signal.RETRY_STEP, Signal.SKIP_STEP], Signal.RETRY_STEP),
        ([Signal.RETRY_STEP, Signal.STOP_TRAINING], Signal.STOP_TRAINING),
        ([Signal.STOP_TRAINING, Signal.RETRY_STEP], Signal.STOP_TRAINING),
        ([Signal.CONTINUE, Signal.CONTINUE], Signal.CONTINUE),
        ([Signal.STOP_TRAINING, Signal.SKIP_STEP, Signal.RETRY_STEP], Signal.STOP_TRAINING),
    ],
)
def test_invariant_signal_aggregation_strict_precedence(signals, expected):
    """Invariant: STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE.

    The aggregator returns the MAX signal regardless of callback order
    (line 161-162 of base.py: ``if signal > result: result = signal``).

    Setup: build N callbacks each returning a specified signal; dispatch.
    Expected: returned signal == max(signals).
    """
    callbacks = [_Counter(signal=s) for s in signals]
    bus = EventBus(callbacks)
    out = bus.dispatch("on_step_begin", step=1)
    assert out == expected


# ---------------------------------------------------------------------------
# Quarantine after exactly N failures
# ---------------------------------------------------------------------------

def test_invariant_quarantine_after_exactly_max_consecutive_failures(capsys):
    """Invariant: a non-critical callback that fails ``max_consecutive_failures``
    times in a row gets quarantined; the (N+1)-th dispatch does NOT call it.

    Setup: ``max=3``; a callback that raises every time; dispatch 4 times.
    Expected: the callback is called exactly 3 times (calls 1-3 raised,
    quarantine kicked in BEFORE the 4th dispatch).
    """
    bad = _RaisingTimes(raise_count=10)  # always raises
    bus = EventBus([bad], max_consecutive_failures=3)
    for _ in range(4):
        bus.dispatch("on_step_begin", step=1)
    assert bad.calls == 3, (
        f"after 3 failures, callback should be quarantined; got {bad.calls} calls"
    )
    # quarantined property reports the class name
    assert "_RaisingTimes" in bus.quarantined


def test_invariant_other_callbacks_keep_firing_while_one_is_quarantined():
    """Invariant: quarantine is per-callback — a failing callback being
    counted toward (and then placed into) quarantine never suppresses the
    healthy callbacks sharing the same event.

    Setup: a thrower that always raises plus a quiet counter; max=3.
    Expected: across 3 failing dispatches the quiet callback fires 3 times
    and the thrower lands in ``quarantined`` only after the 3rd; a 4th
    dispatch skips the (now quarantined) thrower yet the quiet callback
    still fires (4 total).
    """
    class _Quiet:
        def __init__(self) -> None:
            self.seen = 0

        def on_step_end(self, **_):
            self.seen += 1
            return Signal.CONTINUE

    class _AlwaysThrows:
        def on_step_end(self, **_):
            raise RuntimeError("boom")

    thrower = _AlwaysThrows()
    quiet = _Quiet()
    bus = EventBus([thrower, quiet], max_consecutive_failures=3)

    bus.dispatch("on_step_end")
    assert "_AlwaysThrows" not in bus.quarantined
    bus.dispatch("on_step_end")
    assert "_AlwaysThrows" not in bus.quarantined
    bus.dispatch("on_step_end")
    assert "_AlwaysThrows" in bus.quarantined
    assert quiet.seen == 3

    # After quarantine the thrower is silently skipped; quiet still fires.
    bus.dispatch("on_step_end")
    assert quiet.seen == 4


def test_invariant_failure_counter_resets_on_successful_invocation():
    """Invariant: the failure counter is CONSECUTIVE — one successful
    invocation resets it. Two failures + one success + three more failures
    yields a quarantine after the LAST three (not the cumulative six).

    Setup: callback that raises twice, succeeds once, raises forever.
    Tick max=3.
    Expected:
        dispatch 1: raise, failure_count=1, no quarantine
        dispatch 2: raise, failure_count=2, no quarantine
        dispatch 3: success → failure_count reset to 0
        dispatch 4: raise, failure_count=1
        dispatch 5: raise, failure_count=2
        dispatch 6: raise, failure_count=3 → quarantined
        dispatch 7: skipped → calls stays at 6
    """
    class _RaiseTwiceSucceedThenRaise:
        def __init__(self) -> None:
            self.calls = 0
            self.script = ["raise", "raise", "ok", "raise", "raise", "raise"]

        def on_step_begin(self, **_):
            self.calls += 1
            action = self.script[self.calls - 1] if self.calls <= len(self.script) else "raise"
            if action == "raise":
                raise RuntimeError("scripted")

    cb = _RaiseTwiceSucceedThenRaise()
    bus = EventBus([cb], max_consecutive_failures=3)
    for _ in range(7):
        bus.dispatch("on_step_begin", step=1)
    # The success at dispatch 3 reset the counter; quarantine only fires at
    # dispatch 6 (after 3 consecutive failures 4-5-6).
    assert cb.calls == 6


# ---------------------------------------------------------------------------
# Critical callback semantics
# ---------------------------------------------------------------------------

def test_invariant_critical_callback_by_class_name_re_raises_immediately():
    """A callback whose class name is in the default critical list raises
    OUT of dispatch on the very first failure (no swallow).

    Setup: bus with one LineageRecorderCallback that always raises.
    Expected: dispatch raises RuntimeError on the first call.
    """
    bus = EventBus([LineageRecorderCallback()])
    with pytest.raises(RuntimeError):
        bus.dispatch("on_step_begin", step=1)


def test_invariant_critical_callback_by_instance_attribute_re_raises_immediately():
    """An instance with ``critical=True`` attribute is also treated as critical.

    Setup: regular _Counter with critical_attr=True, plus a non-critical
    one before it; the critical one raises via a subclass.
    Expected: the first raise propagates.
    """
    class CriticalRaiser:
        critical = True

        def on_step_begin(self, **_):
            raise ValueError("critical-by-instance-attr")

    bus = EventBus([CriticalRaiser()])
    with pytest.raises(ValueError):
        bus.dispatch("on_step_begin", step=1)


def test_invariant_critical_callback_does_not_quarantine_itself():
    """Critical callbacks bypass the quarantine machinery — the failure
    counter is NOT incremented (because the exception propagates first).

    Setup: bus with a critical callback that raises. Catch the first raise.
    Re-dispatch.
    Expected: the second dispatch ALSO raises (no quarantine). The bus's
    ``_failure_counts`` does NOT track this callback.
    """
    cb = LineageRecorderCallback()
    bus = EventBus([cb])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            bus.dispatch("on_step_begin", step=1)
    # No quarantine entry for the critical cb
    assert id(cb) not in bus._quarantined
    # No failure-count tracking either
    assert id(cb) not in bus._failure_counts


def test_pin_default_critical_names_include_lineage_checkpoint_invariants():
    """Pin: the default critical class-name list is exactly
    {LineageRecorderCallback, CheckpointCallback, InvariantsCallback}.

    Goal: a refactor that drops one of these from the critical list would
    silently turn a checkpoint failure into a quarantined no-op. Pin the
    full set.

    If you intentionally add/remove a class from this list, update this
    test AND document the change.
    """
    assert set(_DEFAULT_CRITICAL) == {
        "LineageRecorderCallback",
        "CheckpointCallback",
        "InvariantsCallback",
    }


def test_critical_set_can_be_overridden_via_constructor():
    """Pin: passing ``critical=("MyCustomCallback",)`` replaces the default
    list. The OLD default names are no longer critical.

    Setup: pass an empty critical tuple; LineageRecorderCallback should be
    swallowed (not raise out).
    """
    bus = EventBus([LineageRecorderCallback()], critical=())
    # Dispatch should NOT propagate — the bus treats the (now non-critical)
    # raiser as a normal failure (swallow + count).
    bus.dispatch("on_step_begin", step=1)


# ---------------------------------------------------------------------------
# on_error hook
# ---------------------------------------------------------------------------

def test_custom_on_error_invoked_with_event_callback_exception():
    """Custom ``on_error`` receives (event_name, callback_instance, exception).

    Setup: bus with a recording on_error hook + a callback that raises.
    Expected: one record with (event, cb_type, exc_type) matching.
    """
    recorder = _OnErrorRecorder()
    bus = EventBus([_RaisingTimes(raise_count=1)], on_error=recorder)
    bus.dispatch("on_step_begin", step=1)
    assert len(recorder.records) == 1
    event, cb_type, exc_type = recorder.records[0]
    assert event == "on_step_begin"
    assert cb_type == "_RaisingTimes"
    assert exc_type == "RuntimeError"


# ---------------------------------------------------------------------------
# _coerce input handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, Signal.CONTINUE),
        (Signal.SKIP_STEP, Signal.SKIP_STEP),
        (0, Signal.CONTINUE),
        (1, Signal.SKIP_STEP),
        (2, Signal.RETRY_STEP),
        (3, Signal.STOP_TRAINING),
        (99, Signal.CONTINUE),         # invalid int → CONTINUE
        ("skip_step", Signal.SKIP_STEP),
        ("SKIP_STEP", Signal.SKIP_STEP),
        ("retry_step", Signal.RETRY_STEP),
        ("stop_training", Signal.STOP_TRAINING),
        ("nonsense", Signal.CONTINUE),  # unknown string → CONTINUE
        (object(), Signal.CONTINUE),    # arbitrary object → CONTINUE
    ],
)
def test_invariant_coerce_handles_all_input_types(value, expected):
    """Invariant: ``_coerce`` maps all observed input types into a valid Signal.

    Goal: pin defensive coercion — unexpected inputs degrade to CONTINUE
    rather than raising or returning None.
    """
    assert _coerce(value) == expected


# ---------------------------------------------------------------------------
# callbacks property + bus internals
# ---------------------------------------------------------------------------

def test_invariant_callbacks_property_returns_copy_not_internal_list():
    """Invariant: ``bus.callbacks`` returns a copy. Mutating the returned
    list does NOT mutate the bus's internal list.

    Goal: pin defensive isolation — a user adding to the returned list
    must not silently insert callbacks into the dispatch chain.
    """
    cb = _Counter()
    bus = EventBus([cb])
    snap = bus.callbacks
    snap.append("intruder")
    assert "intruder" not in bus._callbacks


def test_max_consecutive_failures_clamped_to_at_least_one():
    """Constructor clamps ``max_consecutive_failures`` to >= 1 (line 80).

    Setup: pass 0; pass -5.
    Expected: ``bus._max_failures`` is 1 in both cases.
    """
    bus = EventBus([], max_consecutive_failures=0)
    assert bus._max_failures == 1
    bus2 = EventBus([], max_consecutive_failures=-5)
    assert bus2._max_failures == 1


def test_dispatch_returns_continue_when_no_callbacks_implement_event():
    """If no callback implements the event, dispatch returns CONTINUE
    (no signal escalation).
    """
    bus = EventBus([_Counter()])  # only implements on_step_begin
    assert bus.dispatch("on_train_end") == Signal.CONTINUE


def test_dispatch_calls_only_callbacks_implementing_method_others_untouched():
    """Invariant: a dispatch invokes every callback that implements the event
    and skips the rest — a callback lacking the method is not called, and the
    others still fire exactly once.

    Setup: two _Counter callbacks (only implement on_step_begin); dispatch
    on_step_begin then on_step_end.
    Expected: both fire once on on_step_begin; neither fires on on_step_end.
    """
    a, b = _Counter(), _Counter()
    bus = EventBus([a, b])
    bus.dispatch("on_step_begin", step=1)
    bus.dispatch("on_step_end", step=1)  # neither implements it
    assert (a.calls, b.calls) == (1, 1)


def test_dispatch_coerces_int_and_string_callback_return_values():
    """Invariant: dispatch coerces each callback's raw return value through
    ``_coerce`` before aggregating — an int (2) and a string ("skip_step")
    are mapped to Signals and the strongest wins.

    Setup: one callback returns int 2 (RETRY_STEP), another returns the
    string "skip_step".
    Expected: aggregated dispatch result is Signal.RETRY_STEP.
    """
    a = _Counter(signal=2)  # RETRY_STEP
    b = _Counter(signal="skip_step")
    out = EventBus([a, b]).dispatch("on_step_begin", step=1)
    assert out == Signal.RETRY_STEP


def test_add_appends_callback_to_internal_list():
    """``bus.add(cb)`` registers a new callback that participates in
    subsequent dispatches.
    """
    cb = _Counter()
    bus = EventBus([])
    bus.add(cb)
    bus.dispatch("on_step_begin", step=1)
    assert cb.calls == 1


def test_quarantined_property_returns_empty_initially():
    """Before any failures, ``bus.quarantined`` is empty."""
    bus = EventBus([_Counter()])
    assert bus.quarantined == []


# ---------------------------------------------------------------------------
# Concurrency contract — main-thread-only pin
# ---------------------------------------------------------------------------

def test_pin_event_bus_dispatch_not_thread_safe_main_thread_only():
    """Pin: ``EventBus`` is main-thread-only by design (PyTorch Lightning /
    HuggingFace Trainer convention — these frameworks also do not promise
    thread-safety on their dispatch paths).

    Observable signal:
      (a) the source of ``EventBus.dispatch`` contains no
          ``threading.Lock``/``RLock``/``with self._lock`` token, AND
      (b) the EventBus class docstring contains the literal substring
          ``"Thread-safety: NOT thread-safe"``.

    Rationale: forcing thread-safety would add a mutex to a hot path,
    deadlock when a callback dispatches sub-events, and maintain a
    "thread-safe but untested" contract that nobody verifies.

    If multi-thread support is added (with proper locking), update this
    test AND document the new concurrency contract in the module
    docstring.
    """
    # Signal (a): no Lock token in EventBus methods.
    src = inspect.getsource(EventBus)
    forbidden_tokens = ("threading.Lock", "threading.RLock", "self._lock", "RLock(", "Lock(")
    found = [tok for tok in forbidden_tokens if tok in src]
    assert not found, (
        f"EventBus source unexpectedly contains lock token(s) {found}; "
        "either the concurrency contract changed (and this pin should be "
        "updated) or the lock is incomplete and the docstring should be "
        "updated too."
    )

    # Signal (b): explicit docstring marker.
    doc = (EventBus.__doc__ or "")
    assert "Thread-safety: NOT thread-safe" in doc, (
        "EventBus class docstring must include the explicit "
        "'Thread-safety: NOT thread-safe' contract marker. "
        "If you intentionally make EventBus thread-safe, update both "
        "this test and the docstring to reflect the new contract."
    )
