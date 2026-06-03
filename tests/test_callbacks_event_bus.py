"""EventBus dispatch + Signal aggregation + per-callback isolation."""

from __future__ import annotations

from lighttrain.callbacks.base import EventBus, Signal


class _Counter:
    def __init__(self, signal=None) -> None:
        self.calls = 0
        self.signal = signal

    def on_step_begin(self, **_):
        self.calls += 1
        return self.signal

    # Deliberately *no* on_step_end so getattr returns None.


class _Broken:
    def on_step_begin(self, **_):
        raise RuntimeError("kaboom")


def test_dispatch_only_calls_callbacks_with_method():
    a, b = _Counter(), _Counter()
    bus = EventBus([a, b])
    bus.dispatch("on_step_begin", step=1)
    bus.dispatch("on_step_end", step=1)  # neither implements it
    assert (a.calls, b.calls) == (1, 1)


def test_signal_aggregation_picks_strongest():
    a = _Counter(signal=Signal.SKIP_STEP)
    b = _Counter(signal=Signal.STOP_TRAINING)
    c = _Counter(signal=Signal.RETRY_STEP)
    bus = EventBus([a, b, c])
    out = bus.dispatch("on_step_begin", step=1)
    assert out == Signal.STOP_TRAINING


def test_int_and_string_returns_coerce():
    a = _Counter(signal=2)  # RETRY_STEP
    b = _Counter(signal="skip_step")
    out = EventBus([a, b]).dispatch("on_step_begin", step=1)
    assert out == Signal.RETRY_STEP


def test_callback_exception_does_not_kill_others(capsys):
    good = _Counter()
    bus = EventBus([_Broken(), good])
    out = bus.dispatch("on_step_begin", step=1)
    assert good.calls == 1
    assert out == Signal.CONTINUE
    err = capsys.readouterr().err
    assert "kaboom" in err


def test_unknown_event_is_tolerated():
    bus = EventBus([_Counter()])
    assert bus.dispatch("on_made_up_event", x=1) == Signal.CONTINUE
