"""EventBus critical-callback + quarantine semantics (REVIEW #5, DESIGN §18.5)."""

from __future__ import annotations

import pytest

from lighttrain.callbacks.base import EventBus, Signal


class _NormalThrower:
    """Non-critical callback that raises every time."""
    def on_step_end(self, **_):
        raise RuntimeError("boom")


class _CriticalThrower:
    """Critical via attribute."""
    critical = True
    def on_train_start(self, **_):
        raise RuntimeError("lineage broken")


class _CritByClassName:
    """Critical via class-name list (default DESIGN §18.5 entries)."""
    def on_train_start(self, **_):
        raise RuntimeError("via name")


class _Quiet:
    def __init__(self):
        self.seen = 0
    def on_step_end(self, **_):
        self.seen += 1
        return Signal.CONTINUE


def test_critical_callback_propagates_first_exception():
    cb = _CriticalThrower()
    bus = EventBus([cb])
    with pytest.raises(RuntimeError, match="lineage broken"):
        bus.dispatch("on_train_start")


def test_critical_by_class_name_default_list():
    # Use the documented default critical names by passing one through under
    # a class name that matches.
    cb = _CritByClassName()
    bus = EventBus([cb], critical=["_CritByClassName"])
    with pytest.raises(RuntimeError, match="via name"):
        bus.dispatch("on_train_start")


def test_noncritical_failure_counts_and_quarantines():
    thrower = _NormalThrower()
    quiet = _Quiet()
    bus = EventBus([thrower, quiet], max_consecutive_failures=3)

    # 1st failure — not yet quarantined.
    bus.dispatch("on_step_end")
    assert thrower.__class__.__name__ not in bus.quarantined
    # 2nd failure.
    bus.dispatch("on_step_end")
    assert thrower.__class__.__name__ not in bus.quarantined
    # 3rd failure → quarantine triggers.
    bus.dispatch("on_step_end")
    assert "_NormalThrower" in bus.quarantined

    # Other callbacks keep firing throughout (3 quiet calls).
    assert quiet.seen == 3

    # After quarantine the thrower is silently skipped, quiet still fires.
    bus.dispatch("on_step_end")
    assert quiet.seen == 4
