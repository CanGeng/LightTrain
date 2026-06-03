"""EventBus + Signal aggregator.

Callbacks are loosely typed: the bus uses ``getattr(cb, event, None)`` so a
callback only implements the hooks it cares about. Per-callback exceptions
are caught and routed to a sink so a single bad callback can't kill a run.
Returned signals are aggregated with strict precedence:

    STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE

Critical callbacks: a callback marked as critical — either by setting
``critical = True`` on the instance, or by appearing in the EventBus's
``critical`` class-name list — bypasses the swallow-and-continue policy:
the first exception it raises propagates out of ``dispatch()`` so the
trainer can take it down hard (silently masking a lineage or checkpoint
failure would leave the user without diagnostics).

Non-critical callbacks are isolated: they may raise up to
``max_consecutive_failures`` times before they get **quarantined** and skipped
for the remainder of the run.
"""

from __future__ import annotations

import enum
import sys
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from ..protocols import CALLBACK_EVENTS


class Signal(enum.IntEnum):
    """Callback return signal."""

    CONTINUE = 0
    SKIP_STEP = 1
    RETRY_STEP = 2
    STOP_TRAINING = 3


def _coerce(value: Any) -> Signal:
    if value is None:
        return Signal.CONTINUE
    if isinstance(value, Signal):
        return value
    if isinstance(value, int):
        try:
            return Signal(int(value))
        except ValueError:
            return Signal.CONTINUE
    if isinstance(value, str):
        try:
            return Signal[value.upper()]
        except KeyError:
            return Signal.CONTINUE
    return Signal.CONTINUE


_DEFAULT_CRITICAL = ("LineageRecorderCallback", "CheckpointCallback", "InvariantsCallback")


class EventBus:
    """Dispatches lifecycle events to a list of callbacks.

    Thread-safety: NOT thread-safe. Call only from the main training thread.
    """

    EVENTS: tuple[str, ...] = CALLBACK_EVENTS

    def __init__(
        self,
        callbacks: Iterable[Any] | None = None,
        *,
        on_error: Callable[[str, Any, BaseException], None] | None = None,
        critical: Iterable[str] | None = None,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._callbacks: list[Any] = list(callbacks or [])
        self._on_error = on_error or self._default_on_error
        self._critical: tuple[str, ...] = tuple(
            critical if critical is not None else _DEFAULT_CRITICAL
        )
        self._max_failures = max(1, int(max_consecutive_failures))
        # Per-callback bookkeeping. Keyed by id(cb) so we don't require
        # callbacks to be hashable.
        self._failure_counts: dict[int, int] = {}
        self._quarantined: set[int] = set()

    def add(self, callback: Any) -> None:
        self._callbacks.append(callback)

    @property
    def callbacks(self) -> list[Any]:
        return list(self._callbacks)

    @property
    def quarantined(self) -> list[str]:
        """Class names of callbacks that have been quarantined."""
        out: list[str] = []
        for cb in self._callbacks:
            if id(cb) in self._quarantined:
                out.append(type(cb).__name__)
        return out

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._callbacks)

    @staticmethod
    def _default_on_error(event: str, cb: Any, exc: BaseException) -> None:
        print(
            f"[lighttrain.callbacks] {type(cb).__name__}.{event} raised "
            f"{type(exc).__name__}; continuing.",
            file=sys.stderr,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    def _is_critical(self, cb: Any) -> bool:
        if getattr(cb, "critical", False) is True:
            return True
        return type(cb).__name__ in self._critical

    def dispatch(self, event: str, **kwargs: Any) -> Signal:
        """Invoke ``event`` on every callback that has it.

        Returns the *strongest* signal returned by any callback (CONTINUE
        if none returned anything actionable). Unknown events are tolerated
        — they simply have no listeners.

        Failure handling:

        * Critical callbacks: first exception re-raises out.
        * Non-critical callbacks: exception → swallow + count;
          after ``max_consecutive_failures`` errors the callback is
          quarantined and skipped on all subsequent events.
        """
        result = Signal.CONTINUE
        for cb in self._callbacks:
            if id(cb) in self._quarantined:
                continue
            fn = getattr(cb, event, None)
            if fn is None:
                continue
            try:
                signal = _coerce(fn(**kwargs))
            except BaseException as exc:  # noqa: BLE001
                if self._is_critical(cb):
                    self._on_error(event, cb, exc)
                    raise
                self._on_error(event, cb, exc)
                self._failure_counts[id(cb)] = self._failure_counts.get(id(cb), 0) + 1
                if self._failure_counts[id(cb)] >= self._max_failures:
                    self._quarantined.add(id(cb))
                    print(
                        f"[lighttrain.callbacks] quarantining "
                        f"{type(cb).__name__} after {self._failure_counts[id(cb)]} "
                        f"failures.",
                        file=sys.stderr,
                    )
                continue
            # Successful invocation → reset its failure counter (consecutive,
            # not cumulative.
            if id(cb) in self._failure_counts:
                self._failure_counts.pop(id(cb), None)
            if signal > result:
                result = signal
        return result


__all__ = ["CALLBACK_EVENTS", "EventBus", "Signal"]
