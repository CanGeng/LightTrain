"""Tests for ``lighttrain.builtin_plugins.logging_backends.console.ConsoleLogger``.

Layered alongside ``tests/callbacks/logging/test_bus.py`` (LoggerBus fan-out)
and ``tests/callbacks/logging/test_jsonl_backend.py`` (JSONL records). This
module pins the ConsoleLogger throttling behavior.

Relocated from ``tests/test_logging_bus.py`` (the ConsoleLogger throttle test;
the LoggerBus and JSONL portions are already subsumed by the sibling mirrors).
"""

from __future__ import annotations

from lighttrain.builtin_plugins.logging_backends.console import ConsoleLogger


def test_invariant_console_backend_throttles_to_log_every(capsys):
    """Invariant: ``ConsoleLogger(log_every=N)`` prints only when ``step`` is a
    multiple of ``N`` and stays silent otherwise.

    Setup: log_every=10; emit at step 5 (silent) and step 10 (printed).
    Expected: exactly one ``step=`` line on stdout.
    """
    cl = ConsoleLogger(log_every=10)
    cl.log_scalars({"loss": 1.0}, step=5)  # not a multiple → silent
    cl.log_scalars({"loss": 1.0}, step=10)  # multiple of 10 → printed
    out = capsys.readouterr().out
    # Rich uses ANSI; check core substrings only.
    assert out.count("step=") == 1
