"""Tests for lighttrain.utils.log.warn_once (D1 hot-loop dedup helper)."""

from __future__ import annotations

import logging

from lighttrain.utils.log import warn_once


def test_warn_once_emits_only_first_time_per_key(caplog) -> None:
    seen: set[str] = set()
    logger = logging.getLogger("test.warn_once")
    with caplog.at_level(logging.WARNING, logger="test.warn_once"):
        for _ in range(5):
            warn_once(seen, "k1", logger, "boom %d", 1)
    # Flooded 5×, logged once.
    assert sum("boom" in r.message for r in caplog.records) == 1


def test_warn_once_distinct_keys_each_emit_once(caplog) -> None:
    seen: set[str] = set()
    logger = logging.getLogger("test.warn_once2")
    with caplog.at_level(logging.WARNING, logger="test.warn_once2"):
        for _ in range(3):
            warn_once(seen, "a", logger, "msg-a")
            warn_once(seen, "b", logger, "msg-b")
    assert sum(r.message == "msg-a" for r in caplog.records) == 1
    assert sum(r.message == "msg-b" for r in caplog.records) == 1
    assert seen == {"a", "b"}


def test_safe_metrics_warns_once_per_bad_metric(caplog) -> None:
    """_safe_metrics runs per step; a non-coercible metric warns once per name."""
    import lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder as lr

    lr._WARNED_METRICS.clear()

    class _Bad:
        def __float__(self):
            raise ValueError("nope")

        def __str__(self):
            return "bad"

    with caplog.at_level(logging.WARNING):
        for _ in range(4):  # simulate 4 steps
            out = lr._safe_metrics({"loss": 1.0, "weird": _Bad()})
    assert out == {"loss": 1.0, "weird": "bad"}
    assert sum("not float-coercible" in r.message for r in caplog.records) == 1
    lr._WARNED_METRICS.clear()
