"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Any


def warn_once(
    seen: set[str],
    key: str,
    logger: logging.Logger,
    msg: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Log ``msg`` at WARNING level only the first time ``key`` is seen.

    ``seen`` is a caller-owned set (an instance attribute or a module global)
    recording which keys have already been warned, so a warning emitted from a
    hot loop (per step / per layer / per metric) fires once instead of flooding
    the logs. Mirrors the ``_warn_once`` pattern in ``FrozenStepCallback``.
    """
    if key in seen:
        return
    seen.add(key)
    logger.warning(msg, *args, **kwargs)


__all__ = ["warn_once"]
