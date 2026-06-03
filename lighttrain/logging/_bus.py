"""LoggerBus — fans scalar/text records out to N registered backends.

Each backend implements ``LoggerProtocol`` from ``lighttrain.protocols``.
Per-backend exceptions are caught and routed to ``stderr`` so a single bad
sink can't crash a 12-hour training run.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Mapping
from typing import Any


class LoggerBus:
    """Aggregator over a list of backends. Each call fans out, isolating errors."""

    def __init__(self, backends: list[Any] | None = None) -> None:
        self._backends: list[Any] = list(backends or [])

    def add(self, backend: Any) -> None:
        self._backends.append(backend)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._backends)

    @property
    def backends(self) -> list[Any]:
        return list(self._backends)

    def _safe(self, fn_name: str, *args: Any, **kwargs: Any) -> None:
        for b in self._backends:
            fn = getattr(b, fn_name, None)
            if fn is None:
                continue
            try:
                fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 — log isolation is the point
                print(
                    f"[lighttrain.logging] backend {type(b).__name__}."
                    f"{fn_name} raised; continuing.",
                    file=sys.stderr,
                )
                traceback.print_exc()

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        self._safe("log_scalars", dict(scalars), step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_dict(
        self, d: Mapping[str, Any], step: int, *, prefix: str | None = None
    ) -> None:
        items = {f"{prefix}/{k}" if prefix else k: float(v) for k, v in d.items()}
        if items:
            self.log_scalars(items, step)

    def log_text(self, text: str, step: int) -> None:
        self._safe("log_text", text, step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self._safe("log_artifact", path, name)

    def flush(self) -> None:
        self._safe("flush")

    def close(self) -> None:
        self._safe("close")


__all__ = ["LoggerBus"]
