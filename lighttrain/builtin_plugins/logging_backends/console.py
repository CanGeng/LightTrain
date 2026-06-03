"""Rich-based console logger."""

from __future__ import annotations

from typing import Any, Mapping

from rich.console import Console

from lighttrain.registry import register


@register("logger", "console")
class ConsoleLogger:
    """Single-line throttled console logger.

    Prints ``step=NN | tag=v.v | ...`` every ``log_every`` step, and any
    received text immediately. Cheap, dependency-light fallback.
    """

    def __init__(self, log_every: int = 1, console: Console | None = None) -> None:
        self.log_every = max(1, int(log_every))
        self.console = console or Console()

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        if step % self.log_every != 0:
            return
        parts = " | ".join(
            f"{k}={_fmt(v)}" for k, v in scalars.items() if v is not None
        )
        self.console.print(f"[cyan]step={step:>6}[/] | {parts}")

    def log_text(self, text: str, step: int) -> None:
        self.console.print(f"[dim]step={step}[/] {text}")

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self.console.print(f"[green]artifact[/] {name or ''} -> {path}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _fmt(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e4 or (0 < abs(f) < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4f}"


__all__ = ["ConsoleLogger"]
