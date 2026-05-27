"""Banner renderer for PrepGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from rich.console import Console
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover — rich is in our base deps
    _HAS_RICH = False


@dataclass
class PlanEntry:
    name: str
    kind: str
    fingerprint: str  # short form (first 16 chars)
    full_fp: str
    hit: bool
    reason: str  # cache_hit | config_changed | code_version_changed | upstream_changed | schema_version_bumped | first_run
    eta_s: float | None = None
    rows: int | None = None


def format_plan(plan: Iterable[PlanEntry]) -> str:
    """Plain-text fallback / diagnostic dump."""
    plan = list(plan)
    n_total = len(plan)
    n_cached = sum(1 for p in plan if p.hit)
    n_run = n_total - n_cached
    lines = [
        f"PrepGraph: {n_total} nodes, {n_cached} cached, {n_run} to run",
        "─" * 65,
    ]
    for p in plan:
        tag = "[CACHE]" if p.hit else "[ RUN ]"
        eta = f"ETA: {p.eta_s:.0f}s" if p.eta_s is not None else "ETA: ?"
        reason = "(hit)" if p.hit else f"reason: {p.reason}"
        lines.append(
            f"{tag}  {p.kind:<10} {p.name:<24} fp={p.fingerprint}…  {eta:<12} {reason}"
        )
    lines.append("─" * 65)
    return "\n".join(lines)


def print_plan(console: "Console | None", plan: Iterable[PlanEntry]) -> None:
    plan = list(plan)
    if not _HAS_RICH or console is None:
        print(format_plan(plan))
        return
    n_total = len(plan)
    n_cached = sum(1 for p in plan if p.hit)
    n_run = n_total - n_cached
    table = Table(
        title=f"PrepGraph: {n_total} nodes, {n_cached} cached, {n_run} to run",
        show_lines=False,
    )
    table.add_column("status", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    table.add_column("name")
    table.add_column("fp", no_wrap=True)
    table.add_column("eta", justify="right")
    table.add_column("reason")
    for p in plan:
        tag = "[green][CACHE][/]" if p.hit else "[yellow][ RUN ][/]"
        eta = f"{p.eta_s:.0f}s" if p.eta_s is not None else "?"
        reason = "(hit)" if p.hit else p.reason
        table.add_row(tag, p.kind, p.name, p.fingerprint, eta, reason)
    console.print(table)


__all__ = ["PlanEntry", "format_plan", "print_plan"]
