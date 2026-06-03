"""PrepGraph commands: prep / prep-graph / prep-clean / prep-status / inspect-data.

(``inspect-data`` lives in ``diagnostics.py`` — see the assembler.)
"""

from __future__ import annotations

from pathlib import Path

import typer

from lighttrain.cli._context import console
from lighttrain.cli._helpers import _fmt_metric
from lighttrain.cli._runtime import build_prep_runner
from lighttrain.config import ConfigError


def prep_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    workers: int = typer.Option(1, "--workers"),
    pool: str = typer.Option(
        "thread", "--pool", help="In-layer pool: thread | process."
    ),
    only: str | None = typer.Option(None, "--only"),
    from_: str | None = typer.Option(None, "--from"),
) -> None:
    """Run the PrepGraph and print a cache-status banner.

    With ``--dry-run`` we resolve fingerprints + reasons but write nothing.
    ``--pool process`` enables true CPU parallelism for pickle-safe nodes.
    """
    _ = (only, from_)  # filtering knobs land later
    try:
        bundle = build_prep_runner(
            config, workers=int(workers), console=console, pool_kind=pool
        )
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]prep error:[/] {e}")
        raise typer.Exit(code=1) from e
    runner = bundle["runner"]
    plan = runner.plan()
    runner.print_banner(plan)
    if dry_run:
        return
    runner.run()
    console.print("[green]prep complete[/]")


def prep_graph_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Render the PrepGraph as DOT (or print to stdout)."""
    try:
        bundle = build_prep_runner(config, console=console)
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]prep-graph error:[/] {e}")
        raise typer.Exit(code=1) from e
    graph = bundle["graph"]
    lines = ["digraph prepgraph {", "  rankdir=LR;"]
    for name, node in graph.nodes.items():
        terminal = " style=bold" if name in graph.terminals else ""
        lines.append(f'  "{name}" [label="{name}\\n[{node.kind}]"{terminal}];')
    for name, node in graph.nodes.items():
        for u in node.inputs:
            lines.append(f'  "{u}" -> "{name}";')
    lines.append("}")
    dot = "\n".join(lines)
    if out is not None:
        out.write_text(dot, encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    else:
        console.print(dot)


def prep_clean_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    orphans: bool = typer.Option(False, "--orphans"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Remove unreferenced PrepGraph cache directories."""
    try:
        bundle = build_prep_runner(config, console=console)
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]prep-clean error:[/] {e}")
        raise typer.Exit(code=1) from e
    runner = bundle["runner"]
    if not orphans:
        console.print(
            "[yellow]prep-clean currently only supports --orphans[/]"
        )
        raise typer.Exit(code=2)
    removed = runner.cleanup_orphans(dry_run=dry_run)
    if not removed:
        console.print("[green]nothing to clean[/]")
        return
    for p in removed:
        prefix = "[yellow]would remove[/]" if dry_run else "[red]removed[/]"
        console.print(f"{prefix} {p}")


def prep_status_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    extras: bool = typer.Option(
        False, "--extras", help="Also print each node's persisted extras metrics."
    ),
) -> None:
    """Show PrepGraph cache status without executing anything."""
    try:
        bundle = build_prep_runner(config, console=console)
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]prep-status error:[/] {e}")
        raise typer.Exit(code=1) from e
    runner = bundle["runner"]
    runner.print_banner()
    if extras:
        node_extras = runner.node_extras()
        if not node_extras:
            console.print(
                "[yellow]no extras on disk — run `prep` first to materialize manifests[/]"
            )
            return
        console.print("[bold]extras[/]")
        for name, metrics in node_extras.items():
            if not metrics:
                continue
            rendered = "  ".join(
                f"{k}={_fmt_metric(v)}" for k, v in sorted(metrics.items())
            )
            console.print(f"  [cyan]{name}[/]: {rendered}")
