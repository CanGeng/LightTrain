"""Experiment-management commands: sweep / compare / fork."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from lighttrain.cli._context import console
from lighttrain.config import ConfigError, load_config


def sweep_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    sweep: Path = typer.Option(..., "-s", "--sweep"),
    strategy: str = typer.Option("grid", "--strategy"),
    top_k: int = typer.Option(5, "--top-k", help="Show top-K trials in report."),
    report_out: Path | None = typer.Option(
        None, "--report-out", help="Path for sweep_report.md (default: auto)."
    ),
) -> None:
    """Hyperparameter sweep — grid / random / optuna.

    Run all trials defined in the sweep spec YAML and write a Markdown report.

    \\b
    Example:
      lighttrain sweep -c recipes/sweep_demo.yaml -s recipes/sweep_r15.yaml
    """
    from lighttrain.lab.auto_report import write_sweep_report
    from lighttrain.lab.sweep import SweepRunner

    if not config.exists():
        console.print(f"[red]config not found:[/] {config}")
        raise typer.Exit(code=1)
    if not sweep.exists():
        console.print(f"[red]sweep spec not found:[/] {sweep}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]sweep[/] strategy={strategy}  spec={sweep.name}")
    try:
        runner = SweepRunner(config, sweep, strategy=strategy)
        report = runner.run()
    except Exception as exc:
        # escape so bracketed hints (e.g. ``pip install -e '.[sweep]'``) aren't
        # eaten by Rich markup.
        from rich.markup import escape

        console.print(f"[red]sweep failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    # Summary table
    table = Table(title=f"Sweep: {report.sweep_name}", show_lines=False)
    table.add_column("Trial", justify="right")
    table.add_column("Status")
    table.add_column("Metric", justify="right")
    for t in report.trials:
        metric_str = f"{t.metric:.6g}" if t.metric is not None else "—"
        table.add_row(str(t.trial_id), t.status, metric_str)
    console.print(table)

    if report.best_metric is not None:
        console.print(f"\n[green]best metric:[/] {report.best_metric:.6g}")
        console.print(f"[green]best config:[/] {report.best_config}")

    path = write_sweep_report(report, report_out, top_k=top_k)
    console.print(f"\n[green]report written →[/] {path}")


def compare_cmd(
    runs: list[str] = typer.Argument(..., help="Run directories to compare."),
    png: Path | None = typer.Option(None, "--png", help="Also write a PNG chart."),
    metric: list[str] | None = typer.Option(
        None, "--metric", help="Restrict the table to these metric(s); repeatable."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the comparison to a file: Markdown sweep table (.md) or "
        "per-run records (.json), by extension.",
    ),
) -> None:
    """Diff multiple runs — config changes + metric alignment.

    \\b
    Examples:
      lighttrain compare runs/exp/run_001 runs/exp/run_002
      lighttrain compare runs/exp/run_* --metric loss --output table.md
    """
    from lighttrain.lab.compare import (
        compare,
        render_ascii,
        render_markdown,
        render_png,
        to_records,
    )

    run_paths = [Path(r) for r in runs]
    missing = [p for p in run_paths if not p.exists()]
    if missing:
        console.print(f"[red]run dirs not found:[/] {missing}")
        raise typer.Exit(code=1)

    try:
        report = compare(run_paths)
    except Exception as exc:
        console.print(f"[red]compare failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    metrics = list(metric) if metric else None
    if metrics:
        unknown = [m for m in metrics if m not in report.metrics_table]
        if unknown:
            console.print(f"[yellow]no such metric in runs:[/] {unknown}")

    # --metric / --output switch to the sweep-style Markdown table.
    if metrics or output is not None:
        console.print(render_markdown(report, metrics))
    else:
        console.print(render_ascii(report))

    if output is not None:
        if output.suffix.lower() == ".json":
            import json

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(to_records(report, metrics), indent=2), encoding="utf-8")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report, metrics) + "\n", encoding="utf-8")
        console.print(f"[green]written →[/] {output}")

    if png is not None:
        try:
            render_png(report, png)
            console.print(f"[green]PNG written →[/] {png}")
        except RuntimeError as exc:
            console.print(f"[yellow]PNG skipped:[/] {exc}")


def fork_cmd(
    from_: Path = typer.Option(..., "--from", help="Checkpoint directory to fork from."),
    config: Path = typer.Option(..., "-c", "--config"),
    symlink: bool = typer.Option(False, "--symlink", help="Symlink instead of copying."),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Fork a checkpoint into a new run with an updated config.

    \\b
    Example:
      lighttrain fork --from runs/exp/run_001/checkpoints/step_500 \\
                      -c recipes/pretrain_causal.yaml ++optim.lr=1e-4
    """
    from lighttrain.lab.fork import fork

    if not from_.exists():
        console.print(f"[red]checkpoint not found:[/] {from_}")
        raise typer.Exit(code=1)
    if not config.exists():
        console.print(f"[red]config not found:[/] {config}")
        raise typer.Exit(code=1)

    try:
        cfg = load_config(config, overrides=list(overrides or []))
    except (ConfigError, Exception) as exc:
        console.print(f"[red]config error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        report = fork(from_, cfg, symlink=symlink)  # type: ignore[arg-type]
    except Exception as exc:
        console.print(f"[red]fork failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]forked →[/] {report.new_run_dir}")
    if report.lineage_edge_recorded:
        console.print("[green]lineage fork_of edge recorded[/]")
    else:
        console.print("[yellow]lineage not recorded (no lineage.sqlite in parent)[/]")
    console.print(
        f"\nResume with:\n  lighttrain resume --run {report.new_run_dir}"
    )
