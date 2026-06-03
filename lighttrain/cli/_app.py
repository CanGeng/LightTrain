"""Typer CLI entry point.

Run ``lighttrain --help`` to see the full command map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .. import __version__
from ..config import ConfigError, dump_resolved, load_config
from ..utils.env import load_dotenv_if_present
from ._runtime import (
    _validate_mode_override,
    build_prep_runner,
    setup_run_from_config,
)

console = Console()

app = typer.Typer(
    name="lighttrain",
    help="Single-GPU PyTorch LM training framework for research labs.",
    no_args_is_help=True,
    add_completion=False,
)

lineage_app = typer.Typer(name="lineage", help="Lineage operations.", no_args_is_help=True)
app.add_typer(lineage_app)

migrate_app = typer.Typer(
    name="migrate", help="Schema migration operations.", no_args_is_help=True
)
app.add_typer(migrate_app)


def _todo(milestone: str, what: str = "") -> None:
    """Emit a friendly not-yet-implemented message and exit non-zero."""
    msg = f"[yellow]not yet implemented ({milestone})[/]"
    if what:
        msg = f"{msg} — {what}"
    console.print(msg)
    raise typer.Exit(code=2)


def _flatten_patch_to_overrides(patch: object, prefix: str = "") -> list[str]:
    """Turn a nested dict from ``--apply-degrade patch.yaml`` into
    ``++a.b.c=value`` OmegaConf overrides.

    Strings, ints, floats, bools, and None map to the literal yaml repr.
    Lists are passed through ``yaml.safe_dump`` so OmegaConf parses them
    as sequences.
    """
    out: list[str] = []
    if not isinstance(patch, dict):
        return out
    for k, v in patch.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_patch_to_overrides(v, key))
        elif isinstance(v, (list, tuple)):
            try:
                import yaml as _yaml

                # Flow style is required: _parse_override_value only dispatches
                # to YAML for inputs starting with ``[ { ' "``, so block-style
                # sequences would otherwise be stored as a multi-line string.
                out.append(
                    f"++{key}={_yaml.safe_dump(list(v), default_flow_style=True).strip()}"
                )
            except Exception:  # noqa: BLE001
                out.append(f"++{key}={v!r}")
        elif v is None:
            out.append(f"++{key}=null")
        else:
            out.append(f"++{key}={v}")
    return out


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"lighttrain {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool | None = typer.Option(  # noqa: UP007
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-essential output."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output."),
) -> None:
    """Global options. Per-command options live on each subcommand."""
    _ = (quiet, verbose)
    # Load HF_TOKEN / HF_ENDPOINT from project-local .env, if present.
    load_dotenv_if_present()


# ---------------------------------------------------------------------------
# Train / eval / prep / produce-artifact
# ---------------------------------------------------------------------------


def _final_loss_from_run(run_dir: Path) -> float | None:
    """Return the last logged ``loss`` from ``<run_dir>/logs/metrics.jsonl``."""
    import json

    metrics = Path(run_dir) / "logs" / "metrics.jsonl"
    if not metrics.exists():
        return None
    last: float | None = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in row:
            last = float(row["loss"])
    return last


def _eval_perplexity(trainer: object, max_batches: int) -> float | None:
    """Perplexity on the val loader, falling back to the train loader.

    Mirrors ``eval_cmd``'s perplexity path so ``train --eval`` and
    ``lighttrain eval`` agree. Returns ``None`` if no loader / eval fails.
    """
    from ..eval.metrics import perplexity

    model = getattr(trainer, "model", None)
    data_module = getattr(trainer, "data_module", None)
    if model is None or data_module is None:
        return None
    loader = None
    if hasattr(data_module, "val_loader"):
        loader = data_module.val_loader()
    if loader is None and hasattr(data_module, "train_loader"):
        loader = data_module.train_loader()
    if loader is None:
        return None
    mb = max_batches if max_batches > 0 else None
    try:
        return perplexity(model, loader, device=getattr(trainer, "device", None), max_batches=mb)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]perplexity eval failed:[/] {exc}")
        return None


def _append_run_summary(path: Path, row: dict) -> None:
    """Append ``row`` to a JSON-list summary at ``path``, replacing any existing
    entry with the same ``exp`` so a multi-variant shell loop accumulates one
    row per variant. Atomic (tmp + replace)."""
    import json
    import os

    rows: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [r for r in loaded if not (isinstance(r, dict) and r.get("exp") == row.get("exp"))]
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., "-c", "--config", help="Recipe YAML path."),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
    mode: str | None = typer.Option(None, "--mode", help="lab | prod"),
    print_config: bool = typer.Option(False, "--print-config", help="Print resolved config and exit."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable all caches."),
    apply_degrade: Path | None = typer.Option(
        None, "--apply-degrade", help="Apply OOM degrade patch."
    ),
    allow_stale_artifact: bool = typer.Option(
        False, "--allow-stale-artifact", help="Bypass artifact header check."
    ),
    eval: bool = typer.Option(
        False, "--eval", help="Run perplexity eval after training."
    ),
    eval_max_batches: int = typer.Option(
        0, "--eval-max-batches", help="Limit post-train eval to N batches (0 = no limit)."
    ),
    eval_json: Path | None = typer.Option(
        None, "--eval-json", help="Write post-train eval metrics to this JSON path."
    ),
    output_summary: Path | None = typer.Option(
        None,
        "--output-summary",
        help="Append a one-row run summary (exp, wall, final_loss, eval_ppl, "
        "checkpoint, status) to this JSON list; keyed by exp.",
    ),
) -> None:
    """Train a model from a recipe YAML."""
    _ = no_cache

    # ``--apply-degrade patch.yaml`` flattens the patch into
    # ``++key.path=value`` overrides appended *after* the user-supplied
    # ones so they win.
    overrides = list(overrides or [])
    if apply_degrade is not None:
        if not apply_degrade.exists():
            console.print(f"[red]patch not found:[/] {apply_degrade}")
            raise typer.Exit(code=1)
        try:
            import yaml as _yaml

            patch = _yaml.safe_load(apply_degrade.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]invalid patch yaml:[/] {e}")
            raise typer.Exit(code=1) from e
        overrides.extend(_flatten_patch_to_overrides(patch))
        console.print(
            f"[yellow]applying degrade patch[/] {apply_degrade} "
            f"({len(overrides)} overrides total)"
        )

    if print_config:
        try:
            # Pure config dump — don't trigger built-in registry population or
            # user_modules plugin imports (both are torch-heavy / side-effecting).
            cfg = load_config(
                config,
                overrides=overrides,
                import_user_modules=False,
                register_components=False,
            )
            if mode is not None:
                cfg.mode = _validate_mode_override(mode)  # type: ignore[union-attr]
        except (ConfigError, FileNotFoundError) as e:
            console.print(f"[red]config error:[/] {e}")
            raise typer.Exit(code=1) from e
        console.print(dump_resolved(cfg))
        return

    try:
        bundle = setup_run_from_config(
            config,
            overrides=overrides,
            mode=mode,
            allow_stale_artifact=allow_stale_artifact,
        )
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e

    run_dir: Path = bundle["run_dir"]
    trainer = bundle["trainer"]
    cfg = bundle.get("cfg")
    exp = getattr(cfg, "exp", None) or "default"
    console.print(f"[green]run_dir[/] = {run_dir}")

    import time as _time

    def _last_checkpoint() -> str | None:
        mgr = getattr(trainer, "ckpt_manager", None)
        if mgr is None:
            return None
        try:
            latest = mgr.latest()
            return str(latest) if latest is not None else None
        except Exception:  # noqa: BLE001
            return None

    t0 = _time.perf_counter()
    fit_error: BaseException | None = None
    try:
        try:
            trainer.fit()
        except BaseException as exc:  # noqa: BLE001 — capture to still emit a summary row
            fit_error = exc
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()
    wall_seconds = _time.perf_counter() - t0

    # Post-fit eval (only on a clean run).
    eval_ppl: float | None = None
    if eval and fit_error is None:
        eval_ppl = _eval_perplexity(trainer, eval_max_batches)
        if eval_ppl is not None:
            console.print(f"[green]eval perplexity[/] = {eval_ppl:.6g}")
        if eval_json is not None:
            import json
            import time

            eval_json.parent.mkdir(parents=True, exist_ok=True)
            eval_json.write_text(
                json.dumps(
                    {"task_name": "train_eval", "metrics": {"perplexity": eval_ppl},
                     "step": 0, "timestamp": time.time()},
                    indent=2,
                ),
                encoding="utf-8",
            )

    if output_summary is not None:
        _append_run_summary(
            output_summary,
            {
                "exp": exp,
                "run_dir": str(run_dir),
                "status": "error" if fit_error is not None else "ok",
                "wall_seconds": round(wall_seconds, 4),
                "final_loss": _final_loss_from_run(run_dir),
                "eval_ppl": eval_ppl,
                "last_checkpoint": _last_checkpoint(),
                "error": (f"{type(fit_error).__name__}: {fit_error}" if fit_error else None),
            },
        )
        console.print(f"[green]summary →[/] {output_summary}")

    if fit_error is not None:
        console.print(f"[red]training failed:[/] {type(fit_error).__name__}: {fit_error}")
        raise typer.Exit(code=1)
    console.print("[green]training complete[/]")



@app.command("prep")
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


@app.command("prep-graph")
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


@app.command("prep-clean")
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


@app.command("prep-status")
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


def _fmt_metric(v: Any) -> str:
    """Compact metric value rendering: round floats, pass through the rest."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


@app.command("produce-artifact")
def produce_artifact_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    estimate: bool = typer.Option(False, "--estimate"),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Run an ArtifactProducer offline.

    Reads ``cfg.artifacts.producer`` + ``cfg.artifacts.store`` from the recipe,
    iterates the configured train dataset (no DataLoader / collator), runs
    ``model.forward`` per sample under ``no_grad``, and writes the resulting
    tensors to the store.
    """
    from ._produce import run_produce  # local import to avoid pulling torch eagerly

    try:
        manifest = run_produce(
            config,
            overrides=list(overrides or []),
            estimate=estimate,
            console=console,
        )
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]produce-artifact error:[/] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]artifact finalized[/] -> {manifest}")


# ---------------------------------------------------------------------------
# Lab tools (sweep / compare / fork / replay / estimate)
# ---------------------------------------------------------------------------


@app.command("sweep")
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
    from ..lab.auto_report import write_sweep_report
    from ..lab.sweep import SweepRunner

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


@app.command("compare")
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
    from ..lab.compare import (
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


@app.command("fork")
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
    from ..lab.fork import fork

    if not from_.exists():
        console.print(f"[red]checkpoint not found:[/] {from_}")
        raise typer.Exit(code=1)
    if not config.exists():
        console.print(f"[red]config not found:[/] {config}")
        raise typer.Exit(code=1)

    try:
        cfg = load_config(config, list(overrides or []))
    except (ConfigError, Exception) as exc:
        console.print(f"[red]config error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        report = fork(from_, cfg, symlink=symlink)
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


@app.command("replay")
def replay_cmd(
    run: Path = typer.Option(..., "--run"),
    at: str | None = typer.Option(None, "--at"),
) -> None:
    """Replay the last crash bundle (or frozen step) of a run.

    Without ``--at`` we pick the most recent ``diagnostics/crash_*``; with
    ``--at step_<n>`` we look up a frozen step bundle at that step.
    """
    if not run.exists():
        console.print(f"[red]run dir not found:[/] {run}")
        raise typer.Exit(code=1)

    # Locate a target bundle.
    target: Path | None = None
    if at is not None and at.startswith("step_"):
        cands = sorted((run / "frozen_steps").glob(f"{at}_*.zip")) if (
            run / "frozen_steps"
        ).exists() else []
        if cands:
            target = cands[-1]
    if target is None:
        # Most recent crash bundle.
        diag = run / "diagnostics"
        crashes = sorted(diag.glob("crash_*"), reverse=True) if diag.exists() else []
        for c in crashes:
            batch = c / "batch.pt"
            state = c / "model_state.safetensors"
            spec = c / "model_spec.json"
            if batch.exists() and state.exists() and spec.exists():
                target = c
                break
    if target is None:
        # Fall back to the most recent frozen step.
        fs = sorted((run / "frozen_steps").glob("*.zip")) if (
            run / "frozen_steps"
        ).exists() else []
        if fs:
            target = fs[-1]

    if target is None:
        console.print(f"[red]no replayable bundle found under {run}[/]")
        raise typer.Exit(code=1)

    if target.suffix == ".zip":
        # frozen step path — same as replay-step.
        return replay_step_cmd(bundle=target, debugger=False, inject=None)

    # Crash bundle directory — rebuild model + load state + forward.
    import json as _json

    import torch as _torch

    from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss

    from ..minimal import build_minimal_model, load_state
    from ..protocols import LossContext

    try:
        import lighttrain.builtin_plugins.models.adapters  # noqa: F401 — populate registry
    except Exception:
        pass

    spec = _json.loads((target / "model_spec.json").read_text(encoding="utf-8"))
    model = build_minimal_model(spec)
    load_state(model, target / "model_state.safetensors", strict=False)
    batch = _torch.load(str(target / "batch.pt"), weights_only=True)
    model.train()
    out = model(**batch)
    loss_fn = CrossEntropyLoss()
    loss_dict = loss_fn(out, batch, LossContext(step=0, epoch=0))
    console.print(
        f"[green]replayed crash bundle[/] {target.relative_to(run)} "
        f":: loss={float(loss_dict['loss'].detach().item()):.4f}"
    )


@app.command("estimate")
def estimate_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
    json_out: Path | None = typer.Option(
        None, "--json", help="Also write the report as JSON to this path."
    ),
) -> None:
    """Pre-flight resource estimate.

    Builds the recipe's model, sums trainable / total params, bounds per-step
    memory (param + grad + optim state + activations), and prints a coarse
    tokens/s figure. For ``engine.name == layer_offload`` it additionally
    reports the per-layer "recompute vs transfer" breakdown so the user can
    pick ``resident_layers`` knowingly.
    """
    import json

    from ..lab.estimate import estimate, report_to_dict

    try:
        # load_config populates the registry (register_components default True);
        # estimate() also self-imports as a safety net for direct dict callers.
        cfg = load_config(config, overrides=list(overrides or []))
    except ConfigError as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e

    rpt = estimate(cfg)
    table = Table(title="lighttrain estimate", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value")

    def _fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} GB"

    table.add_row("model", rpt.model_name)
    table.add_row("optimizer", rpt.optimizer_name)
    table.add_row("engine", rpt.engine_name)
    table.add_row("trainable_params", f"{rpt.trainable_params:,}")
    table.add_row("all_params", f"{rpt.all_params:,}")
    table.add_row("trainable_ratio", f"{rpt.trainable_ratio * 100:.2f}%")
    table.add_row("param_bytes", _fmt_bytes(rpt.param_bytes))
    table.add_row("grad_bytes", _fmt_bytes(rpt.grad_bytes))
    table.add_row("optim_state_bytes", _fmt_bytes(rpt.optim_state_bytes))
    table.add_row("activation_bytes_per_step", _fmt_bytes(rpt.activation_bytes_per_step))
    table.add_row("total_bytes_per_step", _fmt_bytes(rpt.total_bytes_per_step))
    table.add_row("tokens_per_sec_estimate", f"{rpt.tokens_per_sec_estimate:.1f}")
    console.print(table)

    if rpt.offload is not None:
        off = rpt.offload
        off_table = Table(title="LayerOffload breakdown", show_header=False)
        off_table.add_column("metric", style="bold")
        off_table.add_column("value")
        off_table.add_row("layers", str(off.layers))
        off_table.add_row("resident_layers", str(off.resident_layers))
        off_table.add_row("layer_param_bytes", _fmt_bytes(off.layer_param_bytes))
        off_table.add_row(
            "recompute_us_per_layer", f"{off.recompute_us_per_layer:.1f}"
        )
        off_table.add_row(
            "transfer_us_per_layer", f"{off.transfer_us_per_layer:.1f}"
        )
        off_table.add_row("recommended_mode", off.recommended_mode)
        off_table.add_row("pcie_bandwidth_used", off.pcie_bandwidth_used)
        console.print(off_table)

    for note in rpt.notes:
        console.print(f"[dim]• {note}[/]")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report_to_dict(rpt), indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/] {json_out}")


# ---------------------------------------------------------------------------
# EvalSuite commands
# ---------------------------------------------------------------------------


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="Checkpoint directory to evaluate (overrides config)."
    ),
    json_out: Path | None = typer.Option(
        None, "--json", help="Write EvalReport to this JSON path."
    ),
    max_batches: int = typer.Option(
        0, "--max-batches", help="Limit perplexity eval to N batches (0 = no limit)."
    ),
) -> None:
    """Run EvalSuite on a checkpoint.

    Loads the recipe, optionally restores the given checkpoint, runs all
    configured eval tasks, and prints a rich.Table summary.  Pass ``--json``
    to also write the full EvalReport to disk.
    """
    import json
    import tempfile

    load_dotenv_if_present()

    # `eval` is read-only — don't litter run_root with an empty run dir per
    # invocation (Issue #6). Mint into a temp dir that we clean up afterwards.
    _tmp_run = tempfile.TemporaryDirectory(prefix="lighttrain-eval-")
    try:
        bundle = setup_run_from_config(
            config,
            overrides=list(overrides or []),
            existing_run_dir=Path(_tmp_run.name),
        )
        trainer = bundle["trainer"]
        cfg = bundle["cfg"]

        loaded_ckpt = False
        if checkpoint is not None:
            ckpt_manager = getattr(trainer, "ckpt_manager", None)
            if ckpt_manager is not None:
                try:
                    trainer.load_checkpoint(checkpoint)
                    console.print(f"[green]loaded checkpoint[/] {checkpoint}")
                    loaded_ckpt = True
                except Exception as exc:
                    console.print(f"[yellow]checkpoint load failed:[/] {exc}")

        # Loud guard: scoring init weights produces a meaningless (random)
        # perplexity. Make it impossible to mistake for a trained result.
        if not loaded_ckpt:
            console.print(
                "[bold yellow]⚠ evaluating UNTRAINED weights[/] — no checkpoint "
                "loaded (pass --checkpoint <dir>); metrics below reflect random "
                "initialization, not a trained model."
            )

        model = getattr(trainer, "model", None)
        if model is None:
            console.print("[red]trainer has no model[/]")
            raise typer.Exit(code=1)

        device = getattr(trainer, "device", None)

        # ---- perplexity via val loader, falling back to the train loader ----
        # The fallback is what lets `lighttrain eval` work on recipes without a
        # dedicated val split (e.g. the mamba3 reproduction), so the experiment's
        # direct-call bypass of this CLI is no longer needed (Issue #10).
        ppl_value = _eval_perplexity(trainer, max_batches)

        metrics: dict[str, float] = {}
        if ppl_value is not None:
            metrics["perplexity"] = ppl_value

        # ---- run configured evaluator if present ----
        evaluator = getattr(trainer, "evaluator", None) or getattr(cfg, "evaluator", None)
        if evaluator is not None:
            try:
                from ..eval.suite import Evaluator
                if isinstance(evaluator, Evaluator):
                    report = evaluator.run(model, step=0, device=device, force=True)
                    if report is not None:
                        metrics.update(report.metrics)
            except Exception as exc:
                console.print(f"[yellow]evaluator failed:[/] {exc}")

        # ---- display ----
        table = Table(title="lighttrain eval", show_header=True)
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.6g}")
        console.print(table)

        if json_out is not None:
            import time
            report_dict = {
                "task_name": "eval",
                "metrics": metrics,
                "step": 0,
                "timestamp": time.time(),
            }
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            console.print(f"[green]wrote[/] {json_out}")
    finally:
        try:
            _tmp_run.cleanup()
        except Exception:  # noqa: BLE001 — open handles (e.g. sqlite) on some OSes
            pass


@app.command("regression-gate")
def regression_gate_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    metric: str = typer.Option(..., "--metric", help="Metric name to check."),
    threshold: float = typer.Option(..., "--threshold", help="Gate threshold."),
    op: str = typer.Option("<", "--op", help="Comparison operator: <, <=, >, >=, ==, !="),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
    action: str = typer.Option("abort", "--action", help="abort | warn | skip"),
    max_batches: int = typer.Option(8, "--max-batches"),
) -> None:
    """Check a regression gate against an eval metric.

    Exits with code 0 if the gate passes, code 1 if it fails.  Designed for
    use in CI/CD pipelines and sweep early-stopping.

    Example::

        lighttrain regression-gate -c recipe.yaml --metric perplexity --threshold 50.0 --op "<"
    """

    from ..builtin_plugins.eval.regression_gate import RegressionGate
    from ..eval.metrics import perplexity
    from ..eval.suite import EvalReport

    load_dotenv_if_present()
    bundle = setup_run_from_config(config)
    trainer = bundle["trainer"]

    if checkpoint is not None:
        ckpt_manager = getattr(trainer, "ckpt_manager", None)
        if ckpt_manager is not None:
            try:
                trainer.load_checkpoint(checkpoint)
            except Exception as exc:
                console.print(f"[yellow]checkpoint load failed:[/] {exc}")

    model = getattr(trainer, "model", None)
    data_module = getattr(trainer, "data_module", None)
    device = getattr(trainer, "device", None)

    metrics: dict[str, float] = {}
    if data_module is not None:
        val_loader = data_module.val_loader() if hasattr(data_module, "val_loader") else None
        if val_loader is not None:
            try:
                mb = max_batches if max_batches > 0 else None
                metrics["perplexity"] = perplexity(model, val_loader, device=device, max_batches=mb)
            except Exception as exc:
                console.print(f"[yellow]eval failed:[/] {exc}")
                raise typer.Exit(code=1) from None

    report = EvalReport(task_name="regression_gate", metrics=metrics)
    gate = RegressionGate(
        metric_name=metric,
        threshold=threshold,
        op=op,
        action=action,
    )

    try:
        gate.check(report)
        val = metrics.get(metric, "N/A")
        console.print(
            f"[green]PASS[/] {metric} {op} {threshold}  (value={val})"
        )
    except Exception as exc:
        console.print(f"[red]FAIL[/] {exc}")
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# Failure-first
# ---------------------------------------------------------------------------


@app.command("freeze-step")
def freeze_step_cmd(
    run: Path = typer.Option(..., "--run"),
    step: int = typer.Option(..., "--step"),
) -> None:
    """Freeze a single step from a previous run into a replayable bundle.

    Loads the recipe stored under ``<run>/config.snapshot.yaml``, reuses the
    existing run dir, restores the nearest checkpoint, then runs one step
    with the FrozenStepCallback so the bundle lands under
    ``<run>/frozen_steps/step_<n>_cli.zip``.
    """
    if not run.exists():
        console.print(f"[red]run dir not found:[/] {run}")
        raise typer.Exit(code=1)
    cfg_path = run / "config.snapshot.yaml"
    if not cfg_path.exists():
        console.print(f"[red]no recipe at {cfg_path}[/]")
        raise typer.Exit(code=1)
    overrides = [
        "++trainer.max_steps=1",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
        "++trainer.log_every=1",
        "++diagnostics.frozen_step_every=1",
    ]
    try:
        bundle = setup_run_from_config(
            cfg_path,
            overrides=overrides,
            existing_run_dir=run,
        )
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e
    trainer = bundle["trainer"]
    # Best-effort: restore the closest checkpoint <= step. Skip if absent.
    ckpts = trainer.ckpt_manager.list_steps()
    target = None
    for p in ckpts:
        try:
            n = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        if n <= step and (target is None or n > int(target.name.split("_", 1)[1])):
            target = p
    if target is not None:
        try:
            trainer.load_checkpoint(target)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]warning:[/] could not load {target}: {e}")
    # Override step on the ctx so the produced zip is named for the user's step.
    trainer.ctx.step = int(step)
    # Switch reason to cli on any FrozenStepCallback instances.
    for cb in trainer.callbacks:
        if type(cb).__name__ == "FrozenStepCallback":
            cb.reason = "cli"
            cb.every = 1
    try:
        trainer.fit(steps=int(step) + 1)
    finally:
        if bundle.get("logger") is not None:
            try:
                bundle["logger"].close()
            except Exception:
                pass
    zips = sorted((run / "frozen_steps").glob("*.zip"))
    if zips:
        console.print(f"[green]frozen step bundle[/] -> {zips[-1]}")
    else:
        console.print("[yellow]no bundle produced (check callback wiring)[/]")


@app.command("replay-step")
def replay_step_cmd(
    bundle: Path = typer.Argument(...),
    debugger: bool = typer.Option(False, "--debugger"),
    inject: Path | None = typer.Option(None, "--inject"),
) -> None:
    """Replay a frozen step bundle (functional replay).

    Loads the model + batch + RNG from the zip, then runs forward+backward
    once and prints loss / grad_norm. ``--debugger`` drops into pdb before
    forward; ``--inject path.py`` exec's a snippet in a tiny namespace.
    """
    if not bundle.exists():
        console.print(f"[red]bundle not found:[/] {bundle}")
        raise typer.Exit(code=1)
    from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss

    from ..diagnostics.frozen_step import read_frozen_step_bundle, replay_step_bundle

    try:
        bdl = read_frozen_step_bundle(bundle)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]invalid bundle:[/] {e}")
        raise typer.Exit(code=1) from e
    # Default loss = CE; works for any tiny_lm / hf_causal-style model.
    try:
        result = replay_step_bundle(
            bdl,
            loss_fn=CrossEntropyLoss(),
            debugger=debugger,
            inject=inject,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]replay failed:[/] {type(e).__name__}: {e}")
        raise typer.Exit(code=2) from e
    table = Table(title=f"replay step {result['step']} ({result['reason']})")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="green")
    for k in ("step", "reason", "loss", "grad_norm", "logits_shape"):
        v = result.get(k)
        table.add_row(k, str(v))
    console.print(table)


@app.command("doctor")
def doctor_cmd(run: Path = typer.Option(..., "--run")) -> None:
    """Lineage / schema / checkpoint health check.

    Checks:

    * checkpoint inventory (latest + count)
    * lineage SQLite presence + node counts per kind
    * schema_version sanity against ``SCHEMA_VERSION`` registry
    * dangling edges (src/dst pointing at deleted node)
    * frozen step bundles, NaN repros, callback failure aggregation

    Exit code: 0 = healthy, 2 = problems detected.
    """
    if not run.exists() or not run.is_dir():
        console.print(f"[red]not a run dir:[/] {run}")
        raise typer.Exit(code=1)

    problems = 0
    ckpt_dir = run / "checkpoints"
    if ckpt_dir.exists():
        from ..checkpoint.manager import CheckpointManager

        mgr = CheckpointManager(run)
        steps = mgr.list_steps()
        latest = mgr.latest()
        best = mgr.best()
        latest_name = latest.name if latest else "—"
        best_name = best.name if best else "—"
        console.print(
            f"[green]✔ checkpoints[/]    n={len(steps)}  latest={latest_name}  best={best_name}"
        )
    else:
        console.print("[yellow]…  checkpoints[/]    N/A (no checkpoints/ dir)")

    lineage_path = run / "lineage.sqlite"
    if lineage_path.exists():
        from ..lineage.store import LineageStore
        from ..prepgraph._fp import SCHEMA_VERSION

        ls = LineageStore(lineage_path)
        try:
            counts: dict[str, int] = {}
            schema_misses: list[tuple[int, str, str | None, str]] = []
            for n in ls.iter_nodes():
                counts[n["kind"]] = counts.get(n["kind"], 0) + 1
                sk = n.get("schema_kind")
                sv = n.get("schema_version")
                if sk and sk in SCHEMA_VERSION and sv != SCHEMA_VERSION[sk]:
                    schema_misses.append((n["id"], sk, sv, SCHEMA_VERSION[sk]))
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "<empty>"
            console.print(f"[green]✔ lineage[/]        {summary}")

            if schema_misses:
                problems += 1
                console.print(
                    f"[red]✘ schemas[/]        {len(schema_misses)} nodes lag current "
                    f"SCHEMA_VERSION; run `lighttrain migrate ...`"
                )
                for nid, sk, sv, want in schema_misses[:5]:
                    console.print(
                        f"   - id={nid} kind={sk} schema_version={sv!r} (want {want!r})"
                    )
            else:
                console.print("[green]✔ schemas[/]        all current")

            # Dangling edges — scan the edges table directly so we also see
            # edges whose ``src`` node has been deleted (the previous
            # implementation only iterated edges_from(existing_nodes) and
            # missed half of the orphan cases).
            valid_ids = {n["id"] for n in ls.iter_nodes()}
            dangling: list[tuple[int, int, str]] = []
            for e in ls.iter_edges():
                if e["src"] not in valid_ids or e["dst"] not in valid_ids:
                    dangling.append((e["src"], e["dst"], e["kind"]))
            if dangling:
                problems += 1
                console.print(
                    f"[red]✘ lineage edges[/]  {len(dangling)} dangling edges; "
                    "run `lighttrain lineage prune-orphans`"
                )
            else:
                console.print("[green]✔ lineage edges[/]  no orphans")
        finally:
            ls.close()
    else:
        console.print(f"[yellow]…  lineage[/]        N/A (no {lineage_path.name})")

    # Diagnostics: frozen_step bundles + NaN repros + callback failures.
    frozen_dir = run / "frozen_steps"
    if frozen_dir.exists():
        zips = sorted(frozen_dir.glob("*.zip"))
        last = zips[-1].name if zips else "—"
        console.print(f"[green]✔ frozen_steps[/]   n={len(zips)}  last={last}")
    else:
        console.print("[yellow]…  frozen_steps[/]   N/A (no frozen_steps/ dir)")

    diag = run / "diagnostics"
    nan_repros = sorted(diag.glob("repro_nan_*")) if diag.exists() else []
    if nan_repros:
        problems += 1
        console.print(
            f"[red]✘ NaN repros[/]    {len(nan_repros)} repro kit(s) under diagnostics/repro_nan_*"
        )
    else:
        console.print("[green]✔ NaN repros[/]    none")

    crash = sorted(diag.glob("crash_*")) if diag.exists() else []
    if crash:
        problems += 1
        console.print(
            f"[red]✘ crash bundles[/] {len(crash)} crash(es); see {crash[-1].relative_to(run)}"
        )
    else:
        console.print("[green]✔ crash bundles[/] none")

    cb_log = diag / "callback_failures.jsonl"
    if cb_log.exists():
        try:
            n = sum(
                1
                for line in cb_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:
            n = 0
        if n > 0:
            console.print(
                f"[yellow]…  callback report[/] {n} isolated failure(s); "
                "see diagnostics/callback_report.md"
            )
        else:
            console.print("[green]✔ callback report[/] no failures")
    else:
        console.print("[green]✔ callback report[/] no failures")

    if problems:
        console.print(f"[red]{problems} issue(s) found[/]")
        raise typer.Exit(code=2)
    console.print("[green]ok[/]")


# ---------------------------------------------------------------------------
# Lineage subcommands
# ---------------------------------------------------------------------------


def _open_lineage(db: Path) -> Any:
    """Open a LineageStore at ``db`` — caller closes."""
    from ..lineage.store import LineageStore

    if not db.exists():
        console.print(f"[red]lineage db not found at {db}[/]")
        raise typer.Exit(code=1)
    return LineageStore(db)


def _resolve_node(store: Any, ref: str) -> int:
    nid = store.resolve_ref(ref)
    if nid is None:
        console.print(f"[red]no lineage node matches ref {ref!r}[/]")
        raise typer.Exit(code=1)
    return int(nid)


@lineage_app.command("tag")
def lineage_tag_cmd(
    node: str = typer.Argument(...),
    tag: str = typer.Option(..., "--tag"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.tag(nid, tag)
        console.print(f"[green]tagged[/] #{nid} += {tag!r}")
    finally:
        store.close()


@lineage_app.command("untag")
def lineage_untag_cmd(
    node: str = typer.Argument(...),
    tag: str = typer.Option(..., "--tag"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.untag(nid, tag)
        console.print(f"[green]untagged[/] #{nid} -= {tag!r}")
    finally:
        store.close()


@lineage_app.command("invalidate")
def lineage_invalidate_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.invalidate(nid)
        console.print(f"[yellow]invalidated[/] #{nid}")
    finally:
        store.close()


@lineage_app.command("pin")
def lineage_pin_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.pin(nid)
        console.print(f"[green]pinned[/] #{nid}")
    finally:
        store.close()


@lineage_app.command("gc")
def lineage_gc_cmd(
    db: Path = typer.Option(..., "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    keep_last: int = typer.Option(3, "--keep-last"),
    kind: str = typer.Option("artifact", "--kind", help="artifact|checkpoint|config|run"),
) -> None:
    from ..lineage.retention import RetentionPolicy, gc_artifacts

    store = _open_lineage(db)
    try:
        report = gc_artifacts(
            store,
            policy=RetentionPolicy(keep_last=keep_last, keep_tagged=True, keep_pinned=True),
            kind=kind,
            dry_run=dry_run,
            delete_paths=not dry_run,
        )
        console.print(
            f"[green]gc[/] deprecated={len(report.deprecated)} deleted={len(report.deleted)} "
            f"paths_deleted={len(report.paths_deleted)}"
        )
    finally:
        store.close()


@lineage_app.command("prune-orphans")
def lineage_prune_cmd(
    db: Path = typer.Option(..., "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from ..lineage.retention import prune_orphans

    store = _open_lineage(db)
    try:
        removed = prune_orphans(store, dry_run=dry_run)
        console.print(f"[green]pruned[/] {len(removed)} orphan node(s)")
    finally:
        store.close()


@lineage_app.command("graph")
def lineage_graph_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
    depth: int = typer.Option(5, "--depth"),
    out: Path | None = typer.Option(None, "--out", help="Write to file; ext=.dot or .mermaid."),
    fmt: str = typer.Option("mermaid", "--fmt", help="mermaid | dot"),
) -> None:
    from ..lineage.dag import to_dot, to_mermaid

    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        if fmt == "dot":
            text = to_dot(store, nid, depth=depth)
        else:
            text = to_mermaid(store, nid, depth=depth)
        if out is not None:
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]wrote[/] {out}")
        else:
            console.print(text)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Migration subcommands
# ---------------------------------------------------------------------------


@migrate_app.command("config")
def migrate_config_cmd(
    path: Path = typer.Argument(...),
    in_place: bool = typer.Option(False, "--in-place"),
    to_profiles: bool = typer.Option(
        False,
        "--to-profiles",
        help="Rewrite a bare-dict `model:` block into `model_profiles:` + a "
        "`model: <name>` selector (v0.1.8 structural migration).",
    ),
    profile_name: str = typer.Option(
        "default", "--profile-name", help="Name for the migrated profile."
    ),
) -> None:
    from ..lineage.migration import SchemaMigrationError, migrate_file

    # --to-profiles is a structural (comment-preserving) text rewrite, not a
    # schema_version hop, so it takes its own path through migration.py.
    if to_profiles:
        from ..lineage.migration import (
            migrate_model_to_profiles_text,
            rewrite_model_to_profiles_file,
        )

        if in_place:
            changed = rewrite_model_to_profiles_file(
                path, profile_name=profile_name, in_place=True
            )
            if changed:
                console.print(
                    f"[green]migrated[/] {path} → model_profiles "
                    f"(backup at {path}.pre-migration-bak)"
                )
            else:
                console.print(
                    f"[yellow]no change[/] {path} (already profile form or no "
                    "top-level `model:` block)"
                )
        else:
            raw = path.read_text(encoding="utf-8")
            new_text, changed = migrate_model_to_profiles_text(
                raw, profile_name=profile_name
            )
            console.print(new_text)
        return

    try:
        migrated = migrate_file(path, schema_kind="config", in_place=in_place)
    except SchemaMigrationError as e:
        console.print(f"[red]migrate-config error:[/] {e}")
        raise typer.Exit(code=1) from e
    if in_place:
        console.print(f"[green]migrated[/] {path} (backup at {path}.pre-migration-bak)")
    else:
        import yaml

        console.print(yaml.safe_dump(migrated, sort_keys=False))


@migrate_app.command("artifact-header")
def migrate_artifact_header_cmd(
    path: Path = typer.Argument(...),
    in_place: bool = typer.Option(True, "--in-place"),
) -> None:
    from ..lineage.migration import SchemaMigrationError, migrate_file

    try:
        migrated = migrate_file(path, schema_kind="artifact_header", in_place=in_place)
    except SchemaMigrationError as e:
        console.print(f"[red]migrate-artifact-header error:[/] {e}")
        raise typer.Exit(code=1) from e
    if in_place:
        console.print(f"[green]migrated[/] {path}")
    else:
        import json as _json

        console.print(_json.dumps(migrated, indent=2))


@migrate_app.command("checkpoint")
def migrate_checkpoint_cmd(
    path: Path = typer.Argument(..., help="step_<n>/ directory or manifest.json"),
    in_place: bool = typer.Option(True, "--in-place"),
) -> None:
    from ..lineage.migration import SchemaMigrationError, migrate_file

    manifest = path if path.is_file() else path / "manifest.json"
    if not manifest.exists():
        console.print(f"[red]no manifest at {manifest}[/]")
        raise typer.Exit(code=1)
    try:
        migrated = migrate_file(manifest, schema_kind="checkpoint_manifest", in_place=in_place)
    except SchemaMigrationError as e:
        console.print(f"[red]migrate-checkpoint error:[/] {e}")
        raise typer.Exit(code=1) from e
    if in_place:
        console.print(f"[green]migrated[/] {manifest}")
    else:
        import json as _json

        console.print(_json.dumps(migrated, indent=2))


# ---------------------------------------------------------------------------
# Debug / smoke
# ---------------------------------------------------------------------------


@app.command("dry-run")
def dry_run_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
    build: bool = typer.Option(
        False,
        "--build",
        help="Also import user_modules and construct the primary model from "
        "`model:`/`model_profiles:` or a `models:` set. No run dir, no training.",
    ),
) -> None:
    """Resolve a recipe and print the resolved config — no training.

    ``load_config`` alone does not build the model, so it cannot catch a recipe
    whose ``model:`` selection is wrong (e.g. a bare-dict block left un-migrated,
    or a selector naming a missing profile). ``--build`` constructs the model so
    the resolver path runs, making it a real migration/build verifier.
    """
    try:
        # Only populate the built-in registry when --build: a plain dry-run is a
        # pure config dump and must not pull in torch-heavy modules. (user_modules
        # stay imported either way — matching the pre-refactor behaviour.)
        cfg = load_config(
            config,
            overrides=list(overrides or []),
            register_components=build,
        )
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e
    if build:
        from ._runtime import _build_model

        try:
            # cfg came from load_config above with register_components=True, so
            # the registry has the same adapter set as a real `train` run.
            model = _build_model(cfg)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]build error:[/] {e}")
            raise typer.Exit(code=1) from e
        n_params = (
            sum(p.numel() for p in model.parameters())
            if hasattr(model, "parameters")
            else 0
        )
        console.print(
            f"[green]model built[/] = {type(model).__name__} ({n_params:,} params)"
        )
        return
    console.print(dump_resolved(cfg))


@app.command("overfit")
def overfit_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    n: int = typer.Option(200, "--n", help="Step count for the overfit smoke run."),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Overfit on the configured train loader for a few hundred steps.

    Equivalent to ``lighttrain train -c <cfg> ++trainer.max_steps=<n>
    ++trainer.val_every=0 ++trainer.ckpt_every=0``.
    """
    extra = list(overrides or []) + [
        f"++trainer.max_steps={int(n)}",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
    ]
    try:
        bundle = setup_run_from_config(config, overrides=extra)
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]overfit run_dir[/] = {bundle['run_dir']}")
    try:
        bundle["trainer"].fit()
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()
    console.print("[green]overfit complete[/]")


@app.command("profile")
def profile_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    steps: int = typer.Option(50, "--steps"),
) -> None:
    """Run ``torch.profiler`` over N training steps.

    Drops a Chrome-trace JSON under ``<run_dir>/diagnostics/profile_<ts>.json``
    and prints the top-10 kernels by CPU time to the console.
    """
    import time as _time

    import torch as _torch
    from torch.profiler import ProfilerActivity, profile, schedule

    overrides = [
        f"++trainer.max_steps={int(steps)}",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
        "++trainer.log_every=1000",
    ]
    try:
        bundle = setup_run_from_config(config, overrides=overrides)
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e
    trainer = bundle["trainer"]
    run_dir: Path = bundle["run_dir"]

    activities = [ProfilerActivity.CPU]
    if _torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"profile_{int(_time.time())}.json"

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=max(1, int(steps) - 2)),
        record_shapes=False,
    ) as prof:
        try:
            for _ in range(int(steps)):
                trainer.fit(steps=trainer.ctx.step + 1)
                prof.step()
        finally:
            if bundle.get("logger") is not None:
                try:
                    bundle["logger"].close()
                except Exception:
                    pass

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]chrome trace export failed: {e}[/]")
    table = prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=10
    )
    console.print(table)
    console.print(f"[green]profile trace[/] -> {trace_path}")


@app.command("inspect-data")
def inspect_data_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    n: int = typer.Option(32, "--n"),
    decoded: bool = typer.Option(False, "--decoded"),
) -> None:
    """Decode + summarize first N samples from the configured train loader."""
    try:
        bundle = setup_run_from_config(config, overrides=[])
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e
    data_module = bundle["data"]
    dataset = getattr(data_module, "dataset", None)
    if dataset is None:
        console.print("[red]data module has no `dataset` attribute[/]")
        raise typer.Exit(code=1)

    table = Table(title=f"first {n} samples")
    table.add_column("idx", style="cyan", justify="right")
    table.add_column("len", style="green", justify="right")
    table.add_column("kept_labels", style="green", justify="right")
    if decoded:
        table.add_column("decoded[:80]", style="white")

    tokenizer = getattr(data_module, "tokenizer", None)
    label_ignore = -100
    lengths: list[int] = []
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        ids = list(sample.get("input_ids", []))
        labels = list(sample.get("labels", ids))
        kept = sum(1 for x in labels if int(x) != label_ignore)
        lengths.append(len(ids))
        row = [str(i), str(len(ids)), f"{kept}/{len(labels)}"]
        if decoded:
            text = ""
            if tokenizer is not None and hasattr(tokenizer, "decode"):
                try:
                    text = tokenizer.decode(ids)[:80]
                except Exception:
                    text = "<decode error>"
            row.append(text.replace("\n", "\\n"))
        table.add_row(*row)
    console.print(table)
    if lengths:
        console.print(
            f"[green]length[/] min={min(lengths)} max={max(lengths)} "
            f"mean={sum(lengths) / len(lengths):.1f}"
        )


@app.command("resume")
def resume_cmd(
    run: Path = typer.Option(..., "--run", help="Existing run dir to resume from."),
    config: Path | None = typer.Option(
        None, "-c", "--config", help="Recipe YAML (defaults to run_dir/config.snapshot.yaml)."
    ),
    mode: str = typer.Option("functional", "--mode"),
) -> None:
    """Resume a previous run.

    Functional resume restores model + optimizer + scheduler + RNG state and
    continues training from the saved step. Exact (bit-identical) resume
    is not yet implemented and falls back to functional.
    """
    if mode not in ("functional", "exact"):
        console.print(f"[red]unknown --mode {mode!r} (use functional|exact)[/]")
        raise typer.Exit(code=1)
    if mode == "exact":
        console.print("[yellow]exact resume is not yet implemented; falling back to functional[/]")

    cfg_path = config or (run / "config.snapshot.yaml")
    if not cfg_path.exists():
        console.print(f"[red]no recipe found at {cfg_path}[/]")
        raise typer.Exit(code=1)

    try:
        # Resume keeps writing into the original
        # run_dir so logs, ckpt, and lineage stay on a single self-consistent
        # timeline. ``existing_run_dir=run`` short-circuits make_run_dir().
        bundle = setup_run_from_config(cfg_path, overrides=[], existing_run_dir=run)
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e

    trainer = bundle["trainer"]
    latest = trainer.ckpt_manager.latest()
    if latest is None:
        console.print(f"[red]no resumable checkpoint under {run}[/]")
        raise typer.Exit(code=1)
    trainer.load_checkpoint(latest)
    console.print(f"[green]resumed[/] from {latest}")
    try:
        trainer.fit()
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()
    console.print("[green]resume complete[/]")


@app.command("resume-verify")
def resume_verify_cmd(
    config: Path = typer.Option(..., "-c", "--config", help="Recipe YAML path."),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
    phase1_steps: int = typer.Option(..., "--phase1-steps", help="Steps before the checkpoint."),
    phase2_steps: int = typer.Option(..., "--phase2-steps", help="Steps after resuming."),
    tol: float = typer.Option(
        None,
        "--tol",
        help="Max allowed |Δloss| per step. Default 1e-2 (bf16-realistic); use a "
        "tighter value only for fp32 + single-worker bit-exact runs.",
    ),
) -> None:
    """Verify resume == single pass: compare step-aligned losses for a run of
    ``phase1+phase2`` steps against a checkpoint-and-continue at ``phase1``.

    \\b
    Example:
      lighttrain resume-verify -c recipe.yaml model=transformer \\
          --phase1-steps 5 --phase2-steps 5
    """
    from ..lab.resume_verify import DEFAULT_TOL, render_report, resume_verify

    try:
        report = resume_verify(
            config,
            phase1_steps,
            phase2_steps,
            tol=DEFAULT_TOL if tol is None else tol,
            overrides=list(overrides or []),
        )
    except (ConfigError, FileNotFoundError) as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e

    console.print(render_report(report))
    if not report.passed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Convert / export
# ---------------------------------------------------------------------------


@app.command("convert-checkpoint")
def convert_checkpoint_cmd(
    from_: str = typer.Option(..., "--from", help="Source format: safetensors | pt | hf"),
    to: str = typer.Option(..., "--to", help="Target format: safetensors | pt | hf"),
    path: Path = typer.Option(..., "--path", help="Path to checkpoint file or directory."),
    out: Path | None = typer.Option(None, "--out", help="Output path (default: next to source)."),
) -> None:
    """Convert a checkpoint between storage formats.

    \\b
    Supported conversions:
      pt → safetensors   Load torch .pt state dict, save as safetensors
      safetensors → pt   Load safetensors, save as torch .pt
      hf → safetensors   Load HuggingFace model dir, save merged safetensors

    \\b
    Examples:
      lighttrain convert-checkpoint --from pt --to safetensors --path model.pt
      lighttrain convert-checkpoint --from safetensors --to pt --path model.safetensors
    """
    import torch

    from_ = from_.lower().strip()
    to = to.lower().strip()

    if not path.exists():
        console.print(f"[red]path not found:[/] {path}")
        raise typer.Exit(code=1)

    try:
        if from_ in ("pt", "torch") and to == "safetensors":
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            if hasattr(state, "items"):
                state_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError("checkpoint is not a state dict")
            from safetensors.torch import save_file

            out_path = out or path.with_suffix(".safetensors")
            save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out_path))
            console.print(f"[green]written →[/] {out_path}")

        elif from_ == "safetensors" and to in ("pt", "torch"):
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
            out_path = out or path.with_suffix(".pt")
            torch.save(state_dict, str(out_path))
            console.print(f"[green]written →[/] {out_path}")

        elif from_ == "hf" and to == "safetensors":
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "hf→safetensors requires transformers: pip install transformers"
                ) from exc
            model = AutoModelForCausalLM.from_pretrained(str(path))
            from safetensors.torch import save_file

            out_path = out or (path / "model_merged.safetensors")
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(out_path),
            )
            console.print(f"[green]written →[/] {out_path}")

        else:
            console.print(
                f"[red]unsupported conversion:[/] {from_!r} → {to!r}. "
                "Supported: pt→safetensors, safetensors→pt, hf→safetensors"
            )
            raise typer.Exit(code=1)

    except Exception as exc:
        console.print(f"[red]convert-checkpoint error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("export")
def export_cmd(
    to: str = typer.Option(..., "--to", help="Export format: safetensors | hf | gguf"),
    ckpt: Path = typer.Option(..., "--ckpt", help="Checkpoint directory (step_<n>/)."),
    out: Path = typer.Option(..., "--out", help="Output path or directory."),
    config: Path | None = typer.Option(
        None, "-c", "--config", help="Recipe YAML (needed for hf / gguf export)."
    ),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Export a checkpoint to safetensors, HuggingFace, or GGUF format.

    \\b
    Examples:
      # Export model weights as a single safetensors file
      lighttrain export --to safetensors --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --out model.safetensors

      # Export as HuggingFace model directory (requires --config)
      lighttrain export --to hf --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --config recipes/pretrain_causal.yaml --out hf_model/

      # Export as GGUF (requires llama.cpp convert script on PATH)
      lighttrain export --to gguf --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --config recipes/pretrain_causal.yaml --out model.gguf
    """
    import torch

    to = to.lower().strip()

    if not ckpt.exists():
        console.print(f"[red]checkpoint not found:[/] {ckpt}")
        raise typer.Exit(code=1)

    # Locate model.safetensors or model.pt
    weight_file = ckpt / "model.safetensors"
    if not weight_file.exists():
        weight_file = ckpt / "model.pt"
    if not weight_file.exists():
        console.print(f"[red]no model weights found under:[/] {ckpt}")
        raise typer.Exit(code=1)

    try:
        if to == "safetensors":
            if weight_file.suffix == ".safetensors":
                import shutil as _sh

                out.parent.mkdir(parents=True, exist_ok=True)
                _sh.copy2(str(weight_file), str(out))
            else:
                state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
                from safetensors.torch import save_file

                out.parent.mkdir(parents=True, exist_ok=True)
                save_file({k: v.contiguous() for k, v in state.items()}, str(out))
            console.print(f"[green]exported →[/] {out}")

        elif to == "hf":
            if config is None:
                console.print("[red]--config required for hf export[/]")
                raise typer.Exit(code=1)
            try:
                from transformers import AutoConfig, AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "hf export requires transformers: pip install transformers"
                ) from exc
            model = _export_primary_model(config, overrides)
            if weight_file.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(weight_file))
            else:
                state_dict = torch.load(str(weight_file), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out))
            console.print(f"[green]exported →[/] {out}")

        elif to == "gguf":
            import shutil as _sh
            import subprocess as _sp

            if config is None:
                console.print("[red]--config required for gguf export[/]")
                raise typer.Exit(code=1)
            convert_script = _sh.which("convert_hf_to_gguf.py") or _sh.which("convert.py")
            if convert_script is None:
                console.print(
                    "[red]gguf export requires llama.cpp convert script on PATH.[/] "
                    "Clone https://github.com/ggerganov/llama.cpp and add to PATH."
                )
                raise typer.Exit(code=1)
            # Step 1: export to a temporary HF directory
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                from transformers import AutoConfig, AutoModelForCausalLM  # noqa: F401

                model = _export_primary_model(config, overrides)
                if weight_file.suffix == ".safetensors":
                    from safetensors.torch import load_file as _load_sf

                    state_dict = _load_sf(str(weight_file))
                else:
                    state_dict = torch.load(
                        str(weight_file), map_location="cpu", weights_only=True
                    )
                model.load_state_dict(state_dict, strict=False)
                model.save_pretrained(tmpdir)
                # Step 2: invoke llama.cpp conversion script on the HF directory
                out.parent.mkdir(parents=True, exist_ok=True)
                result = _sp.run(
                    [convert_script, tmpdir, "--outfile", str(out)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    console.print(f"[red]gguf conversion failed:[/]\n{result.stderr}")
                    raise typer.Exit(code=1)
            console.print(f"[green]exported →[/] {out}")

        else:
            console.print(
                f"[red]unknown export format:[/] {to!r}. "
                "Expected: safetensors | hf | gguf"
            )
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]export error:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _export_primary_model(config: Path, overrides: list[str] | None = None) -> Any:
    """Build the primary model to export, via the single source of truth
    (`build_primary_model`), so export supports every declaration form
    (`model:`+`model_profiles:` or a `models:` set) like every other command.

    Warns when the recipe declares multiple trainable models: export ships the
    primary one — the model the trainer checkpoints (`ctx.model`).
    """
    from ..config._models import build_primary_model

    cfg = load_config(config, overrides=list(overrides or []))
    model, n_trainable = build_primary_model(cfg)
    if n_trainable > 1:
        console.print(
            f"[yellow]note:[/] recipe declares {n_trainable} trainable models; "
            "exporting the primary one (the model the trainer checkpoints)."
        )
    return model


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


_INIT_RECIPE = '''\
# =============================================================================
# lighttrain recipe — runnable out of the box, and a guided tour.
#
#   lighttrain dry-run -c cfg.yaml                      # resolve & print, no train
#   lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50-step smoke run
#
# The ACTIVE part below (model/data/optim + sensible defaults) trains a tiny
# byte-level LM immediately — just drop a corpus at ``corpus.txt`` (one example
# per line). Everything under "OPTIONAL" is commented out: uncomment a block to
# grow into SFT / preference / RL / distillation / PEFT / distributed.
#
# Docs: docs/getting-started.md · docs/configuration.md · docs/recipes.md
# =============================================================================

mode: lab                 # lab = auto-attach diagnostics; prod = lean
seed: 1337
exp: demo                 # names the run dir: runs/<exp>/<ts>-<slug>-<hash>/
run_root: runs

# --- model -------------------------------------------------------------------
# The model is a config group: declare named profiles and select one with
# ``model: <name>`` (override on the CLI with ``model=big``). For a frozen
# teacher / multi-model, use ``models:`` instead (see OPTIONAL C).
model: demo
model_profiles:
  demo:
    name: tiny_lm         # built-ins: tiny_lm | hf_causal | tiny_rwkv | tiny_mamba | jepa
    vocab_size: 260       # byte tokenizer = 256 bytes + PAD/BOS/EOS/UNK
    d_model: 256
    n_layers: 4
    n_heads: 4
    max_seq_len: 256
    dropout: 0.0
    # tie_weights: true

# --- data --------------------------------------------------------------------
data:
  name: simple            # simple | prep_graph (DAG-based prep, OPTIONAL E)
  dataset:
    name: line_file_text  # line_file_text | preference_jsonl | artifact_joined
    path: corpus.txt      # <-- put one training example per line here
    max_len: 256
  tokenizer:
    name: byte            # byte (no external deps); hf tokenizers via processors
  collator:
    name: causal_lm       # causal_lm | preference | multimodal
    max_len: 256
  sampler:
    name: shuffle         # shuffle | sequential | length_grouped | curriculum | stateful_resumable
    seed: ${seed}
  batch_size: 4
  num_workers: 0
  # pin_memory: false
  # drop_last: false

# --- loss --------------------------------------------------------------------
# The algorithm lives here, not in a separate trainer (DPO/PPO/… are losses).
loss:
  name: cross_entropy     # ce | mlm | z_loss | composite | dpo | ppo_surrogate | grpo | kl_topk | ...

# --- optimizer ---------------------------------------------------------------
optim:
  name: adamw             # adamw | lion
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  # param_groups:                       # regex-based, first-match-wins
  #   - { pattern: ".*bias|.*norm.*", weight_decay: 0.0 }
  #   - { pattern: "attn|mlp", min_ndim: 2, module_type: Linear, weight_decay: 0.1 }

# --- scheduler ---------------------------------------------------------------
scheduler:
  name: warmup_cosine     # constant | linear | warmup_cosine | wsd
  warmup_steps: 50
  total_steps: ${trainer.max_steps}
  min_lr_ratio: 0.1

# --- engine ------------------------------------------------------------------
engine:
  name: standard          # standard | layer_offload (large models on small GPUs)
  mixed_precision: bf16   # no | fp16 | bf16
  # update_rule: { name: standard }     # standard | sam | mezo | rl

# --- trainer -----------------------------------------------------------------
trainer:
  name: pretrain          # pretrain | preference | reward_model | ppo | grpo | <your own>
  max_steps: 200
  val_every: 0
  ckpt_every: 100
  log_every: 25
  grad_clip: 1.0
  accumulate: 1

# --- callbacks ---------------------------------------------------------------
callbacks:
  - { name: throughput, window: 25 }
  - { name: nan_skip, max_skips: 10 }
  - { name: best_ckpt, monitor: loss, mode: min }
  - { name: lineage_recorder }          # writes lineage.sqlite
  # - { name: ema, decay: 0.999 }
  # - { name: early_stop, monitor: loss, patience: 5 }

# --- logger ------------------------------------------------------------------
logger:
  - { name: console, log_every: 25 }
  - { name: jsonl }
  # - { name: tensorboard }

# --- invariants (failure-first; lab mode runs these) -------------------------
# invariants:
#   - { name: loss_finite, action: abort }
#   - { name: grad_norm_bounded, max: 1000, action: warn }


# =============================================================================
# OPTIONAL blocks — uncomment one to switch paradigm. See docs/training.md.
# =============================================================================

# --- A. SFT over chat data (PrepGraph) ---------------------------------------
# Replace data: with a prep_graph data module; see recipes/sft_chat.yaml.

# --- B. Model variants (model_profiles) --------------------------------------
# Swap on the CLI with `model=big`; tweak with `model_profiles.base.d_model=384`.
# model: base
# model_profiles:
#   base: { name: tiny_lm, d_model: 256, n_layers: 4, n_heads: 8, max_seq_len: 256 }
#   big:  { name: tiny_lm, d_model: 512, n_layers: 8, n_heads: 8, max_seq_len: 256 }

# --- C. Multi-model: frozen teacher + trainable student ----------------------
# A custom trainer reads self.models["teacher"] / self.optimizers["student"].
# See examples/online_distill.py and recipes/online_distill_demo.yaml.
# models:
#   student: { spec: { name: tiny_lm, d_model: 128, n_layers: 4, n_heads: 4 }, trainable: true,  optimizer: main }
#   teacher: { spec: { name: tiny_lm, d_model: 128, n_layers: 4, n_heads: 4 }, trainable: false, checkpoint: path/to/teacher }
# optimizers:
#   main: { name: adamw, lr: 1.0e-3 }

# --- D. Online RL (PPO / GRPO) ----------------------------------------------
# trainer: { name: ppo, rollout_steps: 32, ppo_epochs: 4, rollout_backend: hf_generate, temperature: 1.0, top_p: 1.0 }
# loss:    { name: ppo_surrogate, clip_eps: 0.2 }
# judge:   { name: verifier, verify_pattern: "\\\\d+" }   # judge -> reward_fn via reward_adapter

# --- E. Preference (DPO / IPO / SimPO / ORPO / KTO) --------------------------
# trainer: { name: preference, ref_namespace: ref }
# loss:    { name: dpo, beta: 0.1 }

# --- F. PEFT (LoRA / QLoRA) --------------------------------------------------
# pip install -e ".[peft]"  (QLoRA also needs ".[quant]", Linux+CUDA)
# model: lora
# model_profiles:
#   lora:
#     name: lora
#     base: { name: hf_causal, pretrained: meta-llama/Llama-3.2-1B }
#     r: 8
#     target_modules: [q_proj, v_proj]

# --- G. Distributed (no model/trainer changes needed) ------------------------
# Launch with: torchrun --nproc_per_node=N -m lighttrain.cli train -c cfg.yaml
# parallel:
#   backend: nccl          # gloo for CPU/CI
#   dp: 4
#   grad_sync: { name: ddp, find_unused_parameters: false }   # ddp | fsdp | deepspeed

# --- H. PrepGraph data pipeline ----------------------------------------------
# data: { name: prep_graph }
# prep_graph:
#   nodes:
#     - { name: load,     kind: load,     config: { source: "jsonl:data/chat.jsonl" } }
#     - { name: tok,      kind: tokenize, inputs: [load] }
#     - { name: packed,   kind: pack,     inputs: [tok], config: { strategy: best_fit, seq_len: 256 } }
'''

_INIT_README = """\
# lighttrain project

Generated by `lighttrain init`. `cfg.yaml` is a fully runnable recipe (tiny_lm +
byte tokenizer + warmup_cosine) **and** a guided tour: the active blocks train
immediately, and the commented `OPTIONAL` blocks (A–H) switch the recipe into
SFT / preference / online RL / distillation / PEFT / distributed / PrepGraph.

## Quickstart

1. Drop a corpus at `corpus.txt` (one example per line).
2. `lighttrain dry-run -c cfg.yaml` — validate the recipe without training.
3. `lighttrain train -c cfg.yaml ++trainer.max_steps=50` — 50-step smoke run.

Outputs land in `runs/<exp>/<ts>-<slug>-<hash>/` (config snapshot, `env.json`,
`logs/metrics.jsonl`, `checkpoints/`).

## Grow the recipe

Uncomment one `OPTIONAL` block in `cfg.yaml`:

| Block | Paradigm |
| ----- | -------- |
| B | model variants (`model_profiles`) |
| C | multi-model (frozen teacher + student) |
| D | online RL (PPO / GRPO) |
| E | preference (DPO / IPO / SimPO / ORPO / KTO) |
| F | PEFT (LoRA / QLoRA) |
| G | distributed (DDP / FSDP / ZeRO) |
| H | PrepGraph data pipeline |

## Docs

See the framework's `docs/` directory — start at `docs/getting-started.md`,
`docs/configuration.md`, and `docs/recipes.md`.
"""


@app.command("init")
def init_cmd(
    path: Path = typer.Argument(..., help="Target directory (created if absent)."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Generate a minimal recipe + run-dir skeleton."""
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not force:
        console.print(f"[red]target {path} is not empty (pass --force to overwrite)[/]")
        raise typer.Exit(code=1)
    path.mkdir(parents=True, exist_ok=True)
    (path / "cfg.yaml").write_text(_INIT_RECIPE, encoding="utf-8")
    (path / "README.md").write_text(_INIT_README, encoding="utf-8")
    (path / "runs").mkdir(exist_ok=True)
    (path / "artifacts").mkdir(exist_ok=True)

    table = Table(title="lighttrain init")
    table.add_column("file", style="cyan")
    table.add_column("status", style="green")
    table.add_row(str(path / "cfg.yaml"), "created")
    table.add_row(str(path / "README.md"), "created")
    table.add_row(str(path / "runs/"), "created")
    table.add_row(str(path / "artifacts/"), "created")
    console.print(table)
    console.print(f"[green]initialized lighttrain project at {path}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
