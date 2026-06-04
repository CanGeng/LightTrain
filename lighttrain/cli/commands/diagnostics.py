"""Diagnostics / smoke commands: estimate / doctor / dry-run / overfit / profile / inspect-data."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from lighttrain.cli._context import console
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import ConfigError, dump_resolved, load_config


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

    from lighttrain.lab.estimate import estimate, report_to_dict

    try:
        # load_config populates the registry (register_components default True);
        # estimate() also self-imports as a safety net for direct dict callers.
        cfg = load_config(config, overrides=list(overrides or []))
    except ConfigError as e:
        console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=1) from e

    rpt = estimate(cfg)  # type: ignore[arg-type]
    table = Table(title="lighttrain estimate", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value")

    def _fmt_bytes(n: float) -> str:
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
        from lighttrain.checkpoint.manager import CheckpointManager

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
        from lighttrain.lineage.store import LineageStore
        from lighttrain.prepgraph._fp import SCHEMA_VERSION

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
            n_failures = sum(
                1
                for line in cb_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:  # noqa: BLE001
            n_failures = 0
        if n_failures > 0:
            console.print(
                f"[yellow]…  callback report[/] {n_failures} isolated failure(s); "
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
        from lighttrain.cli._runtime import _build_model

        try:
            # cfg came from load_config above with register_components=True, so
            # the registry has the same adapter set as a real `train` run.
            model = _build_model(cfg)  # type: ignore[arg-type]
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
                except Exception:  # noqa: BLE001
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
                except Exception:  # noqa: BLE001
                    text = "<decode error>"
            row.append(text.replace("\n", "\\n"))
        table.add_row(*row)
    console.print(table)
    if lengths:
        console.print(
            f"[green]length[/] min={min(lengths)} max={max(lengths)} "
            f"mean={sum(lengths) / len(lengths):.1f}"
        )
