"""Run-lifecycle commands: train / resume / resume-verify."""

from __future__ import annotations

from pathlib import Path

import typer

from lighttrain.cli._context import console
from lighttrain.cli._helpers import (
    _append_run_summary,
    _eval_perplexity,
    _final_loss_from_run,
    _flatten_patch_to_overrides,
)
from lighttrain.cli._runtime import _validate_mode_override, setup_run_from_config
from lighttrain.config import ConfigError, dump_resolved, load_config


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
            cfg_pc = load_config(
                config,
                overrides=overrides,
                import_user_modules=False,
                register_components=False,
            )
            if mode is not None:
                cfg_pc.mode = _validate_mode_override(mode)  # type: ignore[union-attr]
        except (ConfigError, FileNotFoundError) as e:
            console.print(f"[red]config error:[/] {e}")
            raise typer.Exit(code=1) from e
        console.print(dump_resolved(cfg_pc))
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
    from lighttrain.lab.resume_verify import DEFAULT_TOL, render_report, resume_verify

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
