"""Determinism / replay commands: replay / freeze-step / replay-step."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from lighttrain.cli._context import console
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import ConfigError

_log = logging.getLogger(__name__)


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
    from lighttrain.minimal import build_minimal_model, load_state
    from lighttrain.protocols import LossContext

    try:
        import lighttrain.builtin_plugins.models.adapters  # noqa: F401 — populate registry
    except Exception:  # noqa: BLE001
        _log.warning(
            "cli replay-step: model adapter registry import failed; "
            "proceeding with whatever is already registered",
            exc_info=True,
        )

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
        except Exception:  # noqa: BLE001
            _log.warning(
                "cli freeze-step: could not parse step number from checkpoint %r; "
                "skipping it as a restore candidate",
                p.name,
                exc_info=True,
            )
            continue
        if n <= step and (target is None or n > int(target.name.split("_", 1)[1])):
            target = p
    if target is not None:
        try:
            trainer.load_checkpoint(target)
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "cli freeze-step: checkpoint restore of %s failed; "
                "continuing from current trainer state",
                target,
                exc_info=True,
            )
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
            except Exception:  # noqa: BLE001
                _log.warning(
                    "cli freeze-step: logger close failed during cleanup; ignoring",
                    exc_info=True,
                )
    zips = sorted((run / "frozen_steps").glob("*.zip"))
    if zips:
        console.print(f"[green]frozen step bundle[/] -> {zips[-1]}")
    else:
        console.print("[yellow]no bundle produced (check callback wiring)[/]")


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
    from lighttrain.diagnostics.frozen_step import (
        read_frozen_step_bundle,
        replay_step_bundle,
    )

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
