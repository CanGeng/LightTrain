"""Evaluation commands: eval / regression-gate."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from lighttrain.cli._context import console
from lighttrain.cli._helpers import _eval_perplexity
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.utils.env import load_dotenv_if_present

_log = logging.getLogger(__name__)


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
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "cli eval: checkpoint load failed; "
                        "scoring will proceed on untrained weights",
                        exc_info=True,
                    )
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
                from lighttrain.eval.suite import Evaluator
                if isinstance(evaluator, Evaluator):
                    report = evaluator.run(model, step=0, device=device, force=True)
                    if report is not None:
                        metrics.update(report.metrics)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "cli eval: evaluator suite run failed; "
                    "its metrics will be missing from the report",
                    exc_info=True,
                )
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
            _log.warning(
                "cli eval: temp run dir cleanup failed (likely open handles); "
                "leaving it on disk",
                exc_info=True,
            )


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

    from lighttrain.builtin_plugins.callbacks.invariants.regression_gate import RegressionGate
    from lighttrain.eval.metrics import perplexity
    from lighttrain.eval.suite import EvalReport

    load_dotenv_if_present()
    bundle = setup_run_from_config(config)
    trainer = bundle["trainer"]

    if checkpoint is not None:
        ckpt_manager = getattr(trainer, "ckpt_manager", None)
        if ckpt_manager is not None:
            try:
                trainer.load_checkpoint(checkpoint)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "cli regression-gate: checkpoint load failed; "
                    "gate metrics will reflect untrained weights",
                    exc_info=True,
                )
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
            except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAIL[/] {exc}")
        raise typer.Exit(code=1) from None
