"""Typer CLI entry point.

Run ``lighttrain --help`` to see the full command map.

This module is a thin *assembler*: the command implementations live in
``lighttrain.cli.commands.*`` and are registered here, in their original source
order, so ``lighttrain --help`` lists them unchanged. The ``app`` /
``lineage_app`` / ``migrate_app`` Typer objects are owned here and nowhere else.
"""

from __future__ import annotations

import typer

from .. import __version__
from ..utils.env import load_dotenv_if_present
from ._context import console
from ._helpers import (
    _flatten_patch_to_overrides,  # re-export — tests import it from here
)
from .commands import (
    artifacts,
    determinism,
    diagnostics,
    experiment,
    prep,
    run,
    scaffold,
)
from .commands import eval as eval_commands
from .commands import lineage as lineage_commands
from .commands import migrate as migrate_commands
from .commands import tokenizer as tokenizer_commands

__all__ = ["_flatten_patch_to_overrides", "app"]

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


# --- command registration ----------------------------------------------------
# Order matters: this Typer version lists commands in registration order, so the
# sequence below must match the original source order (verified against
# ``lighttrain --help``). Sub-typer groups are listed after the direct commands.
app.command("train")(run.train_cmd)
app.command("prep")(prep.prep_cmd)
app.command("prep-graph")(prep.prep_graph_cmd)
app.command("prep-clean")(prep.prep_clean_cmd)
app.command("prep-status")(prep.prep_status_cmd)
app.command("produce-artifact")(artifacts.produce_artifact_cmd)
app.command("sweep")(experiment.sweep_cmd)
app.command("compare")(experiment.compare_cmd)
app.command("fork")(experiment.fork_cmd)
app.command("replay")(determinism.replay_cmd)
app.command("estimate")(diagnostics.estimate_cmd)
app.command("eval")(eval_commands.eval_cmd)
app.command("regression-gate")(eval_commands.regression_gate_cmd)
app.command("freeze-step")(determinism.freeze_step_cmd)
app.command("replay-step")(determinism.replay_step_cmd)
app.command("doctor")(diagnostics.doctor_cmd)
app.command("dry-run")(diagnostics.dry_run_cmd)
app.command("overfit")(diagnostics.overfit_cmd)
app.command("profile")(diagnostics.profile_cmd)
app.command("inspect-data")(diagnostics.inspect_data_cmd)
app.command("resume")(run.resume_cmd)
app.command("resume-verify")(run.resume_verify_cmd)
app.command("convert-checkpoint")(artifacts.convert_checkpoint_cmd)
app.command("export")(artifacts.export_cmd)
app.command("init")(scaffold.init_cmd)
app.command("prune-tokenizer")(tokenizer_commands.prune_tokenizer_cmd)
app.command("check-tokenizer")(tokenizer_commands.check_tokenizer_cmd)

lineage_app.command("tag")(lineage_commands.lineage_tag_cmd)
lineage_app.command("untag")(lineage_commands.lineage_untag_cmd)
lineage_app.command("invalidate")(lineage_commands.lineage_invalidate_cmd)
lineage_app.command("pin")(lineage_commands.lineage_pin_cmd)
lineage_app.command("gc")(lineage_commands.lineage_gc_cmd)
lineage_app.command("prune-orphans")(lineage_commands.lineage_prune_cmd)
lineage_app.command("graph")(lineage_commands.lineage_graph_cmd)

migrate_app.command("config")(migrate_commands.migrate_config_cmd)
migrate_app.command("artifact-header")(migrate_commands.migrate_artifact_header_cmd)
migrate_app.command("checkpoint")(migrate_commands.migrate_checkpoint_cmd)


if __name__ == "__main__":  # pragma: no cover
    app()
