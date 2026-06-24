"""Migration subcommand functions: config / artifact-header / checkpoint.

Functions only — the ``migrate`` sub-typer is owned by the ``_app`` assembler.
"""

from __future__ import annotations

from pathlib import Path

import typer

from lighttrain.cli._context import console


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
    from lighttrain.observability.lineage.migration import (
        SchemaMigrationError,
        migrate_file,
    )

    # --to-profiles is a structural (comment-preserving) text rewrite, not a
    # schema_version hop, so it takes its own path through migration.py.
    if to_profiles:
        from lighttrain.observability.lineage.migration import (
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


def migrate_artifact_header_cmd(
    path: Path = typer.Argument(...),
    in_place: bool = typer.Option(True, "--in-place"),
) -> None:
    from lighttrain.observability.lineage.migration import (
        SchemaMigrationError,
        migrate_file,
    )

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


def migrate_checkpoint_cmd(
    path: Path = typer.Argument(..., help="step_<n>/ directory or manifest.json"),
    in_place: bool = typer.Option(True, "--in-place"),
) -> None:
    from lighttrain.observability.lineage.migration import (
        SchemaMigrationError,
        migrate_file,
    )

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
