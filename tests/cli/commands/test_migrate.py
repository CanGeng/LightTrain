"""Tests for ``lighttrain/cli/commands/migrate.py``.

Covers every reachable branch in the three migrate sub-commands:

* ``migrate config``   — happy-path (dry, in-place), to-profiles variants, error path
* ``migrate artifact-header`` — happy-path (dry, in-place), error path
* ``migrate checkpoint``    — happy-path (dir, file), missing manifest, error path

All tests use ``typer.testing.CliRunner`` against the assembled ``app``
and monkeypatch the underlying ``migration`` module functions so no real
schema-DAG setup is required.

Skipped: any branch that needs a live GPU / distributed process group or a
real end-to-end schema migration chain (those are covered by the migration
module's own tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test."""
    return CliRunner()


def _write_yaml(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write *payload* as YAML into *tmp_path/<name>* and return the path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write *payload* as JSON into *tmp_path/<name>* and return the path."""
    p = tmp_path / name
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# The module path string used in all monkeypatches
_MIGRATION_MOD = "lighttrain.observability.lineage.migration"

# ---------------------------------------------------------------------------
# migrate config — dry-run (no --in-place)
# Lines 63-73
# ---------------------------------------------------------------------------


class TestMigrateConfigDry:
    """``migrate config <path>`` without ``--in-place`` dumps YAML to stdout."""

    def test_invariant_exits_zero_and_prints_yaml(self, runner, tmp_path):
        """Happy path: migrate_file returns a dict; the command prints YAML and exits 0.

        Covers lines 63-64, 68 (branch false), 71-73.
        """
        cfg = _write_yaml(tmp_path, "cfg.yaml", {"schema_version": "0.3", "mode": "lab"})
        migrated = {"schema_version": "0.4", "mode": "lab"}

        with patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated) as mock_mf:
            res = runner.invoke(app, ["migrate", "config", str(cfg)])

        assert res.exit_code == 0, res.stdout
        # migrate_file called with in_place=False
        mock_mf.assert_called_once()
        call_kwargs = mock_mf.call_args
        assert call_kwargs.kwargs.get("in_place") is False or call_kwargs.args[1:] == ()
        # Output should contain YAML-serialised result
        assert "schema_version" in res.stdout
        assert "0.4" in res.stdout

    def test_invariant_schema_migration_error_exits_one(self, runner, tmp_path):
        """SchemaMigrationError inside migrate_file → exit code 1, error in stdout.

        Covers lines 65-67.
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        cfg = _write_yaml(tmp_path, "cfg.yaml", {"schema_version": "0.3"})

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("boom")):
            res = runner.invoke(app, ["migrate", "config", str(cfg)])

        assert res.exit_code == 1
        assert "migrate-config error" in res.stdout or "boom" in res.stdout


# ---------------------------------------------------------------------------
# migrate config — in-place
# Lines 68-69
# ---------------------------------------------------------------------------


class TestMigrateConfigInPlace:
    """``migrate config <path> --in-place`` prints a 'migrated' banner."""

    def test_invariant_exits_zero_and_prints_migrated_banner(self, runner, tmp_path):
        """Happy path with --in-place: stdout says 'migrated' and exit 0.

        Covers lines 63-64, 68-69.
        """
        cfg = _write_yaml(tmp_path, "cfg.yaml", {"schema_version": "0.3", "mode": "lab"})
        migrated = {"schema_version": "0.4", "mode": "lab"}

        with patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated):
            res = runner.invoke(app, ["migrate", "config", str(cfg), "--in-place"])

        assert res.exit_code == 0, res.stdout
        # Rich markup stripped by CliRunner; plain text should include 'migrated'
        assert "migrated" in res.stdout
        assert str(cfg) in res.stdout

    def test_invariant_schema_error_with_in_place_exits_one(self, runner, tmp_path):
        """SchemaMigrationError with --in-place still exits 1.

        Covers error path lines 65-67 with --in-place flag.
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        cfg = _write_yaml(tmp_path, "cfg.yaml", {"schema_version": "0.3"})

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("bad")):
            res = runner.invoke(app, ["migrate", "config", str(cfg), "--in-place"])

        assert res.exit_code == 1
        assert "migrate-config error" in res.stdout or "bad" in res.stdout


# ---------------------------------------------------------------------------
# migrate config --to-profiles
# Lines 35-61
# ---------------------------------------------------------------------------


class TestMigrateConfigToProfiles:
    """``migrate config <path> --to-profiles`` takes the text-rewrite path."""

    def test_invariant_dry_run_prints_new_text(self, runner, tmp_path):
        """``--to-profiles`` (no --in-place): prints the new text, exit 0.

        Covers lines 55-60 (else branch of ``if in_place``).
        """
        raw = "model:\n  name: tiny_lm\n  d_model: 128\nmode: lab\n"
        cfg = tmp_path / "recipe.yaml"
        cfg.write_text(raw, encoding="utf-8")

        new_text = "model: default\nmodel_profiles:\n  default:\n    name: tiny_lm\n"
        with patch(f"{_MIGRATION_MOD}.migrate_model_to_profiles_text", return_value=(new_text, True)):
            res = runner.invoke(app, ["migrate", "config", str(cfg), "--to-profiles"])

        assert res.exit_code == 0, res.stdout
        assert "model_profiles" in res.stdout or "default" in res.stdout

    def test_invariant_in_place_changed_prints_migrated_banner(self, runner, tmp_path):
        """``--to-profiles --in-place`` on a file with a model block writes the file
        and prints a 'migrated' + 'model_profiles' banner.

        Covers lines 41-49 (in_place=True, changed=True).
        """
        raw = "model:\n  name: tiny_lm\nmode: lab\n"
        cfg = tmp_path / "recipe.yaml"
        cfg.write_text(raw, encoding="utf-8")

        with patch(f"{_MIGRATION_MOD}.rewrite_model_to_profiles_file", return_value=True):
            res = runner.invoke(
                app,
                ["migrate", "config", str(cfg), "--to-profiles", "--in-place"],
            )

        assert res.exit_code == 0, res.stdout
        assert "migrated" in res.stdout
        assert "model_profiles" in res.stdout

    def test_invariant_in_place_unchanged_prints_no_change_banner(self, runner, tmp_path):
        """``--to-profiles --in-place`` on an already-migrated file prints 'no change'.

        Covers lines 50-54 (in_place=True, changed=False).
        """
        raw = "model: default\nmodel_profiles:\n  default:\n    name: tiny_lm\n"
        cfg = tmp_path / "recipe.yaml"
        cfg.write_text(raw, encoding="utf-8")

        with patch(f"{_MIGRATION_MOD}.rewrite_model_to_profiles_file", return_value=False):
            res = runner.invoke(
                app,
                ["migrate", "config", str(cfg), "--to-profiles", "--in-place"],
            )

        assert res.exit_code == 0, res.stdout
        assert "no change" in res.stdout

    def test_invariant_to_profiles_with_custom_profile_name(self, runner, tmp_path):
        """``--profile-name`` is forwarded to the rewrite helper.

        Covers the ``profile_name`` parameter plumbing (lines 36-61).
        """
        raw = "model:\n  name: tiny_lm\nmode: lab\n"
        cfg = tmp_path / "recipe.yaml"
        cfg.write_text(raw, encoding="utf-8")

        with patch(f"{_MIGRATION_MOD}.rewrite_model_to_profiles_file", return_value=True) as mock_rw:
            runner.invoke(
                app,
                [
                    "migrate", "config", str(cfg),
                    "--to-profiles", "--in-place", "--profile-name", "custom",
                ],
            )

        mock_rw.assert_called_once()
        call_kwargs = mock_rw.call_args
        assert call_kwargs.kwargs.get("profile_name") == "custom"


# ---------------------------------------------------------------------------
# migrate artifact-header — dry-run (in_place=False)
# Lines 80-95
#
# Note: the CLI flag ``--in-place`` for artifact-header defaults to True and
# has no negation form (Typer is_flag=True with no secondary).  The
# ``in_place=False`` branch (lines 93-95) is therefore unreachable via the
# CliRunner.  We exercise it by calling the command function directly with
# a rich Console capture so the console.print output is visible.
# ---------------------------------------------------------------------------


class TestMigrateArtifactHeaderDry:
    """``migrate_artifact_header_cmd`` with ``in_place=False`` dumps JSON."""

    def test_invariant_dry_run_prints_json_via_direct_call(self, tmp_path):
        """Happy path with in_place=False: function emits JSON via console.print.

        Covers lines 85-86, 90 (false branch), 93-95.
        The CLI ``--in-place`` flag has no negation form, so we call the
        underlying function directly.
        """
        from io import StringIO

        import rich.console as rich_console

        from lighttrain.cli.commands.migrate import migrate_artifact_header_cmd

        header = _write_json(
            tmp_path,
            "artifact_header.json",
            {"schema_version": "0.3", "kind": "checkpoint"},
        )
        migrated = {"schema_version": "0.4", "kind": "checkpoint", "framework_version": "torch:2.0"}

        buf = StringIO()
        fake_console = rich_console.Console(file=buf, highlight=False)

        with (
            patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated),
            patch("lighttrain.cli.commands.migrate.console", fake_console),
        ):
            migrate_artifact_header_cmd(path=header, in_place=False)

        output = buf.getvalue()
        # Output should contain JSON with the migrated payload
        assert "schema_version" in output
        assert "0.4" in output
        parsed = json.loads(output.strip())
        assert parsed["schema_version"] == "0.4"

    def test_invariant_schema_error_exits_one(self, runner, tmp_path):
        """SchemaMigrationError → exit 1 with error text in stdout.

        Covers lines 87-89.  The default ``--in-place`` flag is True, so we
        can reach the error path via the CliRunner normally.
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        header = _write_json(tmp_path, "hdr.json", {"schema_version": "0.3"})

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("hdr-boom")):
            res = runner.invoke(app, ["migrate", "artifact-header", str(header)])

        assert res.exit_code == 1
        assert "migrate-artifact-header error" in res.stdout or "hdr-boom" in res.stdout


# ---------------------------------------------------------------------------
# migrate artifact-header — in-place (default)
# Lines 90-91
# ---------------------------------------------------------------------------


class TestMigrateArtifactHeaderInPlace:
    """``migrate artifact-header <path>`` (default ``--in-place``) prints banner."""

    def test_invariant_exits_zero_and_prints_migrated_banner(self, runner, tmp_path):
        """Default in-place: stdout contains 'migrated' and the path.

        Covers lines 85-86, 90-91.
        """
        header = _write_json(tmp_path, "hdr.json", {"schema_version": "0.3"})
        migrated = {"schema_version": "0.4"}

        with patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated):
            res = runner.invoke(app, ["migrate", "artifact-header", str(header)])

        assert res.exit_code == 0, res.stdout
        assert "migrated" in res.stdout
        assert str(header) in res.stdout

    def test_invariant_schema_error_in_place_exits_one(self, runner, tmp_path):
        """SchemaMigrationError with the default in-place flag → exit 1.

        Covers error path lines 87-89 with in_place=True.
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        header = _write_json(tmp_path, "hdr.json", {"schema_version": "0.3"})

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("err")):
            res = runner.invoke(app, ["migrate", "artifact-header", str(header)])

        assert res.exit_code == 1


# ---------------------------------------------------------------------------
# migrate checkpoint — directory path (contains manifest.json)
# Lines 102-121
# ---------------------------------------------------------------------------


class TestMigrateCheckpointDirectory:
    """``migrate checkpoint <step_dir>`` resolves to ``<dir>/manifest.json``."""

    def test_invariant_resolves_manifest_from_dir_and_exits_zero(self, runner, tmp_path):
        """Passing a directory that contains manifest.json exits 0, prints banner.

        Covers lines 107 (dir branch), 108 (manifest exists), 111-113, 116-117.
        """
        step_dir = tmp_path / "step_100"
        step_dir.mkdir()
        manifest = step_dir / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "0.3"}), encoding="utf-8")

        migrated = {"schema_version": "0.4"}

        with patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated):
            res = runner.invoke(app, ["migrate", "checkpoint", str(step_dir)])

        assert res.exit_code == 0, res.stdout
        assert "migrated" in res.stdout

    def test_invariant_missing_manifest_in_dir_exits_one(self, runner, tmp_path):
        """A directory with no manifest.json → exit 1, message names the manifest path.

        Covers lines 108-110.
        """
        step_dir = tmp_path / "step_200"
        step_dir.mkdir()

        res = runner.invoke(app, ["migrate", "checkpoint", str(step_dir)])

        assert res.exit_code == 1
        assert "no manifest" in res.stdout or "manifest" in res.stdout

    def test_invariant_schema_error_exits_one(self, runner, tmp_path):
        """SchemaMigrationError during checkpoint migration → exit 1.

        Covers lines 113-115.
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        step_dir = tmp_path / "step_300"
        step_dir.mkdir()
        manifest = step_dir / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "0.3"}), encoding="utf-8")

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("ckpt-err")):
            res = runner.invoke(app, ["migrate", "checkpoint", str(step_dir)])

        assert res.exit_code == 1
        assert "migrate-checkpoint error" in res.stdout or "ckpt-err" in res.stdout


# ---------------------------------------------------------------------------
# migrate checkpoint — file path (manifest.json directly)
# Lines 107, 111-121
# ---------------------------------------------------------------------------


class TestMigrateCheckpointFile:
    """``migrate checkpoint <manifest.json>`` accepts a file path directly."""

    def test_invariant_file_path_in_place_exits_zero(self, runner, tmp_path):
        """Passing the manifest.json file directly (in-place default) exits 0.

        Covers line 107 (is_file() true branch), 111-113, 116-117.
        """
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "0.3"}), encoding="utf-8")
        migrated = {"schema_version": "0.4"}

        with patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated):
            res = runner.invoke(app, ["migrate", "checkpoint", str(manifest)])

        assert res.exit_code == 0, res.stdout
        assert "migrated" in res.stdout
        assert "manifest" in res.stdout

    def test_invariant_file_path_no_in_place_prints_json(self, tmp_path):
        """Passing manifest.json with ``in_place=False`` prints JSON via console.

        Covers lines 118-121.
        The CLI ``--in-place`` flag has no negation form for checkpoint, so we
        call the underlying function directly.
        """
        from io import StringIO

        import rich.console as rich_console

        from lighttrain.cli.commands.migrate import migrate_checkpoint_cmd

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "0.3"}), encoding="utf-8")
        migrated = {"schema_version": "0.4", "steps": 100}

        buf = StringIO()
        fake_console = rich_console.Console(file=buf, highlight=False)

        with (
            patch(f"{_MIGRATION_MOD}.migrate_file", return_value=migrated),
            patch("lighttrain.cli.commands.migrate.console", fake_console),
        ):
            migrate_checkpoint_cmd(path=manifest, in_place=False)

        output = buf.getvalue()
        # Output must be parseable JSON
        parsed = json.loads(output.strip())
        assert parsed["schema_version"] == "0.4"
        assert parsed["steps"] == 100

    def test_invariant_missing_file_path_exits_one(self, runner, tmp_path):
        """Passing a non-existent manifest file → exit 1.

        Covers lines 108-110 (manifest.exists() false when given a file path
        that doesn't exist).
        """
        nonexistent = tmp_path / "no_such_manifest.json"
        # do NOT create it
        res = runner.invoke(app, ["migrate", "checkpoint", str(nonexistent)])

        assert res.exit_code == 1
        assert "no manifest" in res.stdout or "manifest" in res.stdout

    def test_invariant_schema_error_exits_one_via_cli(self, runner, tmp_path):
        """SchemaMigrationError on the default (in-place) path → CLI exit 1.

        Covers lines 113-115.  We use the CLI with the default --in-place flag
        (the only toggle available for checkpoint).
        """
        from lighttrain.observability.lineage.migration import SchemaMigrationError

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "0.3"}), encoding="utf-8")

        with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("dry-ckpt-err")):
            res = runner.invoke(app, ["migrate", "checkpoint", str(manifest)])

        assert res.exit_code == 1
        assert "migrate-checkpoint error" in res.stdout or "dry-ckpt-err" in res.stdout


# ---------------------------------------------------------------------------
# Parametric: all three commands print a nonzero exit for SchemaMigrationError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,extra_args,make_path",
    [
        (
            ["migrate", "config"],
            [],
            lambda t: _write_yaml(t, "c.yaml", {"schema_version": "0.3"}),
        ),
        (
            # artifact-header: --in-place is the only toggleable form (default True)
            ["migrate", "artifact-header"],
            [],
            lambda t: _write_json(t, "h.json", {"schema_version": "0.3"}),
        ),
        (
            # checkpoint: --in-place is the only toggleable form (default True)
            ["migrate", "checkpoint"],
            [],
            lambda t: _write_json(t, "manifest.json", {"schema_version": "0.3"}),
        ),
    ],
)
def test_invariant_all_migrate_subcommands_exit_one_on_schema_error(
    runner, tmp_path, cmd, extra_args, make_path
):
    """All three migrate sub-commands exit non-zero when SchemaMigrationError is raised.

    Contract: error handling is consistent across the three subcommands.
    """
    from lighttrain.observability.lineage.migration import SchemaMigrationError

    path = make_path(tmp_path)

    with patch(f"{_MIGRATION_MOD}.migrate_file", side_effect=SchemaMigrationError("uniform")):
        res = runner.invoke(app, cmd + [str(path)] + extra_args)

    assert res.exit_code == 1
