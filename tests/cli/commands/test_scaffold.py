"""Tests for ``lighttrain.cli.commands.scaffold`` — the ``init`` command.

Covers every reachable line in the module, focusing on the uncovered blocks:
  - lines 228-246: init_cmd body — happy path, non-empty dir guard (error path),
    --force override, scaffold output (Table rows, final console message).

Patterns: typer.testing.CliRunner + tmp_path; no GPU, no network, no real
training.  All branches in init_cmd are reachable without hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test (matches house style in tests/cli/test_app.py)."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Happy-path: fresh directory
# ---------------------------------------------------------------------------


def test_invariant_init_exits_zero_on_fresh_dir(runner: CliRunner, tmp_path: Path) -> None:
    """``init <new_dir>`` on a path that does not yet exist exits 0.

    Goal: cover lines 228-246 (the whole body) on the happy path.
    """
    target = tmp_path / "project"
    res = runner.invoke(app, ["init", str(target)])
    assert res.exit_code == 0, res.stdout


def test_invariant_init_creates_cfg_yaml(runner: CliRunner, tmp_path: Path) -> None:
    """After ``init``, ``cfg.yaml`` must exist in the target directory.

    Covers line 233: ``(path / "cfg.yaml").write_text(...)``
    """
    target = tmp_path / "my_proj"
    runner.invoke(app, ["init", str(target)])
    assert (target / "cfg.yaml").exists()


def test_invariant_init_cfg_yaml_contains_recipe_content(
    runner: CliRunner, tmp_path: Path
) -> None:
    """cfg.yaml written by init must contain key recipe landmarks.

    Covers that ``_INIT_RECIPE`` is actually written (not an empty file).
    Checks for 'mode: lab' and 'model:' which appear in the recipe template.
    """
    target = tmp_path / "my_proj"
    runner.invoke(app, ["init", str(target)])
    content = (target / "cfg.yaml").read_text(encoding="utf-8")
    assert "mode: lab" in content
    assert "model:" in content


def test_invariant_init_creates_readme(runner: CliRunner, tmp_path: Path) -> None:
    """After ``init``, ``README.md`` must exist.

    Covers line 234: ``(path / "README.md").write_text(...)``
    """
    target = tmp_path / "my_proj"
    runner.invoke(app, ["init", str(target)])
    assert (target / "README.md").exists()


def test_invariant_init_creates_runs_subdir(runner: CliRunner, tmp_path: Path) -> None:
    """After ``init``, a ``runs/`` subdirectory must exist.

    Covers line 235: ``(path / "runs").mkdir(exist_ok=True)``
    """
    target = tmp_path / "my_proj"
    runner.invoke(app, ["init", str(target)])
    assert (target / "runs").is_dir()


def test_invariant_init_creates_artifacts_subdir(runner: CliRunner, tmp_path: Path) -> None:
    """After ``init``, an ``artifacts/`` subdirectory must exist.

    Covers line 236: ``(path / "artifacts").mkdir(exist_ok=True)``
    """
    target = tmp_path / "my_proj"
    runner.invoke(app, ["init", str(target)])
    assert (target / "artifacts").is_dir()


def test_invariant_init_prints_table_and_final_message(
    runner: CliRunner, tmp_path: Path
) -> None:
    """init stdout must contain a Rich table and the success message.

    Covers lines 238-246: Table construction + console.print(table) +
    console.print(success message).

    Note: Rich truncates long cell content with '…' when the terminal is narrow,
    so we cannot rely on full filenames appearing when pytest tmp paths are deep.
    We assert on stable markers: the table title, the status column ('created'),
    and the 'initialized' final message.
    """
    # Use a short target name to keep paths within Rich's default terminal width.
    target = tmp_path / "p"
    res = runner.invoke(app, ["init", str(target)])
    out = res.stdout
    # Table title (line 238: Table(title="lighttrain init")).
    assert "lighttrain init" in out
    # At least four 'created' status cells (lines 241-244).
    assert out.count("created") >= 4
    # Final message (line 246).
    assert "initialized" in out.lower()


def test_invariant_init_output_mentions_target_path(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The 'initialized' message must include the resolved target path.

    Covers the f-string on line 246: ``f"initialized lighttrain project at {path}"``
    """
    target = tmp_path / "my_proj"
    res = runner.invoke(app, ["init", str(target)])
    resolved = str(target.resolve())
    assert resolved in res.stdout


def test_invariant_init_creates_parent_dirs(runner: CliRunner, tmp_path: Path) -> None:
    """init creates intermediate parents (``path.mkdir(parents=True, ...)``, line 232).

    If parents are missing, the command must still succeed (exit 0, directory created).
    """
    target = tmp_path / "a" / "b" / "c" / "project"
    res = runner.invoke(app, ["init", str(target)])
    assert res.exit_code == 0, res.stdout
    assert target.is_dir()


# ---------------------------------------------------------------------------
# Error path: non-empty target without --force
# ---------------------------------------------------------------------------


def test_invariant_init_nonempty_dir_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``init`` on a non-empty dir without ``--force`` must exit 1.

    Covers lines 229-231: the guard ``if path.exists() and any(path.iterdir()) and not force``.
    """
    target = tmp_path / "existing"
    target.mkdir()
    (target / "some_file.txt").write_text("occupied", encoding="utf-8")

    res = runner.invoke(app, ["init", str(target)])
    assert res.exit_code == 1, res.stdout


def test_invariant_init_nonempty_error_message(runner: CliRunner, tmp_path: Path) -> None:
    """The error message for a non-empty dir must mention 'not empty' or '--force'.

    Covers line 230: ``console.print(f"[red]target {path} is not empty ...")``.
    """
    target = tmp_path / "existing"
    target.mkdir()
    (target / "blocker.txt").write_text("x", encoding="utf-8")

    res = runner.invoke(app, ["init", str(target)])
    assert "not empty" in res.stdout or "force" in res.stdout.lower()


def test_invariant_init_empty_existing_dir_allowed(runner: CliRunner, tmp_path: Path) -> None:
    """An existing but EMPTY directory is fine — the guard uses ``any(iterdir())``.

    Covers the negative branch: exists() is True but ``any(path.iterdir())`` is False.
    """
    target = tmp_path / "empty_existing"
    target.mkdir()

    res = runner.invoke(app, ["init", str(target)])
    assert res.exit_code == 0, res.stdout
    assert (target / "cfg.yaml").exists()


# ---------------------------------------------------------------------------
# --force flag: overrides non-empty guard
# ---------------------------------------------------------------------------


def test_invariant_init_force_flag_overwrites_nonempty_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``init --force`` on a non-empty dir must succeed (exit 0) and write files.

    Covers the ``force=True`` branch: the guard is bypassed, line 232 runs.
    """
    target = tmp_path / "existing"
    target.mkdir()
    (target / "old.txt").write_text("stale content", encoding="utf-8")

    res = runner.invoke(app, ["init", "--force", str(target)])
    assert res.exit_code == 0, res.stdout
    assert (target / "cfg.yaml").exists()
    assert (target / "README.md").exists()


def test_invariant_init_force_creates_dirs_under_nonempty(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--force on a non-empty dir also creates runs/ and artifacts/.

    Guards that the full scaffold body runs when force bypasses the guard.
    """
    target = tmp_path / "existing"
    target.mkdir()
    (target / "noise.py").write_text("# noise", encoding="utf-8")

    runner.invoke(app, ["init", "--force", str(target)])
    assert (target / "runs").is_dir()
    assert (target / "artifacts").is_dir()


# ---------------------------------------------------------------------------
# Idempotency: running init twice on the same dir with --force
# ---------------------------------------------------------------------------


def test_invariant_init_force_idempotent(runner: CliRunner, tmp_path: Path) -> None:
    """Running init twice with --force on the same dir must both exit 0.

    Exercises ``exist_ok=True`` on mkdir calls (lines 232, 235, 236).
    """
    target = tmp_path / "proj"
    res1 = runner.invoke(app, ["init", str(target)])
    res2 = runner.invoke(app, ["init", "--force", str(target)])
    assert res1.exit_code == 0, res1.stdout
    assert res2.exit_code == 0, res2.stdout


# ---------------------------------------------------------------------------
# Table column presence (lines 239-244)
# ---------------------------------------------------------------------------


def test_invariant_init_table_columns_present(runner: CliRunner, tmp_path: Path) -> None:
    """The Rich table header row must contain 'file' and 'status' columns.

    Covers lines 239-240: ``table.add_column("file", ...)`` and
    ``table.add_column("status", ...)``.
    """
    target = tmp_path / "proj"
    res = runner.invoke(app, ["init", str(target)])
    out = res.stdout
    assert "file" in out.lower()
    assert "status" in out.lower() or "created" in out.lower()


def test_invariant_init_table_row_statuses(runner: CliRunner, tmp_path: Path) -> None:
    """Every table row must have 'created' as its status cell.

    Covers lines 241-244: all four ``table.add_row`` calls use 'created'.
    """
    target = tmp_path / "proj"
    res = runner.invoke(app, ["init", str(target)])
    # Four rows, each with 'created'.
    assert res.stdout.count("created") >= 4
