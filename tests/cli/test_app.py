"""Adversarial tests for the Typer CLI entry points (``lighttrain.cli._app``).

These tests exercise error paths and config-print-only paths that do NOT
require building a real model or running training. Subjects:

* ``train --config <missing>`` → exit code 1, stderr names the file
* ``train --apply-degrade <missing>`` → exit code 1
* ``train --apply-degrade <invalid YAML>`` → exit code 1
* ``train --print-config`` returns the resolved YAML and exits 0 (no train)
* ``train --mode`` mutates resolved cfg before print
* ``dry-run`` with bad override → exit code 1
* ``--version`` flag exits 0 with version string

The CliRunner from ``typer.testing`` is the same harness as the legacy
``tests/test_cli_freeze_replay.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain import __version__
from lighttrain.cli._app import app


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test."""
    return CliRunner()


def _write_minimal_recipe(tmp_path: Path) -> Path:
    """A recipe small enough to load + validate without touching torch."""
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    return cfg


def test_train_with_missing_config_path_exits_nonzero(runner, tmp_path):
    """``train -c /no/such/file.yaml`` exits non-zero with a clear message.

    Goal: pin error path when the user typos the config path.
    Expected: exit code != 0, output mentions ``config error`` or the missing path.
    """
    missing = tmp_path / "nope.yaml"
    res = runner.invoke(app, ["train", "-c", str(missing)])
    assert res.exit_code != 0
    assert "config error" in res.stdout.lower() or "not found" in res.stdout.lower()


def test_train_with_apply_degrade_missing_file_exits_one(runner, tmp_path):
    """``--apply-degrade /no/such/file`` exits with code 1 and names the file.

    Setup: a valid config + an apply-degrade arg pointing nowhere.
    Expected: exit code 1, output names the missing patch path.
    """
    cfg = _write_minimal_recipe(tmp_path)
    missing_patch = tmp_path / "nope_patch.yaml"
    res = runner.invoke(
        app, ["train", "-c", str(cfg), "--apply-degrade", str(missing_patch)]
    )
    assert res.exit_code == 1
    assert "patch not found" in res.stdout.lower() or "not found" in res.stdout.lower()


def test_train_with_apply_degrade_invalid_yaml_exits_one(runner, tmp_path):
    """``--apply-degrade <file with broken YAML>`` exits with code 1.

    Setup: write a deliberately malformed YAML file (unclosed bracket).
    Expected: exit code 1, output mentions "invalid patch yaml".
    """
    cfg = _write_minimal_recipe(tmp_path)
    bad_patch = tmp_path / "broken.yaml"
    bad_patch.write_text("[unclosed_list\n", encoding="utf-8")
    res = runner.invoke(
        app, ["train", "-c", str(cfg), "--apply-degrade", str(bad_patch)]
    )
    assert res.exit_code == 1
    assert "invalid patch yaml" in res.stdout.lower()


def test_train_print_config_prints_resolved_yaml_and_exits_zero(runner, tmp_path):
    """``--print-config`` prints the resolved YAML and returns 0 without training.

    Setup: valid recipe with mode=lab, seed=7.
    Expected: exit 0, stdout contains ``mode: lab`` and ``seed: 7``.
    """
    cfg = _write_minimal_recipe(tmp_path)
    res = runner.invoke(app, ["train", "-c", str(cfg), "--print-config"])
    assert res.exit_code == 0, res.stdout
    assert "mode: lab" in res.stdout
    assert "seed: 7" in res.stdout


def test_train_print_config_with_mode_override_uses_cli_mode(runner, tmp_path):
    """``--mode prod --print-config`` mutates cfg.mode BEFORE dump.

    Setup: recipe with ``mode: lab``, CLI passes ``--mode prod``.
    Expected: resolved YAML in stdout contains ``mode: prod`` (CLI wins).

    Pin: line 149-150 of _app.py — ``cfg.mode = mode`` is applied after load,
    before dump_resolved. If you reorder these, this test fails.
    """
    cfg = _write_minimal_recipe(tmp_path)
    res = runner.invoke(
        app, ["train", "-c", str(cfg), "--mode", "prod", "--print-config"]
    )
    assert res.exit_code == 0, res.stdout
    assert "mode: prod" in res.stdout


def test_train_print_config_with_invalid_yaml_in_recipe_exits_one(runner, tmp_path):
    """Recipe with invalid mode → ``--print-config`` still fails at load time.

    Setup: recipe with ``mode: bogus`` (rejected by RootConfig.mode Literal).
    Expected: exit code 1, output contains "config error".
    """
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("mode: bogus\n", encoding="utf-8")
    res = runner.invoke(app, ["train", "-c", str(cfg), "--print-config"])
    assert res.exit_code == 1
    assert "config error" in res.stdout.lower()


def test_dry_run_with_invalid_override_exits_one(runner, tmp_path):
    """``dry-run`` with a malformed override (no ``=``) exits 1.

    Setup: valid recipe, override ``malformed_no_eq`` (no equals).
    Expected: exit code 1, output mentions "config error".
    """
    cfg = _write_minimal_recipe(tmp_path)
    res = runner.invoke(app, ["dry-run", "-c", str(cfg), "malformed_no_eq"])
    assert res.exit_code == 1
    assert "config error" in res.stdout.lower()


def test_dry_run_prints_resolved_yaml(runner, tmp_path):
    """``dry-run`` returns the resolved cfg as YAML.

    Setup: valid recipe with mode=lab, seed=99.
    Expected: exit 0, stdout contains both keys.
    """
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("mode: lab\nseed: 99\n", encoding="utf-8")
    res = runner.invoke(app, ["dry-run", "-c", str(cfg)])
    assert res.exit_code == 0, res.stdout
    assert "mode: lab" in res.stdout
    assert "seed: 99" in res.stdout


def test_dry_run_with_overrides_applied_to_output(runner, tmp_path):
    """``dry-run`` applies overrides before printing.

    Setup: recipe with seed=1, CLI override ``++seed=42``.
    Expected: stdout contains ``seed: 42`` (override won).
    """
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("mode: lab\nseed: 1\n", encoding="utf-8")
    res = runner.invoke(app, ["dry-run", "-c", str(cfg), "++seed=42"])
    assert res.exit_code == 0, res.stdout
    assert "seed: 42" in res.stdout


def test_version_flag_prints_version_and_exits_zero(runner):
    """``--version`` exits 0 and prints the version string.

    Goal: pin the eager ``--version`` callback (line 76-79 of _app.py).
    Expected: exit 0, stdout contains the literal ``lighttrain <VERSION>``.
    """
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0, res.stdout
    assert __version__ in res.stdout
    assert "lighttrain" in res.stdout.lower()


def test_no_args_shows_help_and_exits_zero(runner):
    """Invoking with no args triggers Typer's ``no_args_is_help=True``.

    Goal: pin the help-on-no-args behavior. The exit code may be 0 or 2
    depending on Typer's behavior — either way, the output must mention "train".
    """
    res = runner.invoke(app, [])
    # no_args_is_help=True yields Typer's help with exit code 0 or 2.
    assert res.exit_code in (0, 2)
    assert "train" in res.stdout.lower() or "help" in res.stdout.lower()


def test_regression_CLI_MODE_01_invalid_mode_override_rejected(runner, tmp_path):
    """Regression (v0.1.10, breaking): ``--mode bogus`` is now rejected instead
    of silently bypassing the ``Literal["lab", "prod"]`` schema.

    Pre-fix bug: ``cfg.mode = mode`` assigned the CLI value directly onto an
    already-validated RootConfig, which Pydantic does NOT re-check
    (``validate_assignment`` is off). ``--mode bogus`` was accepted with exit 0
    and dumped ``mode: bogus`` — an invalid snapshot that fails its own schema
    on reload/resume. The "non-canonical modes for research" rationale was
    hollow: every consumer branches on ``mode == "lab"`` vs else, so a bogus
    mode behaves identically to ``prod`` — no capability was preserved.

    Setup: valid recipe (mode=lab), CLI passes ``--mode bogus --print-config``.
    Closed form: exit 1, stdout names the bad mode and the allowed set; the
    dump's ``mode: bogus`` never appears.

    Valid modes (``--mode prod`` etc.) still apply — see
    ``test_train_print_config_with_mode_override_uses_cli_mode``.
    """
    cfg = _write_minimal_recipe(tmp_path)
    res = runner.invoke(
        app, ["train", "-c", str(cfg), "--mode", "bogus", "--print-config"]
    )
    assert res.exit_code == 1, res.stdout
    assert "bogus" in res.stdout
    assert "lab" in res.stdout and "prod" in res.stdout
    assert "mode: bogus" not in res.stdout


# ===========================================================================
# Issue #10 — `lighttrain eval` CLI must not crash with TypeError
# ===========================================================================


def test_eval_cmd_does_not_crash_on_config_path(runner, tmp_path):
    """Goal (Issue #10): ``lighttrain eval -c <path>`` must NOT raise the
    pre-fix ``TypeError: argument should be a str or an os.PathLike object ...
    not 'RootConfig'`` (or a ``cannot unpack non-iterable dict`` TypeError
    from the wrong bundle unpacking).

    The minimal recipe lacks the model/data/optimizer fields needed to
    build a trainer, so ``setup_run_from_config`` will still fail downstream
    — but that failure mode is different from the regression we guard.
    """
    cfg = _write_minimal_recipe(tmp_path)
    res = runner.invoke(app, ["eval", "-c", str(cfg)])
    combined = (res.stdout or "") + (str(res.exception) if res.exception else "")
    # Pre-fix the eval command crashed with one of these signatures. Any
    # appearance of them means the regression is back.
    assert "argument should be a str" not in combined
    assert "not 'RootConfig'" not in combined
    assert "cannot unpack non-iterable dict" not in combined
