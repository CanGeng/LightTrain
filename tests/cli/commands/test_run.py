"""Coverage tests for ``lighttrain.cli.commands.run``.

Targets the branches that were uncovered at 58 % coverage:

* ``train --apply-degrade`` success path (lines 69-70: overrides extended, console print)
* ``_last_checkpoint()`` inner function variants:
    - ckpt_manager is None → returns None (line 115)
    - ckpt_manager.latest() raises → warning + returns None (lines 119-125)
* ``trainer.fit()`` raises → fit_error captured (lines 132-138)
* eval_json write path (lines 151-155)
* fit_error exit(1) path (lines 181-182)
* ``resume_cmd`` validation:
    - bad --mode (lines 199-201)
    - --mode exact fallback message (lines 202-203)
    - explicit --config missing (lines 206-208)
    - default config.snapshot.yaml missing (lines 205-208)
    - setup_run_from_config raises ConfigError (lines 210-217)
    - no resumable checkpoint (lines 220-223)
    - successful resume happy path (lines 224-231)
* ``resume_verify_cmd``:
    - config error exits 1 (lines 264-266)
    - PASS path exits 0 (lines 268-269)
    - FAIL path exits 1 (line 270)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import lighttrain.cli.commands.run as _run_mod
from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Shared runner + minimal recipe helpers
# ---------------------------------------------------------------------------

_runner = CliRunner()


def _minimal_recipe(tmp_path: Path) -> Path:
    """Write the smallest valid recipe that load_config accepts."""
    p = tmp_path / "recipe.yaml"
    p.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    return p


def _stub_bundle(tmp_path: Path, *, fit_raises: BaseException | None = None) -> dict:
    """Return a minimal bundle dict whose trainer.fit() does nothing (or raises)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = MagicMock()
    if fit_raises is not None:
        trainer.fit.side_effect = fit_raises
    else:
        trainer.fit.return_value = None
    # ckpt_manager that returns None for latest()
    trainer.ckpt_manager = MagicMock()
    trainer.ckpt_manager.latest.return_value = None

    cfg_obj = SimpleNamespace(exp="testexp")

    return {
        "run_dir": run_dir,
        "trainer": trainer,
        "cfg": cfg_obj,
        "logger": None,
    }


# ===========================================================================
# train --apply-degrade success path (lines 69-70)
# ===========================================================================


def test_train_apply_degrade_success_extends_overrides_and_prints(tmp_path: Path):
    """When ``--apply-degrade`` points at a valid YAML, the command should:
    1. Extend overrides with the flattened patch entries (line 69).
    2. Print the "applying degrade patch" banner (line 70).
    Then proceed; we monkeypatch setup_run_from_config to short-circuit training.
    """
    cfg = _minimal_recipe(tmp_path)
    patch_file = tmp_path / "degrade.yaml"
    patch_file.write_text("trainer:\n  lr: 1.0e-5\n", encoding="utf-8")

    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(
            app,
            ["train", "-c", str(cfg), "--apply-degrade", str(patch_file)],
        )

    assert res.exit_code == 0, res.output
    assert "applying degrade patch" in res.output.lower()
    # The banner must mention the patch file name.
    assert patch_file.name in res.output


# ===========================================================================
# _last_checkpoint() inner function variants (lines 115, 119-125)
# ===========================================================================


def test_train_last_checkpoint_no_ckpt_manager(tmp_path: Path):
    """If ``trainer.ckpt_manager`` is absent (line 115), ``_last_checkpoint``
    returns None → summary row records ``last_checkpoint: None``.
    """
    cfg = _minimal_recipe(tmp_path)
    summary = tmp_path / "summary.json"

    bundle = _stub_bundle(tmp_path)
    # Remove ckpt_manager attribute entirely.
    del bundle["trainer"].ckpt_manager

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(
            app,
            ["train", "-c", str(cfg), "--output-summary", str(summary)],
        )

    assert res.exit_code == 0, res.output
    rows = json.loads(summary.read_text())
    assert rows[0]["last_checkpoint"] is None


def test_train_last_checkpoint_ckpt_manager_latest_raises(tmp_path: Path):
    """If ``trainer.ckpt_manager.latest()`` raises (lines 119-125), the inner
    helper catches the exception, logs a warning, and records ``last_checkpoint: None``.
    """
    cfg = _minimal_recipe(tmp_path)
    summary = tmp_path / "summary.json"

    bundle = _stub_bundle(tmp_path)
    bundle["trainer"].ckpt_manager.latest.side_effect = RuntimeError("disk error")

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(
            app,
            ["train", "-c", str(cfg), "--output-summary", str(summary)],
        )

    assert res.exit_code == 0, res.output
    rows = json.loads(summary.read_text())
    assert rows[0]["last_checkpoint"] is None


# ===========================================================================
# trainer.fit() raises → fit_error captured (lines 132-138) and exit 1 (181-182)
# ===========================================================================


def test_train_fit_raises_exits_one(tmp_path: Path):
    """When ``trainer.fit()`` raises, the CLI must:
    * Capture the exception as fit_error (lines 132-138).
    * Still close the logger if present.
    * Print a "training failed" message (line 181).
    * Exit with code 1 (line 182).
    """
    cfg = _minimal_recipe(tmp_path)

    logger_mock = MagicMock()
    bundle = _stub_bundle(tmp_path, fit_raises=RuntimeError("boom"))
    bundle["logger"] = logger_mock

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(app, ["train", "-c", str(cfg)])

    assert res.exit_code == 1, res.output
    assert "training failed" in res.output.lower()
    assert "runtimeerror" in res.output.lower() or "boom" in res.output.lower()
    logger_mock.close.assert_called_once()


def test_train_fit_raises_still_writes_summary_with_error_status(tmp_path: Path):
    """Even when fit raises, if ``--output-summary`` is given the row is written
    with ``status=error`` (the finally-guarded summary block).
    """
    cfg = _minimal_recipe(tmp_path)
    summary = tmp_path / "summary.json"

    bundle = _stub_bundle(tmp_path, fit_raises=ValueError("loss exploded"))

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(
            app,
            ["train", "-c", str(cfg), "--output-summary", str(summary)],
        )

    assert res.exit_code == 1, res.output
    rows = json.loads(summary.read_text())
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] is not None
    assert "ValueError" in rows[0]["error"]


# ===========================================================================
# eval_json write path (lines 151-155)
# ===========================================================================


def test_train_eval_json_written_on_success(tmp_path: Path):
    """``--eval --eval-json`` must write a JSON file with perplexity metrics
    (lines 151-155). We monkeypatch ``_eval_perplexity`` to avoid a real
    model eval.
    """
    cfg = _minimal_recipe(tmp_path)
    eval_json = tmp_path / "sub" / "eval.json"

    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle), \
         patch.object(_run_mod, "_eval_perplexity", return_value=2.71):
        res = _runner.invoke(
            app,
            [
                "train", "-c", str(cfg),
                "--eval",
                "--eval-json", str(eval_json),
            ],
        )

    assert res.exit_code == 0, res.output
    assert eval_json.exists(), "eval JSON file should have been created"
    payload = json.loads(eval_json.read_text())
    assert payload["task_name"] == "train_eval"
    assert payload["metrics"]["perplexity"] == pytest.approx(2.71)
    assert "timestamp" in payload


def test_train_eval_json_not_written_when_fit_fails(tmp_path: Path):
    """If ``trainer.fit()`` raises, the eval block is skipped (``eval and fit_error is None``
    is False), so the eval JSON must NOT be written.
    """
    cfg = _minimal_recipe(tmp_path)
    eval_json = tmp_path / "eval.json"

    bundle = _stub_bundle(tmp_path, fit_raises=RuntimeError("nope"))

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle), \
         patch.object(_run_mod, "_eval_perplexity", return_value=1.5) as mock_eval:
        res = _runner.invoke(
            app,
            [
                "train", "-c", str(cfg),
                "--eval",
                "--eval-json", str(eval_json),
            ],
        )

    assert res.exit_code == 1
    assert not eval_json.exists()
    mock_eval.assert_not_called()


# ===========================================================================
# resume_cmd validation — bad mode (lines 199-201)
# ===========================================================================


def test_resume_bad_mode_exits_one(tmp_path: Path):
    """``resume --mode unknown`` must print an error and exit 1 (lines 199-201)."""
    res = _runner.invoke(app, ["resume", "--run", str(tmp_path), "--mode", "unknown"])
    assert res.exit_code == 1
    assert "unknown" in res.output


def test_invariant_resume_bad_mode_message_mentions_valid_modes(tmp_path: Path):
    """The bad-mode error message must name the valid modes ``functional|exact``
    so users know what to pass.
    """
    res = _runner.invoke(app, ["resume", "--run", str(tmp_path), "--mode", "invalid"])
    assert "functional" in res.output
    assert "exact" in res.output


# ===========================================================================
# resume_cmd — exact mode fallback message (lines 202-203)
# ===========================================================================


def test_resume_exact_mode_prints_fallback_warning(tmp_path: Path, monkeypatch):
    """``--mode exact`` must print the fallback-to-functional warning (line 203)
    and then proceed; we monkeypatch setup_run_from_config to prevent real work.
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    # Make setup_run_from_config return a stub bundle with a checkpoint.
    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = Path("/fake/ckpt")
    trainer_stub.fit.return_value = None
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": None}

    monkeypatch.setattr(_run_mod, "setup_run_from_config", lambda *a, **kw: bundle)

    res = _runner.invoke(app, ["resume", "--run", str(tmp_path), "--mode", "exact"])

    assert "exact resume is not yet implemented" in res.output.lower() or \
           "not yet implemented" in res.output.lower() or \
           "falling back to functional" in res.output.lower()


# ===========================================================================
# resume_cmd — missing config file paths (lines 205-208)
# ===========================================================================


def test_resume_missing_default_snapshot_exits_one(tmp_path: Path):
    """When the default ``run_dir/config.snapshot.yaml`` does not exist, ``resume``
    should print a "no recipe found" error and exit 1 (lines 206-208).
    """
    # tmp_path has no config.snapshot.yaml
    res = _runner.invoke(app, ["resume", "--run", str(tmp_path)])
    assert res.exit_code == 1
    assert "no recipe found" in res.output.lower()


def test_resume_explicit_config_missing_exits_one(tmp_path: Path):
    """When ``-c <path>`` is given but the file is absent, exit 1 with error."""
    missing = tmp_path / "nope.yaml"
    res = _runner.invoke(
        app, ["resume", "--run", str(tmp_path), "-c", str(missing)]
    )
    assert res.exit_code == 1
    assert "no recipe found" in res.output.lower()


# ===========================================================================
# resume_cmd — setup_run_from_config raises (lines 210-217)
# ===========================================================================


def test_resume_setup_run_config_error_exits_one(tmp_path: Path, monkeypatch):
    """If ``setup_run_from_config`` raises ``ConfigError`` during resume, the
    CLI prints a "config error" message and exits 1 (lines 215-217).
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    from lighttrain.config import ConfigError

    monkeypatch.setattr(
        _run_mod, "setup_run_from_config", MagicMock(side_effect=ConfigError("bad"))
    )

    res = _runner.invoke(app, ["resume", "--run", str(tmp_path)])
    assert res.exit_code == 1
    assert "config error" in res.output.lower()


# ===========================================================================
# resume_cmd — no checkpoint under run dir (lines 220-223)
# ===========================================================================


def test_resume_no_checkpoint_exits_one(tmp_path: Path, monkeypatch):
    """When ``trainer.ckpt_manager.latest()`` returns None after setup,
    ``resume`` prints an error and exits 1 (lines 220-223).
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = None
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": None}

    monkeypatch.setattr(_run_mod, "setup_run_from_config", lambda *a, **kw: bundle)

    res = _runner.invoke(app, ["resume", "--run", str(tmp_path)])
    assert res.exit_code == 1
    assert "no resumable checkpoint" in res.output.lower()


# ===========================================================================
# resume_cmd — happy path (lines 219, 224-231)
# ===========================================================================


def test_resume_happy_path_exits_zero(tmp_path: Path, monkeypatch):
    """When setup and checkpoint load succeed, ``resume`` runs fit() and exits 0
    (lines 219, 224-231).
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    ckpt_path = Path("/fake/ckpt/step_5")
    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = ckpt_path
    trainer_stub.fit.return_value = None
    logger_mock = MagicMock()
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": logger_mock}

    monkeypatch.setattr(_run_mod, "setup_run_from_config", lambda *a, **kw: bundle)

    res = _runner.invoke(app, ["resume", "--run", str(tmp_path)])

    assert res.exit_code == 0, res.output
    assert "resumed" in res.output.lower()
    assert "resume complete" in res.output.lower()
    trainer_stub.load_checkpoint.assert_called_once_with(ckpt_path)
    trainer_stub.fit.assert_called_once()
    logger_mock.close.assert_called_once()


def test_resume_happy_path_no_logger_still_exits_zero(tmp_path: Path, monkeypatch):
    """``resume`` with ``bundle['logger'] is None`` must not crash (lines 229-230
    guard ``if bundle.get('logger') is not None``).
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    ckpt_path = Path("/fake/ckpt/step_1")
    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = ckpt_path
    trainer_stub.fit.return_value = None
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": None}

    monkeypatch.setattr(_run_mod, "setup_run_from_config", lambda *a, **kw: bundle)

    res = _runner.invoke(app, ["resume", "--run", str(tmp_path)])
    assert res.exit_code == 0, res.output


def test_resume_explicit_config_used_when_provided(tmp_path: Path, monkeypatch):
    """When ``-c explicit.yaml`` is provided (line 205), that path is used as
    the recipe instead of the default ``run_dir/config.snapshot.yaml``.
    """
    explicit_cfg = tmp_path / "explicit.yaml"
    explicit_cfg.write_text("mode: lab\nseed: 42\n", encoding="utf-8")

    ckpt_path = Path("/fake/ckpt")
    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = ckpt_path
    trainer_stub.fit.return_value = None
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": None}

    captured_calls: list[dict] = []

    def _fake_setup(cfg_path, **kw):
        captured_calls.append({"cfg_path": cfg_path})
        return bundle

    monkeypatch.setattr(_run_mod, "setup_run_from_config", _fake_setup)

    res = _runner.invoke(
        app, ["resume", "--run", str(tmp_path), "-c", str(explicit_cfg)]
    )
    assert res.exit_code == 0, res.output
    assert captured_calls[0]["cfg_path"] == explicit_cfg


# ===========================================================================
# resume_verify_cmd — config error path (lines 264-266)
# ===========================================================================


def test_resume_verify_missing_config_exits_one(tmp_path: Path):
    """``resume-verify -c /no/such/file`` exits 1 with "config error" (lines 264-266)."""
    missing = tmp_path / "nope.yaml"
    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(missing),
            "--phase1-steps", "2", "--phase2-steps", "2",
        ],
    )
    assert res.exit_code == 1
    assert "config error" in res.output.lower()


def test_resume_verify_config_error_from_resume_verify_fn_exits_one(
    tmp_path: Path, monkeypatch
):
    """If ``resume_verify()`` (the lab function) raises ConfigError, the CLI
    must exit 1 with "config error" (lines 264-266).
    """
    cfg = _minimal_recipe(tmp_path)

    from lighttrain.config import ConfigError

    monkeypatch.setattr(
        "lighttrain.lab.resume_verify.resume_verify",
        MagicMock(side_effect=ConfigError("bad config")),
    )

    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(cfg),
            "--phase1-steps", "2", "--phase2-steps", "2",
        ],
    )
    assert res.exit_code == 1
    assert "config error" in res.output.lower()


# ===========================================================================
# resume_verify_cmd — PASS / FAIL exit codes (lines 268-270)
# ===========================================================================


def _make_fake_report(*, passed: bool):
    """Return a fake ResumeVerifyReport-like object."""
    from lighttrain.lab.resume_verify import ResumeVerifyReport

    return ResumeVerifyReport(
        phase1_steps=2,
        phase2_steps=2,
        tol=1e-2,
        single_pass_losses=[1.0, 0.9, 0.8, 0.7],
        resume_losses=[1.0, 0.9, 0.8, 0.7],
        per_step_delta=[0.0, 0.0, 0.0, 0.0],
        max_abs_delta=0.0,
        passed=passed,
        note="",
    )


def test_resume_verify_pass_exits_zero(tmp_path: Path, monkeypatch):
    """When ``resume_verify`` returns a passing report, CLI exits 0 (line 268-269)."""
    cfg = _minimal_recipe(tmp_path)
    report = _make_fake_report(passed=True)

    monkeypatch.setattr(
        "lighttrain.lab.resume_verify.resume_verify",
        MagicMock(return_value=report),
    )

    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(cfg),
            "--phase1-steps", "2", "--phase2-steps", "2",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "PASS" in res.output


def test_resume_verify_fail_exits_one(tmp_path: Path, monkeypatch):
    """When ``resume_verify`` returns a failing report (``report.passed=False``),
    CLI exits 1 (line 270).
    """
    cfg = _minimal_recipe(tmp_path)
    report = _make_fake_report(passed=False)

    monkeypatch.setattr(
        "lighttrain.lab.resume_verify.resume_verify",
        MagicMock(return_value=report),
    )

    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(cfg),
            "--phase1-steps", "2", "--phase2-steps", "2",
        ],
    )
    assert res.exit_code == 1
    assert "FAIL" in res.output


def test_resume_verify_custom_tol_passed_through(tmp_path: Path, monkeypatch):
    """``--tol`` overrides the default ``DEFAULT_TOL`` (line 261 tol expression)."""
    cfg = _minimal_recipe(tmp_path)
    report = _make_fake_report(passed=True)

    captured: list[dict] = []

    def _fake_rv(config, p1, p2, *, tol, overrides):
        captured.append({"tol": tol})
        return report

    monkeypatch.setattr("lighttrain.lab.resume_verify.resume_verify", _fake_rv)

    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(cfg),
            "--phase1-steps", "2", "--phase2-steps", "2",
            "--tol", "1e-5",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured[0]["tol"] == pytest.approx(1e-5)


def test_resume_verify_overrides_passed_through(tmp_path: Path, monkeypatch):
    """Positional overrides are forwarded to ``resume_verify()`` as the
    ``overrides`` arg.
    """
    cfg = _minimal_recipe(tmp_path)
    report = _make_fake_report(passed=True)

    captured: list[dict] = []

    def _fake_rv(config, p1, p2, *, tol, overrides):
        captured.append({"overrides": overrides})
        return report

    monkeypatch.setattr("lighttrain.lab.resume_verify.resume_verify", _fake_rv)

    res = _runner.invoke(
        app,
        [
            "resume-verify", "-c", str(cfg),
            "--phase1-steps", "2", "--phase2-steps", "2",
            "++seed=42",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "++seed=42" in captured[0]["overrides"]


# ===========================================================================
# train happy path control-flow invariants (via monkeypatching)
# ===========================================================================


def test_train_success_prints_completion(tmp_path: Path):
    """The happy path must print a "training complete" message and exit 0."""
    cfg = _minimal_recipe(tmp_path)
    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(app, ["train", "-c", str(cfg)])

    assert res.exit_code == 0, res.output
    assert "training complete" in res.output.lower()


def test_train_logger_closed_on_success(tmp_path: Path):
    """The logger must be closed in the finally block even on a clean fit."""
    cfg = _minimal_recipe(tmp_path)
    logger_mock = MagicMock()
    bundle = _stub_bundle(tmp_path)
    bundle["logger"] = logger_mock

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(app, ["train", "-c", str(cfg)])

    assert res.exit_code == 0
    logger_mock.close.assert_called_once()


def test_invariant_train_prints_run_dir(tmp_path: Path):
    """The CLI must echo the run_dir to stdout so users can find their artifacts."""
    cfg = _minimal_recipe(tmp_path)
    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle):
        res = _runner.invoke(app, ["train", "-c", str(cfg)])

    assert res.exit_code == 0
    assert str(bundle["run_dir"]) in res.output


def test_train_setup_run_file_not_found_exits_one(tmp_path: Path, monkeypatch):
    """If ``setup_run_from_config`` raises ``FileNotFoundError`` (e.g. data
    file absent), the CLI must exit 1 with a "config error" message.
    """
    cfg = _minimal_recipe(tmp_path)

    monkeypatch.setattr(
        _run_mod,
        "setup_run_from_config",
        MagicMock(side_effect=FileNotFoundError("data.txt not found")),
    )

    res = _runner.invoke(app, ["train", "-c", str(cfg)])
    assert res.exit_code == 1
    assert "config error" in res.output.lower()


def test_pin_current_behavior_eval_ppl_printed_when_not_none(tmp_path: Path):
    """Pin: when ``_eval_perplexity`` returns a value, the CLI prints it to
    stdout. If the perplexity-print line is removed, this test fails.
    """
    cfg = _minimal_recipe(tmp_path)
    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle), \
         patch.object(_run_mod, "_eval_perplexity", return_value=3.14):
        res = _runner.invoke(app, ["train", "-c", str(cfg), "--eval"])

    assert res.exit_code == 0
    assert "3.14" in res.output or "perplexity" in res.output.lower()


def test_pin_current_behavior_eval_ppl_none_no_crash(tmp_path: Path):
    """Pin: when ``_eval_perplexity`` returns None (e.g. no val loader), the
    CLI must not crash — perplexity block is silently skipped.
    """
    cfg = _minimal_recipe(tmp_path)
    bundle = _stub_bundle(tmp_path)

    with patch.object(_run_mod, "setup_run_from_config", return_value=bundle), \
         patch.object(_run_mod, "_eval_perplexity", return_value=None):
        res = _runner.invoke(app, ["train", "-c", str(cfg), "--eval"])

    assert res.exit_code == 0


# ===========================================================================
# resume_cmd — fit() raises in resume flow (logger still closed)
# ===========================================================================


def test_resume_fit_raises_logger_still_closed(tmp_path: Path, monkeypatch):
    """Even if ``trainer.fit()`` raises during resume, the finally block (lines
    228-230) must close the logger.
    """
    cfg = tmp_path / "config.snapshot.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")

    ckpt_path = Path("/fake/ckpt")
    trainer_stub = MagicMock()
    trainer_stub.ckpt_manager.latest.return_value = ckpt_path
    trainer_stub.fit.side_effect = RuntimeError("fit exploded during resume")
    logger_mock = MagicMock()
    bundle = {"run_dir": tmp_path, "trainer": trainer_stub, "logger": logger_mock}

    monkeypatch.setattr(_run_mod, "setup_run_from_config", lambda *a, **kw: bundle)

    _runner.invoke(app, ["resume", "--run", str(tmp_path)])

    # The exception propagates uncaught from resume_cmd, so exit != 0.
    # The logger must still have been closed (finally block).
    logger_mock.close.assert_called_once()
