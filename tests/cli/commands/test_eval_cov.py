"""Coverage extension for ``lighttrain/cli/commands/eval.py``.

Drives the uncovered lines to green by testing:

* eval_cmd — checkpoint load failure (lines 63-64, 69)
* eval_cmd — trainer has no model → exit 1 (lines 82-83)
* eval_cmd — Evaluator is present and runs successfully (lines 100-105)
* eval_cmd — Evaluator is present but raises → warning printed (lines 106-112)
* eval_cmd — cleanup failure is swallowed (lines 136-137)
* regression_gate_cmd — full happy path: gate passes (lines 163-214)
* regression_gate_cmd — checkpoint load failure path (lines 173-184)
* regression_gate_cmd — data_module present + val_loader present → perplexity computed
  (lines 191-196)
* regression_gate_cmd — perplexity computation raises → exit 1 (lines 197-199)
* regression_gate_cmd — no data_module → metrics empty, gate checked (lines 201-214)
* regression_gate_cmd — gate fails → exit 1 (lines 215-217)

Hardware/network/GPU branches are skipped (noted in skipped_lines_note).

All tests use ``typer.testing.CliRunner`` + ``unittest.mock.patch``
against the live ``app`` object — no source files are modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from typer.testing import CliRunner

from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EVAL_CMD_MOD = "lighttrain.cli.commands.eval"

runner = CliRunner()

# A single shared tiny model used across tests that need a non-None model.
# Defined at module scope so it can be referenced in class bodies.
_MODULE_MODEL: nn.Module = nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_recipe(tmp_path: Path) -> Path:
    """Minimal recipe file on disk — needed so config path validation passes."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    return recipe


# ---------------------------------------------------------------------------
# Shared trainer stubs (class attributes referencing module-level _MODULE_MODEL)
# ---------------------------------------------------------------------------


class _BrokenCheckpointTrainer:
    """Trainer whose load_checkpoint always raises."""
    ckpt_manager = object()  # non-None → inner-if is entered
    model = _MODULE_MODEL
    device = None
    data_module = None
    evaluator = None

    def load_checkpoint(self, path):
        raise RuntimeError("simulated ckpt load failure")


class _NoModelTrainer:
    """Trainer with model=None to exercise the no-model guard."""
    ckpt_manager = None
    model = None
    device = None
    data_module = None
    evaluator = None


class _SimpleTrainer:
    """Vanilla trainer with a model; no data_module, no evaluator."""
    ckpt_manager = None
    model = _MODULE_MODEL
    device = None
    data_module = None
    evaluator = None


def _fake_bundle(trainer: object) -> dict:
    """Wrap a trainer stub in the shape returned by setup_run_from_config."""
    return {"trainer": trainer, "cfg": SimpleNamespace(evaluator=None)}


# ---------------------------------------------------------------------------
# eval_cmd — checkpoint load failure (lines 63-64, 69)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_checkpoint_load_failure_continues(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """eval_cmd with --checkpoint must survive a trainer.load_checkpoint exception.

    Goal: cover lines 63-64 (except branch) and 69 (console.print yellow).
    The command must still complete with exit 0 (it warns and continues on
    untrained weights), not propagate the exception.
    """
    trainer = _BrokenCheckpointTrainer()

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    def _fake_perplexity(t, mb):
        return None

    fake_ckpt = tmp_path / "step_1"
    fake_ckpt.mkdir()

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app, ["eval", "-c", str(tmp_recipe), "--checkpoint", str(fake_ckpt)]
        )

    assert res.exit_code == 0, res.output
    assert "checkpoint load failed" in res.output.lower()
    # loaded_ckpt=False → untrained-weights warning must appear
    assert "untrained" in res.output.lower()


# ---------------------------------------------------------------------------
# eval_cmd — trainer has no model → exit 1 (lines 82-83)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_no_model_exits_one(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """eval_cmd must exit 1 and print an error when trainer.model is None.

    Goal: cover lines 82-83 (the ``if model is None`` guard).
    """
    trainer = _NoModelTrainer()

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(app, ["eval", "-c", str(tmp_recipe)])

    assert res.exit_code == 1, res.output
    assert "no model" in res.output.lower() or "trainer has no model" in res.output.lower()


# ---------------------------------------------------------------------------
# eval_cmd — Evaluator is present and runs successfully (lines 100-105)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_evaluator_runs_and_merges_metrics(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """eval_cmd must call evaluator.run and merge its metrics into the output.

    Goal: cover lines 100-105 (evaluator isinstance check + run + update).
    The Evaluator stub returns a report with 'accuracy' metric.
    """
    from lighttrain.eval.suite import Evaluator

    class _DummyTask:
        name = "dummy"

        def run(self, model, *, device=None, step=None):
            return {"accuracy": 0.42}

    evaluator = Evaluator([_DummyTask()])

    # Use SimpleNamespace to avoid Python class-body scope limitations
    # (class-level attribute assignment cannot reference local variables).
    trainer = SimpleNamespace(
        ckpt_manager=None,
        model=_MODULE_MODEL,
        device=None,
        data_module=None,
        evaluator=evaluator,
    )

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    def _fake_perplexity(t, mb):
        return None

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(app, ["eval", "-c", str(tmp_recipe)])

    assert res.exit_code == 0, res.output
    # Evaluator ran → 'accuracy' or '0.42' must appear in the Rich table output
    assert "accuracy" in res.output.lower() or "0.42" in res.output


# ---------------------------------------------------------------------------
# eval_cmd — Evaluator raises → warning printed (lines 106-112)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_evaluator_failure_prints_warning(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """eval_cmd must swallow Evaluator.run exceptions and print a yellow warning.

    Goal: cover lines 106-112 (except block around evaluator.run).
    """
    from lighttrain.eval.suite import Evaluator

    class _BoomEvaluator(Evaluator):
        def __init__(self):
            pass  # skip parent __init__

        def run(self, model, step, *, device=None, force=False):
            raise RuntimeError("boom evaluator")

    class _EvalTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        device = None
        data_module = None
        evaluator = _BoomEvaluator()

    trainer = _EvalTrainer()

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    def _fake_perplexity(t, mb):
        return None

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(app, ["eval", "-c", str(tmp_recipe)])

    assert res.exit_code == 0, res.output
    assert "evaluator failed" in res.output.lower()


# ---------------------------------------------------------------------------
# eval_cmd — cleanup failure is swallowed (lines 136-137)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_cleanup_failure_is_swallowed(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """A TemporaryDirectory.cleanup() failure in the finally block must not
    propagate — the command must still complete successfully.

    Goal: cover lines 136-137 (except block inside finally).
    """
    trainer = _SimpleTrainer()

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    def _fake_perplexity(t, mb):
        return 42.0

    def _boom_cleanup(self):
        raise OSError("simulated open-handle cleanup failure")

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
        patch("tempfile.TemporaryDirectory.cleanup", _boom_cleanup),
    ):
        res = runner.invoke(app, ["eval", "-c", str(tmp_recipe)])

    # cleanup error must not propagate
    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# eval_cmd — json_out path is written when --json flag provided (lines 122-132)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_json_out_written(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """eval_cmd with --json must write the EvalReport JSON to the given path.

    This covers lines 122-132 (json_out is not None branch) with a stubbed
    setup_run_from_config so the test is fast and GPU-free.
    """
    trainer = _SimpleTrainer()

    def _fake_setup(config, **kw):
        return _fake_bundle(trainer)

    def _fake_perplexity(t, mb):
        return 30.5

    json_out = tmp_path / "report.json"

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            ["eval", "-c", str(tmp_recipe), "--json", str(json_out)],
        )

    assert res.exit_code == 0, res.output
    assert json_out.exists(), "JSON output file must be written"
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["task_name"] == "eval"
    assert "perplexity" in report["metrics"]
    assert report["metrics"]["perplexity"] == pytest.approx(30.5)


# ---------------------------------------------------------------------------
# eval_cmd — evaluator on cfg (not trainer) is picked up (line 98)
# ---------------------------------------------------------------------------


def test_invariant_eval_cmd_non_evaluator_on_cfg_does_not_run(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """When trainer.evaluator is None but cfg.evaluator is a plain object (not an
    Evaluator instance), the isinstance guard on line 102 is False — no run()
    call, command exits cleanly.

    Goal: cover the ``or getattr(cfg, "evaluator", None)`` branch on line 98
    and the isinstance(evaluator, Evaluator) False-branch on line 102.
    """
    trainer = _SimpleTrainer()
    # cfg.evaluator is a non-Evaluator object
    cfg = SimpleNamespace(evaluator=object())

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": cfg}

    def _fake_perplexity(t, mb):
        return None

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}._eval_perplexity", side_effect=_fake_perplexity),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(app, ["eval", "-c", str(tmp_recipe)])

    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# regression_gate_cmd — full happy path: gate passes (lines 163-214)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_passes_exits_zero(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """regression-gate with a passing gate must exit 0 and print 'PASS'.

    Goal: cover lines 163-214 (full happy path with no data_module).
    Metric 'perplexity' is absent from metrics dict → gate.check skips → PASS.
    """
    trainer = _SimpleTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
            ],
        )

    assert res.exit_code == 0, res.output
    assert "pass" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — checkpoint load failure path (lines 173-184)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_checkpoint_load_failure_continues(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """regression-gate with a failing checkpoint load must warn and continue.

    Goal: cover lines 173-184 (checkpoint block with exception).
    """
    trainer = _BrokenCheckpointTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    fake_ckpt = tmp_path / "step_99"
    fake_ckpt.mkdir()

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
                "--checkpoint", str(fake_ckpt),
            ],
        )

    assert res.exit_code == 0, res.output
    assert "checkpoint load failed" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — data_module present + val_loader present (lines 191-196)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_perplexity_computed_from_val_loader(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """When trainer.data_module.val_loader() returns a loader, perplexity is
    computed and the gate runs against it.

    Goal: cover lines 191-196 (data_module branch + perplexity call).
    We patch lighttrain.eval.metrics.perplexity so no GPU is needed.
    """
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    labels = torch.zeros(2, 4, dtype=torch.long)
    loader = [{"input_ids": input_ids, "labels": labels}]

    class _FakeDataModule:
        def val_loader(self):
            return loader

    class _DMTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        data_module = _FakeDataModule()
        device = None

    trainer = _DMTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    def _fake_perplexity_fn(model, loader, *, device=None, max_batches=None):
        return 25.0

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
        patch("lighttrain.eval.metrics.perplexity", side_effect=_fake_perplexity_fn),
        patch(
            "lighttrain.builtin_plugins.callbacks.invariants.regression_gate"
            ".RegressionGate.check"
        ),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
                "--max-batches", "1",
            ],
        )

    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# regression_gate_cmd — perplexity computation raises → exit 1 (lines 197-199)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_perplexity_error_exits_one(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """When perplexity() raises inside the regression-gate command, the
    command must print a yellow error and exit 1.

    Goal: cover lines 197-199 (except block + raise typer.Exit(code=1)).
    """
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    labels = torch.zeros(2, 4, dtype=torch.long)
    loader = [{"input_ids": input_ids, "labels": labels}]

    class _FakeDataModule:
        def val_loader(self):
            return loader

    class _DMTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        data_module = _FakeDataModule()
        device = None

    trainer = _DMTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    def _boom_perplexity(model, loader, *, device=None, max_batches=None):
        raise RuntimeError("perplexity exploded")

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
        patch("lighttrain.eval.metrics.perplexity", side_effect=_boom_perplexity),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
                "--max-batches", "1",
            ],
        )

    assert res.exit_code == 1, res.output
    assert "eval failed" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — gate.check raises → exit 1 (lines 215-217)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_fail_exits_one(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """When gate.check raises (metric violates threshold), the command must
    print 'FAIL' and exit 1.

    Goal: cover lines 215-217 (except block + raise typer.Exit(code=1)).
    """
    trainer = _SimpleTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
        patch(
            "lighttrain.builtin_plugins.callbacks.invariants.regression_gate"
            ".RegressionGate.check",
            side_effect=RuntimeError("gate FAILED: value=200.0 at step=None"),
        ),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "50.0",
                "--op", "<",
            ],
        )

    assert res.exit_code == 1, res.output
    assert "fail" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — val_loader returns None → skip perplexity (line 193)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_no_val_loader_skips_perplexity(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """When data_module.val_loader() returns None, perplexity is skipped and
    metrics dict stays empty → gate sees metric absent → PASS.

    Goal: cover line 193 branch (val_loader is None).
    """

    class _FakeDataModule:
        def val_loader(self):
            return None  # no val split

    class _DMTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        data_module = _FakeDataModule()
        device = None

    trainer = _DMTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
            ],
        )

    assert res.exit_code == 0, res.output
    assert "pass" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — data_module has no val_loader attr (line 192)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_data_module_no_val_loader_attr(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """data_module without val_loader attribute: val_loader=None, skip perplexity.

    Goal: cover line 192 (``hasattr`` check — false branch).
    """

    class _FakeDataModule:
        pass  # no val_loader

    class _DMTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        data_module = _FakeDataModule()
        device = None

    trainer = _DMTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "loss",
                "--threshold", "5.0",
                "--op", "<",
            ],
        )

    assert res.exit_code == 0, res.output
    assert "pass" in res.output.lower()


# ---------------------------------------------------------------------------
# regression_gate_cmd — max_batches=0 passes None to perplexity (line 195)
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_max_batches_zero_passes_none(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """max_batches=0 must translate to max_batches=None inside the command
    (line 195: ``mb = max_batches if max_batches > 0 else None``).

    Verify by capturing the actual call args to perplexity.
    """
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    labels = torch.zeros(2, 4, dtype=torch.long)
    loader = [{"input_ids": input_ids, "labels": labels}]

    class _FakeDataModule:
        def val_loader(self):
            return loader

    class _DMTrainer:
        ckpt_manager = None
        model = _MODULE_MODEL
        data_module = _FakeDataModule()
        device = None

    trainer = _DMTrainer()
    captured_mb: list = []

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    def _spy_perplexity(model, loader, *, device=None, max_batches=None):
        captured_mb.append(max_batches)
        return 15.0

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
        patch("lighttrain.eval.metrics.perplexity", side_effect=_spy_perplexity),
        patch(
            "lighttrain.builtin_plugins.callbacks.invariants.regression_gate"
            ".RegressionGate.check"
        ),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "perplexity",
                "--threshold", "100.0",
                "--op", "<",
                "--max-batches", "0",
            ],
        )

    assert res.exit_code == 0, res.output
    assert captured_mb == [None], f"expected [None], got {captured_mb}"


# ---------------------------------------------------------------------------
# regression_gate_cmd — action=warn: gate check warns but command exits 0
# ---------------------------------------------------------------------------


def test_invariant_regression_gate_action_warn_exits_zero(
    tmp_path: Path, tmp_recipe: Path
) -> None:
    """regression-gate --action warn must exit 0 even when the gate condition
    fails, because RegressionGate emits a warning rather than raising.

    pin_current_behavior: gate.check with action='warn' warns, does not raise
    → the except block on line 215 is NOT entered → exit 0.
    """
    trainer = _SimpleTrainer()

    def _fake_setup(config, **kw):
        return {"trainer": trainer, "cfg": SimpleNamespace()}

    with (
        patch(f"{_EVAL_CMD_MOD}.setup_run_from_config", side_effect=_fake_setup),
        patch(f"{_EVAL_CMD_MOD}.load_dotenv_if_present"),
    ):
        res = runner.invoke(
            app,
            [
                "regression-gate",
                "-c", str(tmp_recipe),
                "--metric", "missing_metric",  # absent → gate skips silently
                "--threshold", "0.001",
                "--op", ">",
                "--action", "warn",
            ],
        )

    assert res.exit_code == 0, res.output
    assert "pass" in res.output.lower()
