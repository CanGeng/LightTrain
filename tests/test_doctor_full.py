"""Doctor extended scans (M4 — Phase I3)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lighttrain.cli._app import app


def test_doctor_lists_frozen_steps_and_repros(tmp_path):
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "frozen_steps").mkdir()
    (run / "frozen_steps" / "step_5_scheduled.zip").write_text("x", encoding="utf-8")
    (run / "frozen_steps" / "step_10_scheduled.zip").write_text("y", encoding="utf-8")
    (run / "diagnostics").mkdir()
    (run / "diagnostics" / "repro_nan_123").mkdir()
    (run / "diagnostics" / "callback_failures.jsonl").write_text(
        '{"step":1,"callback":"X","event":"on_step_end","exc_type":"R","traceback":"x"}\n',
        encoding="utf-8",
    )

    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    # Two NaN repros + one crash bundle missing → repro should bump problems.
    assert "frozen_steps" in res.stdout
    assert "n=2" in res.stdout
    assert "NaN repros" in res.stdout
    assert "1 isolated failure" in res.stdout or "callback report" in res.stdout
    # NaN repro presence is a "problem" exit (DESIGN §18.3).
    assert res.exit_code == 2


def test_doctor_clean_after_m4_run_zero(tmp_path):
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "frozen_steps").mkdir()
    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    assert res.exit_code == 0
    assert "frozen_steps" in res.stdout and "n=0" in res.stdout
