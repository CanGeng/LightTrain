"""``lighttrain doctor --run <dir>`` health check (REVIEW #14, extended M4).

M3 scope: checkpoint inventory + lineage node/edge sanity. M4 adds real
scans for frozen step bundles, NaN repros, crash bundles, and callback
failures (replacing the M3 ``N/A (M4)`` placeholders).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.engine.checkpoint.manager import CheckpointManager
from lighttrain.observability.lineage.store import LineageStore


def _make_min_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    return run


def test_doctor_clean_run_returns_zero(tmp_path):
    run = _make_min_run(tmp_path)
    # one checkpoint
    mgr = CheckpointManager(run)
    mgr.save(step=1, state={"trainer": {"step": 1}}, kind="step")
    # clean lineage
    ls = LineageStore(run / "lineage.sqlite")
    ls.upsert_node(
        kind="run",
        name="r1",
        version="r1",
        schema_kind="run_meta",
        schema_version="0.4",
    )
    ls.close()

    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    assert res.exit_code == 0, res.stdout
    assert "checkpoints" in res.stdout
    assert "lineage" in res.stdout
    # M4: the deferral markers (N/A (M4)) were replaced with real scans.
    # On a clean run the four new lines all report green/empty.
    assert "frozen_steps" in res.stdout
    assert "NaN repros" in res.stdout
    assert "crash bundles" in res.stdout
    assert "callback report" in res.stdout


def test_doctor_flags_schema_drift(tmp_path):
    run = _make_min_run(tmp_path)
    ls = LineageStore(run / "lineage.sqlite")
    ls.upsert_node(
        kind="run",
        name="r1",
        version="r1",
        schema_kind="run_meta",
        schema_version="0.1",  # stale on purpose
    )
    ls.close()

    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    assert res.exit_code == 2, res.stdout
    assert "schemas" in res.stdout
    assert "lag" in res.stdout or "migrate" in res.stdout


def test_doctor_rejects_nonexistent_path(tmp_path):
    res = CliRunner().invoke(app, ["doctor", "--run", str(tmp_path / "nope")])
    assert res.exit_code == 1
