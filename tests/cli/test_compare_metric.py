"""`lighttrain compare --metric --output` and the Markdown/record renderers (B3)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.lab.compare import compare, render_markdown, to_records

runner = CliRunner()


def _make_run(base: Path, name: str, lr: float, loss: float) -> Path:
    run = base / name
    (run / "logs").mkdir(parents=True)
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump({"optim": {"lr": lr}, "model": "default"}), encoding="utf-8"
    )
    (run / "logs" / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": loss}) + "\n", encoding="utf-8"
    )
    return run


def test_render_markdown_sweep_table(tmp_path):
    r1 = _make_run(tmp_path, "run_a", lr=1e-4, loss=2.5)
    r2 = _make_run(tmp_path, "run_b", lr=3e-4, loss=2.1)
    report = compare([r1, r2])
    md = render_markdown(report, metrics=["loss"])
    lines = md.splitlines()
    assert lines[0].startswith("| run ") and "loss" in lines[0]
    assert lines[1].count("---") >= 2  # header separator row
    assert "run_a" in md and "run_b" in md
    assert "2.5" in md and "2.1" in md


def test_to_records_filters_metric(tmp_path):
    r1 = _make_run(tmp_path, "run_a", lr=1e-4, loss=2.5)
    r2 = _make_run(tmp_path, "run_b", lr=3e-4, loss=2.1)
    recs = to_records(compare([r1, r2]), metrics=["loss"])
    assert {r["run"] for r in recs} == {"run_a", "run_b"}
    assert all("loss" in r for r in recs)


def test_cli_compare_metric_output_md(tmp_path):
    r1 = _make_run(tmp_path, "run_a", lr=1e-4, loss=2.5)
    r2 = _make_run(tmp_path, "run_b", lr=3e-4, loss=2.1)
    out = tmp_path / "table.md"
    res = runner.invoke(
        app, ["compare", str(r1), str(r2), "--metric", "loss", "--output", str(out)]
    )
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert text.startswith("| run ") and "loss" in text


def test_cli_compare_metric_output_json(tmp_path):
    r1 = _make_run(tmp_path, "run_a", lr=1e-4, loss=2.5)
    r2 = _make_run(tmp_path, "run_b", lr=3e-4, loss=2.1)
    out = tmp_path / "table.json"
    res = runner.invoke(
        app, ["compare", str(r1), str(r2), "--metric", "loss", "--output", str(out)]
    )
    assert res.exit_code == 0, res.output
    recs = json.loads(out.read_text())
    assert len(recs) == 2 and all("loss" in r for r in recs)


def test_cli_compare_unknown_metric_warns(tmp_path):
    r1 = _make_run(tmp_path, "run_a", lr=1e-4, loss=2.5)
    r2 = _make_run(tmp_path, "run_b", lr=3e-4, loss=2.1)
    res = runner.invoke(app, ["compare", str(r1), str(r2), "--metric", "ghost"])
    assert res.exit_code == 0, res.output
    assert "no such metric" in res.output.lower()
