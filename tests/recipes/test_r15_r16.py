"""R15 / R16 acceptance tests — DESIGN §26.10 (M8).

R15: sweep ≥ 8 trials, generates markdown report with top-K + sensitivity.
R16: fork → inject LR scale → fork again; 3-generation lineage complete.

Both tests use a tiny 50-step model to keep CI times short.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lighttrain.lab.auto_report import render_sweep_markdown, write_sweep_report
from lighttrain.lab.compare import compare, render_ascii
from lighttrain.lab.fork import fork
from lighttrain.lab.sweep import SweepRunner
from tests._diagnostics import expect_exists

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_run_dir(parent: Path, name: str, metric_val: float) -> Path:
    rd = parent / name
    (rd / "logs").mkdir(parents=True)
    (rd / "logs" / "metrics.jsonl").write_text(
        json.dumps({"step": 50, "loss": metric_val}) + "\n",
        encoding="utf-8",
    )
    (rd / "env.json").write_text("{}", encoding="utf-8")
    return rd


def _make_ckpt(run_dir: Path, step: int = 50) -> Path:
    ckpt = run_dir / "checkpoints" / f"step_{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 16)
    (ckpt / "manifest.json").write_text(json.dumps({"step": step}))
    return ckpt


# ---------------------------------------------------------------------------
# R15 — sweep ≥ 8 trials
# ---------------------------------------------------------------------------


@pytest.fixture()
def sweep_setup(tmp_path: Path):
    """Write base YAML + sweep YAML with a 3×3 grid (9 trials)."""
    run_root = tmp_path / "runs"
    run_root.mkdir()

    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        yaml.safe_dump(
            {
                "mode": "lab",
                "exp": "r15",
                "run_root": str(run_root),
                "model": {"name": "tiny_lm"},
            }
        ),
        encoding="utf-8",
    )

    sweep_cfg = tmp_path / "sweep.yaml"
    sweep_cfg.write_text(
        yaml.safe_dump(
            {
                "name": "r15_sweep",
                "metric": "loss",
                "direction": "minimize",
                "params": {
                    "optim.lr": [1e-4, 3e-4, 1e-3],
                    "optim.weight_decay": [0.0, 0.05, 0.1],
                },
            }
        ),
        encoding="utf-8",
    )

    return base_cfg, sweep_cfg, run_root


def test_r15_sweep_has_nine_trials(sweep_setup, tmp_path: Path):
    base_cfg, sweep_cfg, run_root = sweep_setup
    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    configs = runner._generate_configs()
    assert len(configs) == 9, f"Expected 9 trials from 3×3 grid, got {len(configs)}"


def test_r15_sweep_produces_report(sweep_setup, tmp_path: Path):
    base_cfg, sweep_cfg, run_root = sweep_setup
    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    sweep_run_root = run_root / "sweep_r15_sweep"

    # Stub subprocess + pre-create fake run dirs with metrics
    import itertools

    lrs = [1e-4, 3e-4, 1e-3]
    wds = [0.0, 0.05, 0.1]
    all_metrics = [2.5 - (lr / 1e-3) * 0.5 for lr, _ in itertools.product(lrs, wds)]

    call_count = [0]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        trial_exp = f"r15_sweep_trial_{i:03d}"
        _make_fake_run_dir(sweep_run_root, trial_exp, all_metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    # Acceptance: ≥ 8 trials must complete
    ok_count = sum(1 for t in report.trials if t.status in ("ok", "pruned"))
    assert ok_count >= 8, f"Expected ≥ 8 trials, got {ok_count}"

    # Acceptance: markdown report with top-K + sensitivity
    md = render_sweep_markdown(report, top_k=5)
    assert "Top-" in md
    assert "sensitivity" in md.lower()
    assert "best" in md.lower()

    # Write report to file
    out = write_sweep_report(report, tmp_path / "sweep_report.md")
    expect_exists(out, tmp_path, what="sweep_report.md")
    content = out.read_text()
    assert "r15_sweep" in content


def test_r15_report_contains_sensitivity(sweep_setup, tmp_path: Path):
    """Sensitivity section must be present and non-trivial."""
    base_cfg, sweep_cfg, run_root = sweep_setup
    runner = SweepRunner(base_cfg, sweep_cfg, strategy="grid")
    sweep_run_root = run_root / "sweep_r15_sweep"

    call_count = [0]
    metrics = [2.0 - i * 0.1 for i in range(9)]

    def fake_run(cmd, **kwargs):
        i = call_count[0]
        call_count[0] += 1
        trial_exp = f"r15_sweep_trial_{i:03d}"
        _make_fake_run_dir(sweep_run_root, trial_exp, metrics[i])
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("lighttrain.lab.sweep.subprocess.run", side_effect=fake_run):
        report = runner.run()

    assert "optim.lr" in report.sensitivity or "optim.weight_decay" in report.sensitivity


# ---------------------------------------------------------------------------
# R16 — fork 3-generation lineage
# ---------------------------------------------------------------------------


def _make_gen1(tmp_path: Path):
    from lighttrain.observability.lineage.store import LineageStore

    gen1_dir = tmp_path / "gen1"
    gen1_dir.mkdir()
    ckpt1 = _make_ckpt(gen1_dir)
    (gen1_dir / "env.json").write_text("{}")
    # Create lineage store so fork can record edge
    with LineageStore(gen1_dir / "lineage.sqlite"):
        pass
    return gen1_dir, ckpt1


def test_r16_fork_gen2_from_gen1(tmp_path: Path):
    gen1_dir, ckpt1 = _make_gen1(tmp_path)

    r2 = fork(ckpt1, {"run_root": str(tmp_path), "exp": "gen2", "optim": {"lr": 1.5e-4}})

    expect_exists(r2.new_run_dir, tmp_path, what="gen2 run dir")
    expect_exists(r2.new_run_dir / "fork_meta.json", r2.new_run_dir, what="gen2 fork_meta.json")


def test_r16_three_generation_lineage(tmp_path: Path):
    from lighttrain.observability.lineage.store import LineageStore

    gen1_dir, ckpt1 = _make_gen1(tmp_path)

    # Gen 2
    r2 = fork(ckpt1, {"run_root": str(tmp_path), "exp": "gen2"})
    ckpt2 = _make_ckpt(r2.new_run_dir)
    (r2.new_run_dir / "env.json").write_text("{}")
    with LineageStore(r2.new_run_dir / "lineage.sqlite"):
        pass

    # Gen 3
    r3 = fork(ckpt2, {"run_root": str(tmp_path), "exp": "gen3"})

    # All three generations must have fork provenance
    expect_exists(r2.new_run_dir / "fork_meta.json", r2.new_run_dir, what="gen2 fork_meta.json")
    expect_exists(r3.new_run_dir / "fork_meta.json", r3.new_run_dir, what="gen3 fork_meta.json")

    meta2 = json.loads((r2.new_run_dir / "fork_meta.json").read_text())
    meta3 = json.loads((r3.new_run_dir / "fork_meta.json").read_text())
    assert str(ckpt1.resolve()) in meta2["fork_of_checkpoint"]
    assert str(ckpt2.resolve()) in meta3["fork_of_checkpoint"]

    # Gen1 lineage store records fork_of edge to gen2
    with LineageStore(gen1_dir / "lineage.sqlite") as store:
        edges1 = list(store.iter_edges(kind="fork_of"))
    assert len(edges1) == 1

    # Gen2 lineage store records fork_of edge to gen3
    with LineageStore(r2.new_run_dir / "lineage.sqlite") as store:
        edges2 = list(store.iter_edges(kind="fork_of"))
    assert len(edges2) == 1


def test_r16_compare_shows_lr_diff(tmp_path: Path):
    """compare() on gen1/gen2/gen3 run dirs shows the LR diff."""
    import yaml as _yaml

    def _make_run_with_config(run_dir: Path, lr: float) -> Path:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.resolved.yaml").write_text(
            _yaml.safe_dump({"optim": {"lr": lr}}), encoding="utf-8"
        )
        (run_dir / "logs" / "metrics.jsonl").write_text(
            json.dumps({"step": 50, "loss": 2.0}) + "\n", encoding="utf-8"
        )
        return run_dir

    r1 = _make_run_with_config(tmp_path / "run1", 3e-4)
    r2 = _make_run_with_config(tmp_path / "run2", 1.5e-4)
    r3 = _make_run_with_config(tmp_path / "run3", 7.5e-5)

    report = compare([r1, r2, r3])
    assert "optim.lr" in report.config_diff

    out = render_ascii(report)
    assert "optim.lr" in out
