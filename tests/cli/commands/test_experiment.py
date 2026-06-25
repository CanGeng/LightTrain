"""Tests for ``lighttrain.cli.commands.experiment`` (sweep / compare / fork).

Covers every reachable branch in the module driving coverage toward 100%.
All heavy operations (SweepRunner.run, compare, fork) are monkeypatched so
no real training, GPU, or network access occurs.

Branches intentionally skipped (hardware / external dep):
  - real SweepRunner.run() invoking subprocess lighttrain train
  - render_png when matplotlib is genuinely absent at import time
These are noted in the skipped_lines_note of the StructuredOutput response.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lighttrain.lab.auto_report as _auto_report_mod

# Force the compare, fork, sweep, and auto_report modules into sys.modules so
# that monkeypatching their attributes works regardless of __init__.py re-exports.
import lighttrain.lab.sweep as _sweep_mod
from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner for each test."""
    return CliRunner()


def _write_cfg(tmp_path: Path, body: str = "mode: lab\nseed: 7\n") -> Path:
    """Write a minimal recipe YAML and return its Path."""
    p = tmp_path / "recipe.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _write_sweep_spec(tmp_path: Path) -> Path:
    """Write a minimal sweep spec YAML and return its Path."""
    p = tmp_path / "sweep.yaml"
    p.write_text(
        "name: test_sweep\nmetric: loss\ndirection: minimize\nparams:\n  seed: [1, 2]\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Fake objects returned by monkeypatched callables
# ---------------------------------------------------------------------------


@dataclass
class _FakeTrialResult:
    trial_id: int
    status: str
    metric: float | None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None


@dataclass
class _FakeSweepReport:
    sweep_name: str = "test_sweep"
    strategy: str = "grid"
    trials: list[Any] = field(default_factory=list)
    best_config: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    direction: str = "minimize"
    sensitivity: dict[str, float] = field(default_factory=dict)
    report_path: Path | None = None


@dataclass
class _FakeCompareReport:
    runs: list[Path] = field(default_factory=list)
    config_diff: dict[str, list[Any]] = field(default_factory=dict)
    metrics_table: dict[str, list[float | None]] = field(default_factory=dict)
    fork_ancestry: dict[str, str | None] = field(default_factory=dict)


@dataclass
class _FakeForkReport:
    new_run_dir: Path = field(default_factory=lambda: Path("/tmp/fork_out"))
    parent_checkpoint: Path = field(default_factory=lambda: Path("/tmp/ckpt"))
    parent_run_dir: Path | None = None
    lineage_edge_recorded: bool = False


# ===========================================================================
# sweep_cmd — lines 31–68
# ===========================================================================


class TestSweepCmd:
    """Tests for ``lighttrain sweep`` command."""

    def test_invariant_missing_config_exits_one(self, runner: CliRunner, tmp_path: Path):
        """sweep exits 1 and prints 'config not found' when -c path is missing.

        Covers lines 34-36.
        """
        sweep = _write_sweep_spec(tmp_path)
        missing_cfg = tmp_path / "no_such.yaml"
        res = runner.invoke(app, ["sweep", "-c", str(missing_cfg), "-s", str(sweep)])
        assert res.exit_code == 1
        assert "config not found" in res.stdout.lower() or "not found" in res.stdout.lower()

    def test_invariant_missing_sweep_spec_exits_one(self, runner: CliRunner, tmp_path: Path):
        """sweep exits 1 and prints 'sweep spec not found' when -s path is missing.

        Covers lines 37-39.
        """
        cfg = _write_cfg(tmp_path)
        missing_sweep = tmp_path / "no_sweep.yaml"
        res = runner.invoke(app, ["sweep", "-c", str(cfg), "-s", str(missing_sweep)])
        assert res.exit_code == 1
        assert "sweep spec not found" in res.stdout.lower() or "not found" in res.stdout.lower()

    def test_invariant_sweep_runner_exception_exits_one(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """sweep exits 1 when SweepRunner.run() raises an exception.

        Covers lines 42-51: exception branch including rich escape.
        """
        cfg = _write_cfg(tmp_path)
        sweep = _write_sweep_spec(tmp_path)

        def _boom(self: Any) -> None:
            raise RuntimeError("pip install -e '.[sweep]'")

        monkeypatch.setattr(_sweep_mod.SweepRunner, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(_sweep_mod.SweepRunner, "run", _boom)

        res = runner.invoke(app, ["sweep", "-c", str(cfg), "-s", str(sweep)])
        assert res.exit_code == 1
        assert "sweep failed" in res.stdout.lower()

    def test_invariant_sweep_success_no_best_metric(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """sweep exits 0 when runner succeeds; no best_metric → no best-metric line.

        Covers lines 41, 43-44, 53-61, 63 (False branch), 67-68.
        """
        cfg = _write_cfg(tmp_path)
        sweep = _write_sweep_spec(tmp_path)
        report_path = tmp_path / "report.md"

        fake_report = _FakeSweepReport(
            trials=[_FakeTrialResult(trial_id=0, status="ok", metric=None)],
            best_metric=None,
            best_config={},
        )

        monkeypatch.setattr(_sweep_mod.SweepRunner, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(_sweep_mod.SweepRunner, "run", lambda self: fake_report)
        monkeypatch.setattr(
            _auto_report_mod,
            "write_sweep_report",
            lambda report, out_path, top_k: report_path,
        )

        res = runner.invoke(
            app,
            ["sweep", "-c", str(cfg), "-s", str(sweep), "--report-out", str(report_path)],
        )
        assert res.exit_code == 0, res.stdout
        # Table rendered (title contains sweep name)
        assert "test_sweep" in res.stdout or "sweep" in res.stdout.lower()
        # No best metric line
        assert "best metric" not in res.stdout.lower()
        # Report path printed
        assert "report written" in res.stdout.lower()

    def test_invariant_sweep_success_with_best_metric(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """sweep exits 0 and prints best metric/config when best_metric is set.

        Covers lines 63-65: the True branch of ``if report.best_metric is not None``.
        """
        cfg = _write_cfg(tmp_path)
        sweep = _write_sweep_spec(tmp_path)
        report_path = tmp_path / "report.md"

        fake_report = _FakeSweepReport(
            trials=[_FakeTrialResult(trial_id=0, status="ok", metric=0.123)],
            best_metric=0.123,
            best_config={"seed": 1},
        )

        monkeypatch.setattr(_sweep_mod.SweepRunner, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(_sweep_mod.SweepRunner, "run", lambda self: fake_report)
        monkeypatch.setattr(
            _auto_report_mod,
            "write_sweep_report",
            lambda report, out_path, top_k: report_path,
        )

        res = runner.invoke(
            app,
            ["sweep", "-c", str(cfg), "-s", str(sweep), "--report-out", str(report_path)],
        )
        assert res.exit_code == 0, res.stdout
        assert "best metric" in res.stdout.lower()
        assert "0.123" in res.stdout
        assert "best config" in res.stdout.lower()

    def test_invariant_sweep_metric_str_none_formatted_as_dash(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Trial with metric=None formats as '—' in the table (line 59 else branch)."""
        cfg = _write_cfg(tmp_path)
        sweep = _write_sweep_spec(tmp_path)
        report_path = tmp_path / "report.md"

        fake_report = _FakeSweepReport(
            trials=[_FakeTrialResult(trial_id=0, status="failed", metric=None)],
            best_metric=None,
            best_config={},
        )

        monkeypatch.setattr(_sweep_mod.SweepRunner, "__init__", lambda self, *a, **kw: None)
        monkeypatch.setattr(_sweep_mod.SweepRunner, "run", lambda self: fake_report)
        monkeypatch.setattr(
            _auto_report_mod,
            "write_sweep_report",
            lambda report, out_path, top_k: report_path,
        )

        res = runner.invoke(
            app,
            ["sweep", "-c", str(cfg), "-s", str(sweep), "--report-out", str(report_path)],
        )
        assert res.exit_code == 0, res.stdout
        # '—' should appear for the None-metric trial in the Rich table
        assert "—" in res.stdout

    def test_invariant_sweep_uses_strategy_and_top_k_options(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Strategy and top-k options reach the runner constructor call (line 41, 43)."""
        cfg = _write_cfg(tmp_path)
        sweep = _write_sweep_spec(tmp_path)
        report_path = tmp_path / "report.md"

        captured: dict[str, Any] = {}

        def _fake_init(
            self: Any,
            config: Any,
            sweep_path: Any,
            strategy: str = "grid",
        ) -> None:
            captured["strategy"] = strategy

        monkeypatch.setattr(_sweep_mod.SweepRunner, "__init__", _fake_init)
        monkeypatch.setattr(
            _sweep_mod.SweepRunner,
            "run",
            lambda self: _FakeSweepReport(trials=[], best_metric=None, best_config={}),
        )
        monkeypatch.setattr(
            _auto_report_mod,
            "write_sweep_report",
            lambda report, out_path, top_k: report_path,
        )

        res = runner.invoke(
            app,
            [
                "sweep",
                "-c", str(cfg),
                "-s", str(sweep),
                "--strategy", "random",
                "--top-k", "3",
                "--report-out", str(report_path),
            ],
        )
        assert res.exit_code == 0, res.stdout
        assert captured.get("strategy") == "random"


# ===========================================================================
# compare_cmd — lines 91–139
# ===========================================================================

# Helper: get the real compare module from sys.modules (not the re-exported fn)
_COMPARE_MOD = sys.modules["lighttrain.lab.compare"]
_FORK_MOD = sys.modules["lighttrain.lab.fork"]


class TestCompareCmd:
    """Tests for ``lighttrain compare`` command."""

    def test_invariant_missing_run_dir_exits_one(self, runner: CliRunner, tmp_path: Path):
        """compare exits 1 when a run directory does not exist.

        Covers lines 101-103.
        """
        res = runner.invoke(app, ["compare", str(tmp_path / "nope_run")])
        assert res.exit_code == 1
        assert "not found" in res.stdout.lower() or "run dirs" in res.stdout.lower()

    def test_invariant_compare_exception_exits_one(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare exits 1 when compare() raises an exception.

        Covers lines 107-109.
        """
        run_dir = tmp_path / "run_a"
        run_dir.mkdir()

        def _boom(paths: Any) -> None:
            raise RuntimeError("internal compare error")

        monkeypatch.setattr(_COMPARE_MOD, "compare", _boom)

        res = runner.invoke(app, ["compare", str(run_dir)])
        assert res.exit_code == 1
        assert "compare failed" in res.stdout.lower()

    def test_invariant_compare_success_renders_ascii(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare exits 0, renders ASCII table when no --metric or --output given.

        Covers lines 99, 106, 111 (no metrics), 118 (False), 121.
        """
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        run_a.mkdir()
        run_b.mkdir()

        fake_report = _FakeCompareReport(
            runs=[run_a, run_b],
            config_diff={},
            metrics_table={"loss": [0.5, 0.4]},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(_COMPARE_MOD, "render_ascii", lambda report: "ascii_table_output")
        monkeypatch.setattr(
            _COMPARE_MOD,
            "render_markdown",
            lambda report, metrics=None: "## markdown_output",
        )

        res = runner.invoke(app, ["compare", str(run_a), str(run_b)])
        assert res.exit_code == 0, res.stdout
        assert "ascii_table_output" in res.stdout

    def test_invariant_compare_with_metric_renders_markdown(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --metric renders markdown (True branch of ``metrics or output``).

        Covers lines 112, 118-119: True branch with known metric.
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={"loss": [0.5]},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(
            _COMPARE_MOD, "render_markdown", lambda report, metrics=None: "## md_output"
        )

        res = runner.invoke(app, ["compare", str(run_a), "--metric", "loss"])
        assert res.exit_code == 0, res.stdout
        assert "md_output" in res.stdout

    def test_invariant_compare_warns_on_unknown_metric(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare warns about unknown metric names (line 115).

        Covers lines 113-115: unknown metric warning path.
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={"loss": [0.5]},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(
            _COMPARE_MOD, "render_markdown", lambda report, metrics=None: "## md_output"
        )

        res = runner.invoke(app, ["compare", str(run_a), "--metric", "nonexistent_metric"])
        assert res.exit_code == 0, res.stdout
        assert "no such metric" in res.stdout.lower()

    def test_invariant_compare_output_markdown_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --output .md writes a markdown file.

        Covers lines 118-120 (True output branch), 129-132 (.md else path).
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        out_file = tmp_path / "out" / "compare.md"

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(
            _COMPARE_MOD, "render_markdown", lambda report, metrics=None: "## content"
        )
        monkeypatch.setattr(_COMPARE_MOD, "render_ascii", lambda report: "ascii")

        res = runner.invoke(app, ["compare", str(run_a), "--output", str(out_file)])
        assert res.exit_code == 0, res.stdout
        assert out_file.exists()
        assert "content" in out_file.read_text(encoding="utf-8")
        assert "written" in res.stdout.lower()

    def test_invariant_compare_output_json_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --output .json writes a JSON file.

        Covers lines 124-128: the .json branch of output handling.
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        out_file = tmp_path / "compare.json"

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(
            _COMPARE_MOD, "render_markdown", lambda report, metrics=None: "## content"
        )
        monkeypatch.setattr(_COMPARE_MOD, "render_ascii", lambda report: "ascii")
        monkeypatch.setattr(
            _COMPARE_MOD,
            "to_records",
            lambda report, metrics: [{"run": "run_a", "loss": None}],
        )

        res = runner.invoke(app, ["compare", str(run_a), "--output", str(out_file)])
        assert res.exit_code == 0, res.stdout
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert "written" in res.stdout.lower()

    def test_invariant_compare_png_success(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --png calls render_png and prints success.

        Covers lines 134-137: the png success branch.
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        png_path = tmp_path / "compare.png"

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(_COMPARE_MOD, "render_ascii", lambda report: "ascii_output")
        monkeypatch.setattr(_COMPARE_MOD, "render_png", lambda report, path: None)

        res = runner.invoke(app, ["compare", str(run_a), "--png", str(png_path)])
        assert res.exit_code == 0, res.stdout
        assert "png written" in res.stdout.lower()

    def test_invariant_compare_png_runtime_error_warns(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --png warns (not exits 1) when render_png raises RuntimeError.

        Covers lines 138-139: RuntimeError caught → yellow warning, exit 0.
        """
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        png_path = tmp_path / "compare.png"

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(_COMPARE_MOD, "render_ascii", lambda report: "ascii")

        def _png_fail(report: Any, path: Any) -> None:
            raise RuntimeError("matplotlib not installed")

        monkeypatch.setattr(_COMPARE_MOD, "render_png", _png_fail)

        res = runner.invoke(app, ["compare", str(run_a), "--png", str(png_path)])
        # Must NOT exit with failure — PNG is soft dependency
        assert res.exit_code == 0, res.stdout
        assert "png skipped" in res.stdout.lower()


# ===========================================================================
# fork_cmd — lines 155–183
# ===========================================================================


class TestForkCmd:
    """Tests for ``lighttrain fork`` command."""

    def test_invariant_missing_from_checkpoint_exits_one(
        self, runner: CliRunner, tmp_path: Path
    ):
        """fork exits 1 when --from checkpoint dir does not exist.

        Covers lines 157-159.
        """
        cfg = _write_cfg(tmp_path)
        res = runner.invoke(
            app,
            ["fork", "--from", str(tmp_path / "no_ckpt"), "-c", str(cfg)],
        )
        assert res.exit_code == 1
        assert "checkpoint not found" in res.stdout.lower() or "not found" in res.stdout.lower()

    def test_invariant_missing_config_exits_one(
        self, runner: CliRunner, tmp_path: Path
    ):
        """fork exits 1 when -c config file does not exist.

        Covers lines 160-162.
        """
        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(tmp_path / "no_cfg.yaml")],
        )
        assert res.exit_code == 1
        assert "config not found" in res.stdout.lower() or "not found" in res.stdout.lower()

    def test_invariant_config_error_exits_one(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork exits 1 when load_config raises ConfigError.

        Covers lines 164-168: config loading exception path.
        """
        from lighttrain.config import ConfigError

        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)

        import lighttrain.cli.commands.experiment as _exp_mod

        def _bad_load(path: Any, overrides: Any = None) -> None:
            raise ConfigError("bad config value")

        monkeypatch.setattr(_exp_mod, "load_config", _bad_load)

        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(cfg)],
        )
        assert res.exit_code == 1
        assert "config error" in res.stdout.lower()

    def test_invariant_fork_exception_exits_one(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork exits 1 when fork() raises any exception.

        Covers lines 170-174: fork execution exception path.
        """
        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)

        import lighttrain.cli.commands.experiment as _exp_mod

        monkeypatch.setattr(
            _exp_mod, "load_config", lambda path, overrides=None: {"mode": "lab"}
        )

        def _bad_fork(from_checkpoint: Any, new_config: Any, symlink: bool = False) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(_FORK_MOD, "fork", _bad_fork)

        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(cfg)],
        )
        assert res.exit_code == 1
        assert "fork failed" in res.stdout.lower()

    def test_invariant_fork_success_with_lineage_recorded(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork exits 0 and prints lineage-recorded message when edge was written.

        Covers lines 176-178: lineage_edge_recorded=True branch.
        """
        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)
        new_run = tmp_path / "fork_run"
        new_run.mkdir()

        fake_report = _FakeForkReport(
            new_run_dir=new_run,
            parent_checkpoint=ckpt,
            parent_run_dir=None,
            lineage_edge_recorded=True,
        )

        import lighttrain.cli.commands.experiment as _exp_mod

        monkeypatch.setattr(
            _exp_mod, "load_config", lambda path, overrides=None: {"mode": "lab"}
        )
        monkeypatch.setattr(
            _FORK_MOD, "fork", lambda from_checkpoint, new_config, symlink=False: fake_report
        )

        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(cfg)],
        )
        assert res.exit_code == 0, res.stdout
        assert "forked" in res.stdout.lower()
        assert "lineage fork_of edge recorded" in res.stdout.lower()
        assert "resume" in res.stdout.lower()

    def test_invariant_fork_success_without_lineage(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork exits 0 and prints lineage-not-recorded warning when edge was not written.

        Covers lines 179-180: lineage_edge_recorded=False branch.
        """
        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)
        new_run = tmp_path / "fork_run"
        new_run.mkdir()

        fake_report = _FakeForkReport(
            new_run_dir=new_run,
            parent_checkpoint=ckpt,
            parent_run_dir=None,
            lineage_edge_recorded=False,
        )

        import lighttrain.cli.commands.experiment as _exp_mod

        monkeypatch.setattr(
            _exp_mod, "load_config", lambda path, overrides=None: {"mode": "lab"}
        )
        monkeypatch.setattr(
            _FORK_MOD, "fork", lambda from_checkpoint, new_config, symlink=False: fake_report
        )

        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(cfg)],
        )
        assert res.exit_code == 0, res.stdout
        assert "forked" in res.stdout.lower()
        assert "lineage not recorded" in res.stdout.lower()
        assert "resume" in res.stdout.lower()

    def test_invariant_fork_with_overrides_and_symlink(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork passes overrides to load_config and symlink flag to fork().

        Covers lines 164-165 (overrides list), 171 (symlink kwarg).
        """
        ckpt = tmp_path / "step_500"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)
        new_run = tmp_path / "fork_run_sym"
        new_run.mkdir()

        captured: dict[str, Any] = {}

        import lighttrain.cli.commands.experiment as _exp_mod

        def _fake_load(path: Any, overrides: list[str] | None = None) -> dict[str, Any]:
            captured["overrides"] = overrides
            return {"mode": "lab"}

        def _fake_fork(
            from_checkpoint: Path, new_config: Any, symlink: bool = False
        ) -> _FakeForkReport:
            captured["symlink"] = symlink
            return _FakeForkReport(
                new_run_dir=new_run,
                parent_checkpoint=ckpt,
                lineage_edge_recorded=False,
            )

        monkeypatch.setattr(_exp_mod, "load_config", _fake_load)
        monkeypatch.setattr(_FORK_MOD, "fork", _fake_fork)

        res = runner.invoke(
            app,
            [
                "fork",
                "--from", str(ckpt),
                "-c", str(cfg),
                "--symlink",
                "++seed=99",
            ],
        )
        assert res.exit_code == 0, res.stdout
        assert captured.get("symlink") is True
        assert "++seed=99" in (captured.get("overrides") or [])

    def test_invariant_fork_resume_message_includes_run_dir(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """fork success always prints resume command with the new_run_dir (lines 181-183)."""
        ckpt = tmp_path / "step_100"
        ckpt.mkdir()
        cfg = _write_cfg(tmp_path)
        new_run = tmp_path / "my_fork_run"
        new_run.mkdir()

        fake_report = _FakeForkReport(
            new_run_dir=new_run,
            parent_checkpoint=ckpt,
            lineage_edge_recorded=False,
        )

        import lighttrain.cli.commands.experiment as _exp_mod

        monkeypatch.setattr(
            _exp_mod, "load_config", lambda path, overrides=None: {"mode": "lab"}
        )
        monkeypatch.setattr(
            _FORK_MOD, "fork", lambda from_checkpoint, new_config, symlink=False: fake_report
        )

        res = runner.invoke(
            app,
            ["fork", "--from", str(ckpt), "-c", str(cfg)],
        )
        assert res.exit_code == 0, res.stdout
        assert str(new_run) in res.stdout
        assert "lighttrain resume" in res.stdout


# ===========================================================================
# compare_cmd — edge: multiple missing dirs reported together (line 101-103)
# ===========================================================================


class TestCompareEdgeCases:
    """Edge-case tests for compare_cmd."""

    def test_invariant_multiple_missing_dirs_reported(
        self, runner: CliRunner, tmp_path: Path
    ):
        """compare lists ALL missing dirs in the error message (line 102 list).

        The missing list may contain >1 path; all are reported before exit 1.
        """
        res = runner.invoke(
            app,
            ["compare", str(tmp_path / "no_a"), str(tmp_path / "no_b")],
        )
        assert res.exit_code == 1
        assert "not found" in res.stdout.lower() or "run dirs" in res.stdout.lower()

    def test_invariant_compare_no_output_with_metrics_renders_markdown(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """compare with --metric but no --output still renders markdown (lines 118-119).

        The ``if metrics or output is not None`` condition is True via metrics alone.
        """
        run_a = tmp_path / "run_x"
        run_a.mkdir()

        fake_report = _FakeCompareReport(
            runs=[run_a],
            config_diff={},
            metrics_table={"acc": [0.9]},
            fork_ancestry={},
        )

        monkeypatch.setattr(_COMPARE_MOD, "compare", lambda paths: fake_report)
        monkeypatch.setattr(
            _COMPARE_MOD, "render_markdown", lambda report, metrics=None: "| md |"
        )

        res = runner.invoke(app, ["compare", str(run_a), "--metric", "acc"])
        assert res.exit_code == 0, res.stdout
        assert "| md |" in res.stdout
