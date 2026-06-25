"""Tests for ``lighttrain.cli.commands.diagnostics``.

Covers the branches not yet reached by the existing test suite:

* ``estimate_cmd``  — ConfigError path; offload-breakdown section; notes loop;
  ``--json`` write; ``_fmt_bytes`` GB unit
* ``doctor_cmd``    — no checkpoints/ dir; crash bundles detected; callback
  failures file (> 0 lines); callback file read-error (BLE001 except path)
* ``dry_run_cmd``   — ``--build`` success (with + without parameters()); ``--build``
  error path
* ``overfit_cmd``   — ConfigError/FileNotFoundError path; happy path (logger closed)
* ``profile_cmd``   — ConfigError path; happy path; chrome-trace export failure
* ``inspect_data_cmd`` — ConfigError path; no-dataset error; decoded flag success;
  tokenizer decode failure; empty dataset (no length summary)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.config import ConfigError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_runner = CliRunner()


def _invoke(*args: str):
    """Convenience wrapper around CliRunner.invoke(app, [...])."""
    return _runner.invoke(app, list(args))


def _minimal_cfg(tmp_path: Path, content: str = "mode: lab\nseed: 7\n") -> Path:
    """Write the smallest valid recipe yaml and return the path."""
    p = tmp_path / "recipe.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# estimate_cmd
# ---------------------------------------------------------------------------


class TestEstimateCmd:
    def test_invariant_config_error_exits_one(self, tmp_path, monkeypatch):
        """``estimate -c <cfg>`` with a bad config must exit 1 and name the error.

        Covers lines 41-43 (ConfigError catch + typer.Exit(code=1)).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        monkeypatch.setattr(_diag, "load_config", _config_error_raiser("bad config here"))
        res = _invoke("estimate", "-c", str(cfg))
        assert res.exit_code == 1
        assert "config error" in res.stdout
        assert "bad config here" in res.stdout

    def test_invariant_offload_section_rendered(self, tmp_path, monkeypatch):
        """When ``rpt.offload is not None`` the LayerOffload table is printed.

        Covers lines 72-87 (the offload-breakdown branch).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        rpt = _make_estimate_report(with_offload=True, notes=[])
        monkeypatch.setattr(_diag, "load_config", lambda *a, **kw: object())
        _patch_lab_estimate(monkeypatch, rpt)
        res = _invoke("estimate", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "LayerOffload breakdown" in res.stdout
        assert "resident_layers" in res.stdout
        assert "recommended_mode" in res.stdout

    def test_invariant_notes_loop_prints_each_note(self, tmp_path, monkeypatch):
        """Notes in ``rpt.notes`` must each appear on stdout.

        Covers lines 89-90 (the notes loop).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        rpt = _make_estimate_report(with_offload=False, notes=["alpha note", "beta note"])
        monkeypatch.setattr(_diag, "load_config", lambda *a, **kw: object())
        _patch_lab_estimate(monkeypatch, rpt)
        res = _invoke("estimate", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "alpha note" in res.stdout
        assert "beta note" in res.stdout

    def test_invariant_json_out_written(self, tmp_path, monkeypatch):
        """``--json <path>`` writes a JSON file and prints a confirmation.

        Covers lines 92-95 (json_out branch).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        json_out = tmp_path / "sub" / "report.json"
        rpt = _make_estimate_report(with_offload=False, notes=[])
        monkeypatch.setattr(_diag, "load_config", lambda *a, **kw: object())
        _patch_lab_estimate(monkeypatch, rpt)
        res = _invoke("estimate", "-c", str(cfg), "--json", str(json_out))
        assert res.exit_code == 0, res.stdout
        assert json_out.exists(), "JSON output file was not created"
        assert "wrote" in res.stdout

    def test_invariant_fmt_bytes_gb_unit(self, tmp_path, monkeypatch):
        """A value >= 1 GB must be formatted with the 'GB' unit in the table.

        Covers line 55 (the GB branch inside _fmt_bytes).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        gb2 = int(2 * 1024**3)  # 2 GB — forces the GB-unit branch
        rpt = _make_estimate_report(
            with_offload=False,
            notes=[],
            param_bytes=gb2,
        )
        monkeypatch.setattr(_diag, "load_config", lambda *a, **kw: object())
        _patch_lab_estimate(monkeypatch, rpt)
        res = _invoke("estimate", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "GB" in res.stdout


# ---------------------------------------------------------------------------
# doctor_cmd (additional paths beyond what test_doctor_*.py already covers)
# ---------------------------------------------------------------------------


class TestDoctorCmd:
    def test_invariant_no_checkpoints_dir_shows_na(self, tmp_path):
        """A run dir *without* ``checkpoints/`` prints the N/A message.

        Covers line 130 (the else branch of ``if ckpt_dir.exists()``).
        """
        run = tmp_path / "run"
        run.mkdir()
        res = _invoke("doctor", "--run", str(run))
        # No problems from missing checkpoints dir alone.
        assert "N/A" in res.stdout
        assert "checkpoints" in res.stdout

    def test_invariant_crash_bundles_flagged_as_problem(self, tmp_path):
        """Crash bundles under ``diagnostics/crash_*`` must bump ``problems``
        and produce exit code 2.

        Covers lines 206-207 (crash detection branch).
        """
        run = tmp_path / "run"
        diag_dir = run / "diagnostics"
        diag_dir.mkdir(parents=True)
        (diag_dir / "crash_step42").mkdir()

        res = _invoke("doctor", "--run", str(run))
        assert res.exit_code == 2
        assert "crash bundles" in res.stdout
        assert "1 crash" in res.stdout

    def test_invariant_callback_failures_with_lines_flagged(self, tmp_path):
        """A non-empty ``callback_failures.jsonl`` with one line must be
        reported as an isolated failure (NOT bumping exit-code, just yellow).

        Covers lines 228-232 (n_failures > 0 branch).
        """
        run = tmp_path / "run"
        diag_dir = run / "diagnostics"
        diag_dir.mkdir(parents=True)
        (diag_dir / "callback_failures.jsonl").write_text(
            '{"step":1,"callback":"Foo","event":"on_step_end","exc_type":"E","traceback":"x"}\n',
            encoding="utf-8",
        )

        res = _invoke("doctor", "--run", str(run))
        # One callback failure does NOT add to `problems` — only crash/NaN/schema
        # do.  Exit code should be 0 for this minimal run.
        assert "1 isolated failure" in res.stdout
        assert "callback report" in res.stdout

    def test_pin_current_behavior_callback_file_read_error_treated_as_zero(
        self, tmp_path, monkeypatch
    ):
        """When reading ``callback_failures.jsonl`` raises, n_failures is set to 0
        and the run remains green (no-problem).

        Covers lines 221-222, 227 (BLE001 except branch inside the doctor).

        Pin: the current design swallows the read error and reports 0 failures.
        A strict implementation could raise; we lock the current behaviour.
        """
        run = tmp_path / "run"
        diag_dir = run / "diagnostics"
        diag_dir.mkdir(parents=True)
        cb_log = diag_dir / "callback_failures.jsonl"
        cb_log.write_text("some data\n", encoding="utf-8")

        original_read_text = Path.read_text

        def _boom(self: Path, *args, **kwargs):
            if self.name == "callback_failures.jsonl":
                raise OSError("simulated read error")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _boom)

        res = _invoke("doctor", "--run", str(run))
        # BLE001 path → n_failures=0 → green report, no problems.
        assert "no failures" in res.stdout
        assert res.exit_code == 0

    def test_invariant_callback_failures_zero_lines_stays_green(self, tmp_path):
        """A ``callback_failures.jsonl`` that exists but has zero non-blank lines
        must stay green (covers line 234, the ``n_failures == 0`` else branch).
        """
        run = tmp_path / "run"
        diag_dir = run / "diagnostics"
        diag_dir.mkdir(parents=True)
        # Only blank lines → n_failures = 0.
        (diag_dir / "callback_failures.jsonl").write_text("\n\n", encoding="utf-8")

        res = _invoke("doctor", "--run", str(run))
        assert "no failures" in res.stdout
        assert res.exit_code == 0


# ---------------------------------------------------------------------------
# dry_run_cmd  (--build paths)
# ---------------------------------------------------------------------------


class TestDryRunBuildPaths:
    def test_invariant_build_error_exits_one(self, tmp_path, monkeypatch):
        """``dry-run --build`` with a model-build failure must exit 1.

        Covers lines 280-282 (Exception catch + typer.Exit(code=1)).
        """
        import lighttrain.cli._runtime as _rt

        cfg = _minimal_cfg(tmp_path)
        monkeypatch.setattr(_rt, "_build_model", _runtime_error_raiser("build failed"))
        res = _invoke("dry-run", "-c", str(cfg), "--build")
        assert res.exit_code == 1
        assert "build error" in res.stdout
        assert "build failed" in res.stdout

    def test_invariant_build_success_prints_model_name_and_params(self, tmp_path, monkeypatch):
        """``dry-run --build`` on a model *with* ``parameters()`` reports param count.

        Covers lines 279-291 (successful build + param-count branch).
        """
        import torch

        import lighttrain.cli._runtime as _rt

        cfg = _minimal_cfg(tmp_path)

        class _TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Linear(4, 4, bias=False)

        monkeypatch.setattr(_rt, "_build_model", lambda _cfg: _TinyModel())
        res = _invoke("dry-run", "-c", str(cfg), "--build")
        assert res.exit_code == 0, res.stdout
        assert "model built" in res.stdout
        assert "_TinyModel" in res.stdout
        assert "16 params" in res.stdout  # 4×4 weight

    def test_invariant_build_success_no_parameters_attr_shows_zero(
        self, tmp_path, monkeypatch
    ):
        """``dry-run --build`` on an object *without* ``parameters()`` reports 0 params.

        Covers the ``else: 0`` branch in the n_params assignment (line 287-288).
        """
        import lighttrain.cli._runtime as _rt

        cfg = _minimal_cfg(tmp_path)

        class _NotAModule:
            """Has no parameters() method."""

        monkeypatch.setattr(_rt, "_build_model", lambda _cfg: _NotAModule())
        res = _invoke("dry-run", "-c", str(cfg), "--build")
        assert res.exit_code == 0, res.stdout
        assert "0 params" in res.stdout


# ---------------------------------------------------------------------------
# overfit_cmd
# ---------------------------------------------------------------------------


class TestOverfitCmd:
    def test_invariant_config_error_exits_one(self, tmp_path, monkeypatch):
        """``overfit -c <cfg>`` with a config/file error must exit 1.

        Covers lines 313-314 (ConfigError catch inside overfit_cmd).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        monkeypatch.setattr(
            _diag, "setup_run_from_config", _config_error_raiser("bad overfit cfg")
        )
        res = _invoke("overfit", "-c", str(cfg))
        assert res.exit_code == 1
        assert "config error" in res.stdout

    def test_invariant_happy_path_runs_and_closes_logger(self, tmp_path, monkeypatch):
        """``overfit`` happy-path: trainer.fit() is called, logger is closed.

        Covers lines 305-321 (the full overfit_cmd success flow).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        closed: list[bool] = []

        class _FakeLogger:
            def close(self):
                closed.append(True)

        class _FakeTrainer:
            def fit(self):
                pass

        bundle = {
            "run_dir": run_dir,
            "trainer": _FakeTrainer(),
            "logger": _FakeLogger(),
        }
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("overfit", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "overfit run_dir" in res.stdout
        assert "overfit complete" in res.stdout
        assert closed, "logger.close() was not called"

    def test_invariant_overfit_without_logger_still_completes(self, tmp_path, monkeypatch):
        """``overfit`` with ``bundle['logger'] is None`` must not raise.

        Covers the ``if bundle.get('logger') is not None`` guard (line 319-320).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        class _FakeTrainer:
            def fit(self):
                pass

        bundle = {"run_dir": run_dir, "trainer": _FakeTrainer(), "logger": None}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("overfit", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "overfit complete" in res.stdout

    def test_invariant_overfit_n_flag_forwarded_to_overrides(self, tmp_path, monkeypatch):
        """``overfit --n 50`` propagates ``++trainer.max_steps=50`` to overrides.

        Covers line 305 (the extra-overrides construction).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        captured_overrides: list[list[str]] = []

        def _capture(config, overrides=None):
            captured_overrides.append(list(overrides or []))
            return {
                "run_dir": run_dir,
                "trainer": _NoOpTrainer(),
                "logger": None,
            }

        monkeypatch.setattr(_diag, "setup_run_from_config", _capture)
        _invoke("overfit", "-c", str(cfg), "--n", "50")
        assert captured_overrides, "setup_run_from_config was not called"
        assert "++trainer.max_steps=50" in captured_overrides[0]
        assert "++trainer.val_every=0" in captured_overrides[0]
        assert "++trainer.ckpt_every=0" in captured_overrides[0]


# ---------------------------------------------------------------------------
# profile_cmd
# ---------------------------------------------------------------------------


class TestProfileCmd:
    def test_invariant_config_error_exits_one(self, tmp_path, monkeypatch):
        """``profile -c <cfg>`` with a bad config must exit 1.

        Covers lines 347-348 (ConfigError catch inside profile_cmd).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        monkeypatch.setattr(
            _diag, "setup_run_from_config", _config_error_raiser("bad profile cfg")
        )
        res = _invoke("profile", "-c", str(cfg))
        assert res.exit_code == 1
        assert "config error" in res.stdout

    def test_invariant_happy_path_writes_trace_and_prints_table(
        self, tmp_path, monkeypatch
    ):
        """``profile`` happy path: trace file is referenced in stdout.

        Covers lines 333-392 (profile_cmd main flow, GPU not required).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        bundle = {
            "run_dir": run_dir,
            "trainer": _StepTrainer(),
            "logger": None,
        }
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)

        fake_profiler = _make_fake_profiler_module(raise_on_export=False)
        monkeypatch.setitem(sys.modules, "torch.profiler", fake_profiler)

        res = _invoke("profile", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "profile trace" in res.stdout
        assert "kernel_table" in res.stdout

    def test_invariant_chrome_trace_export_failure_continues(
        self, tmp_path, monkeypatch
    ):
        """When ``prof.export_chrome_trace`` raises, the command prints a
        warning but continues — the kernel table is still printed.

        Covers lines 379-392 (the export-failure branch inside profile_cmd).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        bundle = {
            "run_dir": run_dir,
            "trainer": _StepTrainer(),
            "logger": None,
        }
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)

        fake_profiler = _make_fake_profiler_module(raise_on_export=True)
        monkeypatch.setitem(sys.modules, "torch.profiler", fake_profiler)

        res = _invoke("profile", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "chrome trace export failed" in res.stdout.lower()
        # Table still printed even after trace failure
        assert "kernel_table" in res.stdout

    def test_invariant_profile_logger_closed_after_loop(self, tmp_path, monkeypatch):
        """Logger is closed inside the ``finally`` block after the profiler loop.

        Covers lines 369-377 (the ``finally`` + logger.close() branch).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        closed: list[bool] = []

        class _FakeLogger:
            def close(self):
                closed.append(True)

        bundle = {
            "run_dir": run_dir,
            "trainer": _StepTrainer(),
            "logger": _FakeLogger(),
        }
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        monkeypatch.setitem(sys.modules, "torch.profiler", _make_fake_profiler_module())

        _invoke("profile", "-c", str(cfg))
        assert closed, "logger.close() was not called in the finally block"

    def test_invariant_profile_logger_close_failure_is_swallowed(
        self, tmp_path, monkeypatch
    ):
        """If the logger raises during close, the exception is swallowed
        (BLE001 guard in the finally block, lines 373-377).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        class _BoomLogger:
            def close(self):
                raise RuntimeError("logger close failed")

        bundle = {
            "run_dir": run_dir,
            "trainer": _StepTrainer(),
            "logger": _BoomLogger(),
        }
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        monkeypatch.setitem(sys.modules, "torch.profiler", _make_fake_profiler_module())

        # Must not propagate the exception from the logger.
        res = _invoke("profile", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout


# ---------------------------------------------------------------------------
# inspect_data_cmd
# ---------------------------------------------------------------------------


class TestInspectDataCmd:
    def test_invariant_config_error_exits_one(self, tmp_path, monkeypatch):
        """``inspect-data -c <cfg>`` with a bad config must exit 1.

        Covers lines 403-405 (ConfigError catch inside inspect_data_cmd).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        monkeypatch.setattr(
            _diag, "setup_run_from_config", _config_error_raiser("bad inspect cfg")
        )
        res = _invoke("inspect-data", "-c", str(cfg))
        assert res.exit_code == 1
        assert "config error" in res.stdout

    def test_invariant_no_dataset_attribute_exits_one(self, tmp_path, monkeypatch):
        """A data module with no ``dataset`` attribute must exit 1 with a clear msg.

        Covers lines 409-410.
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _NoDatasetModule()}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg))
        assert res.exit_code == 1
        assert "no `dataset`" in res.stdout

    def test_invariant_basic_table_printed(self, tmp_path, monkeypatch):
        """Happy path: the idx/len/kept_labels table is printed, summary follows.

        Covers lines 401-448 (primary flow, no ``--decoded``).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=3)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "len" in res.stdout
        assert "kept_labels" in res.stdout
        assert "min=" in res.stdout and "max=" in res.stdout and "mean=" in res.stdout

    def test_invariant_decoded_flag_adds_column(self, tmp_path, monkeypatch):
        """``--decoded`` adds a 'decoded[:80]' column using the tokenizer.

        Covers lines 416-418 (decoded-column branch) and lines 431-432 (tokenizer call).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=2, with_tokenizer=True)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg), "--decoded")
        assert res.exit_code == 0, res.stdout
        assert "decoded[:80]" in res.stdout
        assert "hello world" in res.stdout

    def test_invariant_decoded_no_tokenizer_shows_empty(self, tmp_path, monkeypatch):
        """``--decoded`` with ``data_module.tokenizer = None`` shows an empty cell.

        Covers the branch where ``tokenizer is None`` inside the decoded block
        (line 430 skips the decode call).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=1, with_tokenizer=False)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg), "--decoded")
        assert res.exit_code == 0, res.stdout
        assert "decoded[:80]" in res.stdout

    def test_invariant_decoded_tokenizer_error_shows_placeholder(
        self, tmp_path, monkeypatch
    ):
        """A tokenizer that raises during decode → ``<decode error>`` placeholder.

        Covers lines 434-440 (BLE001 decode-exception branch).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=1, tokenizer_raises=True)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg), "--decoded")
        assert res.exit_code == 0, res.stdout
        assert "<decode error>" in res.stdout

    def test_invariant_empty_dataset_skips_length_summary(self, tmp_path, monkeypatch):
        """A dataset with 0 samples must not print the length summary line.

        Covers the ``if lengths:`` guard (line 444).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=0)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        assert "min=" not in res.stdout

    def test_invariant_n_flag_limits_samples(self, tmp_path, monkeypatch):
        """``--n 2`` with a 5-sample dataset shows only 2 rows.

        Covers the ``min(n, len(dataset))`` slice (line 422).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)
        bundle = {"data": _FakeDataModule(n_samples=5)}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg), "--n", "2")
        assert res.exit_code == 0, res.stdout
        # Title says "first 2 samples" and the table only has rows 0 and 1.
        assert "first 2 samples" in res.stdout
        # Row index 4 (which would appear with --n 5) must not be present.
        assert "│   4 │" not in res.stdout

    def test_invariant_labels_without_input_ids_uses_labels_as_ids(
        self, tmp_path, monkeypatch
    ):
        """A sample with no ``input_ids`` uses an empty ids list; kept_labels counts
        non-(-100) entries in ``labels``.

        Line 423: ``ids = list(sample.get("input_ids", []))`` → [] when no input_ids.
        Line 424: ``labels = list(sample.get("labels", ids))`` → [10, 20, -100].
        The table row shows len(ids)=0 and kept_labels=2/3 (two non-(-100) entries).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)

        class _LabelsOnlyDataset:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                # No 'input_ids' key — only 'labels'
                return {"labels": [10, 20, -100]}

        class _DataModule:
            dataset = _LabelsOnlyDataset()
            tokenizer = None

        bundle = {"data": _DataModule()}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg))
        assert res.exit_code == 0, res.stdout
        # ids=[]: length column is 0; kept_labels shows 2 out of 3 labels
        assert "2/3" in res.stdout

    def test_invariant_newlines_in_decoded_text_escaped(self, tmp_path, monkeypatch):
        """Newlines in decoded text must be replaced with ``\\n``.

        Covers line 441 (``text.replace('\\n', '\\\\n')``).
        """
        import lighttrain.cli.commands.diagnostics as _diag

        cfg = _minimal_cfg(tmp_path)

        class _NewlineTokenizer:
            def decode(self, ids):
                return "line1\nline2"

        class _DS:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"input_ids": [1, 2, 3]}

        class _DM:
            dataset = _DS()
            tokenizer = _NewlineTokenizer()

        bundle = {"data": _DM()}
        monkeypatch.setattr(_diag, "setup_run_from_config", lambda *a, **kw: bundle)
        res = _invoke("inspect-data", "-c", str(cfg), "--decoded")
        assert res.exit_code == 0, res.stdout
        # Rich table flattens; the escaped form must be present, not a raw newline
        assert "\\n" in res.stdout


# ===========================================================================
# Private helpers
# ===========================================================================


def _config_error_raiser(msg: str):
    """Return a callable that raises ConfigError with *msg*."""

    def _raiser(*args, **kwargs):
        raise ConfigError(msg)

    return _raiser


def _runtime_error_raiser(msg: str):
    """Return a callable that raises RuntimeError with *msg*."""

    def _raiser(*args, **kwargs):
        raise RuntimeError(msg)

    return _raiser


def _make_estimate_report(
    *,
    with_offload: bool,
    notes: list[str],
    param_bytes: int = 4096,
):
    """Build a minimal ``EstimateReport`` for monkeypatching."""
    from lighttrain.lab.estimate import EstimateReport, OffloadEstimate

    offload = None
    if with_offload:
        offload = OffloadEstimate(
            layers=8,
            resident_layers=2,
            layer_param_bytes=512 * 1024,
            recompute_us_per_layer=30.0,
            transfer_us_per_layer=80.0,
            recommended_mode="offload",
            pcie_bandwidth_used="12.3 GB/s",
        )
    return EstimateReport(
        trainable_params=1_000,
        all_params=1_000,
        trainable_ratio=1.0,
        param_bytes=param_bytes,
        grad_bytes=4096,
        optim_state_bytes=4096,
        activation_bytes_per_step=1024,
        total_bytes_per_step=2048,
        tokens_per_sec_estimate=500.0,
        model_name="test_model",
        optimizer_name="adamw",
        engine_name="base" if not with_offload else "layer_offload",
        notes=list(notes),
        offload=offload,
    )


def _patch_lab_estimate(monkeypatch, rpt):
    """Monkeypatch ``lighttrain.lab.estimate`` module-level ``estimate`` +
    ``report_to_dict`` so that ``estimate_cmd`` picks them up when it does
    ``from lighttrain.lab.estimate import estimate, report_to_dict``  inside
    the function body.
    """
    import importlib

    lab_est = importlib.import_module("lighttrain.lab.estimate")
    monkeypatch.setattr(lab_est, "estimate", lambda cfg: rpt)
    monkeypatch.setattr(lab_est, "report_to_dict", lambda r: {"model_name": r.model_name})


class _NoOpTrainer:
    def fit(self):
        pass


class _StepTrainer:
    """Fake trainer with a ctx.step attribute for the profile loop."""

    class _Ctx:
        step = 0

    ctx = _Ctx()

    def fit(self, steps=None):
        pass


class _NoDatasetModule:
    """Data module without a ``dataset`` attribute."""


class _FakeDataModule:
    """Configurable fake data module for inspect-data tests."""

    def __init__(
        self,
        n_samples: int = 3,
        with_tokenizer: bool = False,
        tokenizer_raises: bool = False,
    ):
        self.dataset = self._build_dataset(n_samples)
        if tokenizer_raises:
            self.tokenizer = _RaisingTokenizer()
        elif with_tokenizer:
            self.tokenizer = _HelloTokenizer()
        else:
            self.tokenizer = None

    @staticmethod
    def _build_dataset(n: int):
        class _DS:
            _n = n

            def __len__(self):
                return self._n

            def __getitem__(self, i):
                return {
                    "input_ids": [10, 20, 30, 40, 50],
                    "labels": [10, 20, -100, 40, 50],
                }

        return _DS()


class _HelloTokenizer:
    def decode(self, ids):
        return "hello world"


class _RaisingTokenizer:
    def decode(self, ids):
        raise ValueError("simulated decode error")


def _make_fake_profiler_module(raise_on_export: bool = False):
    """Return a minimal fake ``torch.profiler`` module replacement."""

    class _FakeKeyAverages:
        def table(self, sort_by="", row_limit=10):
            return "kernel_table\n"

    class _FakeProf:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def step(self):
            pass

        def export_chrome_trace(self, path: str):
            if raise_on_export:
                raise RuntimeError("disk full")

        def key_averages(self):
            return _FakeKeyAverages()

    class _PA:
        CPU = "cpu"
        CUDA = "cuda"

    fake_mod = types.ModuleType("torch.profiler")
    fake_mod.profile = lambda **kw: _FakeProf()  # type: ignore[attr-defined]
    fake_mod.schedule = lambda **kw: None  # type: ignore[attr-defined]
    fake_mod.ProfilerActivity = _PA  # type: ignore[attr-defined]
    return fake_mod
