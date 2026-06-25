"""Edge-case tests for ``lighttrain.lab.resume_verify``.

Pure helpers are pinned directly; the training-orchestration ``resume_verify``
is driven with a monkeypatched ``setup_run_from_config`` + ``_read_step_losses``
so the pass / length-mismatch / no-checkpoint paths run without real training.

* ``_read_step_losses``: missing file, blank/malformed lines, rows lacking
  step/loss;
* ``_losses_in_order``: sorted by step;
* ``_fit_and_close``: fit + logger.close, logger-None branch, close-on-error;
* ``render_report``: PASS/FAIL, note, per-step filter + boundary marker;
* ``resume_verify``: passing run, length-mismatch (note), missing checkpoint.
"""

from __future__ import annotations

import pytest

from lighttrain.lab import resume_verify as rv
from lighttrain.lab.resume_verify import (
    ResumeVerifyReport,
    _fit_and_close,
    _losses_in_order,
    _read_step_losses,
    render_report,
    resume_verify,
)

# ---------------------------------------------------------------------------
# _read_step_losses / _losses_in_order
# ---------------------------------------------------------------------------

def test_read_step_losses_missing_file_returns_empty(tmp_path):
    assert _read_step_losses(tmp_path) == {}


def test_read_step_losses_parses_and_skips_bad_rows(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        "\n".join(
            [
                "",                                  # blank
                "{not json",                         # malformed
                '{"step": 1, "loss": 0.5}',          # good
                '{"step": 2, "loss": 0.3}',          # good
                '{"step": 3}',                       # no loss → skipped
                '{"loss": 9.9}',                     # no step → skipped
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _read_step_losses(tmp_path) == {1: 0.5, 2: 0.3}


def test_losses_in_order_sorts_by_step():
    assert _losses_in_order({3: 0.3, 1: 0.1, 2: 0.2}) == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# _fit_and_close
# ---------------------------------------------------------------------------

class _StubTrainer:
    def __init__(self, ckpt="ckpt", *, raise_on_fit=False):
        self._ckpt = ckpt
        self._raise = raise_on_fit
        self.ckpt_manager = self
        self.loaded = None

    def fit(self):
        if self._raise:
            raise RuntimeError("fit boom")

    def latest(self):
        return self._ckpt

    def load_checkpoint(self, c):
        self.loaded = c


class _StubLogger:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_fit_and_close_calls_fit_then_closes_logger():
    logger = _StubLogger()
    _fit_and_close({"trainer": _StubTrainer(), "logger": logger})
    assert logger.closed is True


def test_fit_and_close_handles_none_logger():
    _fit_and_close({"trainer": _StubTrainer(), "logger": None})  # no raise


def test_fit_and_close_closes_logger_even_when_fit_raises():
    logger = _StubLogger()
    with pytest.raises(RuntimeError, match="fit boom"):
        _fit_and_close({"trainer": _StubTrainer(raise_on_fit=True), "logger": logger})
    assert logger.closed is True  # finally ran


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

def test_render_report_pass_without_note():
    rep = ResumeVerifyReport(
        phase1_steps=2, phase2_steps=2, tol=1e-2,
        single_pass_losses=[0.5, 0.5, 0.5, 0.5],
        resume_losses=[0.5, 0.5, 0.5, 0.5],
        per_step_delta=[0.05, 0.0, 0.001, 0.0],  # step1 delta>tol → shown via OR
        max_abs_delta=0.05, passed=True,
    )
    out = render_report(rep)
    assert "PASS" in out
    assert "← resume boundary" in out          # step 3 == boundary+1
    assert "step    1" in out                  # shown because delta > tol
    assert "note:" not in out


def test_render_report_fail_with_note():
    rep = ResumeVerifyReport(
        phase1_steps=1, phase2_steps=1, tol=1e-2,
        single_pass_losses=[0.1, 0.2],
        resume_losses=[0.1],
        per_step_delta=[0.0],
        max_abs_delta=0.0, passed=False, note="step-count mismatch: ...",
    )
    out = render_report(rep)
    assert "FAIL" in out
    assert "note: step-count mismatch" in out


# ---------------------------------------------------------------------------
# resume_verify orchestration (monkeypatched setup + loss reader)
# ---------------------------------------------------------------------------

def _patch_setup(monkeypatch, tmp_path, *, phase1_ckpt="ckpt"):
    """Patch setup_run_from_config to return stub bundles; the 2nd (phase-1)
    call's checkpoint is configurable to drive the no-checkpoint path."""
    calls = {"n": 0}

    def fake_setup(config, *, overrides=None, existing_run_dir=None):
        calls["n"] += 1
        rd = tmp_path / f"run{calls['n']}"
        rd.mkdir(exist_ok=True)
        ckpt = phase1_ckpt if calls["n"] == 2 else "ckpt"
        return {"trainer": _StubTrainer(ckpt=ckpt), "logger": None, "run_dir": rd}

    monkeypatch.setattr("lighttrain.cli._runtime.setup_run_from_config", fake_setup)
    return calls


def test_resume_verify_passes_when_trajectories_match(monkeypatch, tmp_path):
    _patch_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(rv, "_read_step_losses", lambda run_dir: {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
    report = resume_verify(config=object(), phase1_steps=2, phase2_steps=2)
    assert report.passed is True
    assert report.max_abs_delta == 0.0
    assert report.note == ""


def test_resume_verify_fails_on_length_mismatch(monkeypatch, tmp_path):
    _patch_setup(monkeypatch, tmp_path)
    # Only one logged step → lengths don't reach total=4.
    monkeypatch.setattr(rv, "_read_step_losses", lambda run_dir: {1: 0.5})
    report = resume_verify(config=object(), phase1_steps=2, phase2_steps=2)
    assert report.passed is False
    assert "step-count mismatch" in report.note


def test_resume_verify_raises_when_no_checkpoint(monkeypatch, tmp_path):
    _patch_setup(monkeypatch, tmp_path, phase1_ckpt=None)
    monkeypatch.setattr(rv, "_read_step_losses", lambda run_dir: {1: 0.5})
    with pytest.raises(RuntimeError, match="no checkpoint written"):
        resume_verify(config=object(), phase1_steps=2, phase2_steps=2)
