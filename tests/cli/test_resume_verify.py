"""`resume-verify` — strong single-pass-vs-resume parity (v0.1.8 C1).

These also *characterise* a real finding the tool surfaced: lighttrain's resume
restores sampler state at epoch granularity only (see
`lighttrain/data/core/samplers.py`), so resume is step-exact when the
checkpoint lands on an epoch boundary but diverges for a mid-epoch resume
(the data stream restarts from a different position). The tool correctly
reports PASS in the first case and FAIL in the second — that detection is the
point of building it.

Marked ``heavy`` (each call runs three short training loops). fp32 + sequential
sampler keeps the comparison deterministic on CPU or GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lighttrain.lab.resume_verify import resume_verify

# 8 maximally-distinct lines, batch_size 2 → 4 batches per epoch. High variance
# across lines so a changed batch order produces a clearly different loss — this
# makes the mid-epoch data-position gap observable above the tolerance.
_CORPUS = "\n".join(chr(ord("a") + i) * (20 + i) for i in range(8))


def _recipe(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS + "\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        f"""
mode: lab
seed: 7
exp: rv
run_root: {tmp_path / "runs"}
model: default
model_profiles:
  default:
    name: tiny_lm
    vocab_size: 264
    d_model: 32
    n_layers: 1
    n_heads: 2
    max_seq_len: 64
    dropout: 0.0
data:
  name: simple
  dataset:
    name: line_file_text
    path: {corpus}
    max_len: 32
  tokenizer:
    name: byte
  collator:
    name: causal_lm
    max_len: 32
  sampler:
    name: sequential
  batch_size: 2
  num_workers: 0
loss:
  name: cross_entropy
optim:
  name: adamw
  lr: 1.0e-3
scheduler:
  name: warmup_cosine
  warmup_steps: 1
  total_steps: 8
  min_lr_ratio: 0.1
engine:
  name: standard
  mixed_precision: "no"
trainer:
  name: pretrain
  max_steps: 8
  val_every: 0
  ckpt_every: 1
  log_every: 1
logger:
  - name: jsonl
""",
        encoding="utf-8",
    )
    return recipe


@pytest.mark.heavy
def test_epoch_aligned_resume_is_faithful(tmp_path):
    # phase1 = 4 batches = exactly one epoch → resume restores enough state.
    report = resume_verify(_recipe(tmp_path), phase1_steps=4, phase2_steps=2, tol=1e-2)
    assert len(report.single_pass_losses) == len(report.resume_losses) == 6
    # Steps up to and including the boundary are identical.
    assert report.per_step_delta[3] == pytest.approx(0.0, abs=1e-6)
    assert report.passed, f"epoch-aligned resume should match: max Δ={report.max_abs_delta}"


@pytest.mark.heavy
def test_midepoch_resume_gap_is_flagged(tmp_path):
    # phase1 = 3 batches (mid-epoch) → known data-position gap: the tool must
    # report FAIL with a non-zero post-boundary delta. This codifies the finding.
    report = resume_verify(_recipe(tmp_path), phase1_steps=3, phase2_steps=3, tol=1e-2)
    # Pre-boundary steps still match (state up to the checkpoint is faithful).
    assert report.per_step_delta[2] == pytest.approx(0.0, abs=1e-6)
    # Post-boundary diverges because the sampler restarts the data stream.
    assert not report.passed
    assert report.max_abs_delta > 1e-2


@pytest.mark.heavy
def test_cli_exit_code_reflects_pass(tmp_path):
    from typer.testing import CliRunner

    from lighttrain.cli._app import app

    recipe = _recipe(tmp_path)
    runner = CliRunner()
    ok = runner.invoke(
        app, ["resume-verify", "-c", str(recipe), "--phase1-steps", "4", "--phase2-steps", "2"]
    )
    assert ok.exit_code == 0, ok.output
    assert "PASS" in ok.output

    bad = runner.invoke(
        app, ["resume-verify", "-c", str(recipe), "--phase1-steps", "3", "--phase2-steps", "3"]
    )
    assert bad.exit_code == 1
    assert "FAIL" in bad.output
