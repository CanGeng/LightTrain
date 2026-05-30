"""`resume-verify` — strong single-pass-vs-resume parity (v0.1.8 C1).

v0.1.8 used this tool to *find* BUG-1: resume restored sampler state at epoch
granularity only, so a mid-epoch resume replayed a different batch order and
diverged one step after the boundary. **v0.1.9 fixes it** — the samplers now
resume from the trainer's authoritative consumed-batch count
(`ctx.batch_in_epoch` → `data_module.seek`), so mid-epoch resume is step-exact
and prefetch-independent. These tests now assert PASS in *both* the
epoch-aligned and mid-epoch cases (the fix), where v0.1.8 asserted FAIL for the
latter.

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
def test_midepoch_resume_is_now_step_exact(tmp_path):
    # phase1 = 3 batches (mid-epoch). v0.1.8 flagged this as FAIL (the data
    # stream restarted from a different position). v0.1.9 fixes BUG-1: the
    # sampler resumes from the authoritative consumed-batch count, so the
    # mid-epoch resume is now step-exact and the tool reports PASS.
    report = resume_verify(_recipe(tmp_path), phase1_steps=3, phase2_steps=3, tol=1e-2)
    # Pre-boundary steps match (state up to the checkpoint is faithful).
    assert report.per_step_delta[2] == pytest.approx(0.0, abs=1e-6)
    # Post-boundary now ALSO matches — the data position is restored exactly.
    assert report.per_step_delta[3] == pytest.approx(0.0, abs=1e-6)
    assert report.passed, f"mid-epoch resume should be exact now: max Δ={report.max_abs_delta}"


@pytest.mark.heavy
def test_cli_exit_code_reflects_pass(tmp_path):
    from typer.testing import CliRunner

    from lighttrain.cli._app import app

    recipe = _recipe(tmp_path)
    runner = CliRunner()
    # Epoch-aligned resume: PASS.
    ok = runner.invoke(
        app, ["resume-verify", "-c", str(recipe), "--phase1-steps", "4", "--phase2-steps", "2"]
    )
    assert ok.exit_code == 0, ok.output
    assert "PASS" in ok.output

    # Mid-epoch resume: now also PASS (BUG-1 fixed in v0.1.9).
    mid = runner.invoke(
        app, ["resume-verify", "-c", str(recipe), "--phase1-steps", "3", "--phase2-steps", "3"]
    )
    assert mid.exit_code == 0, mid.output
    assert "PASS" in mid.output


@pytest.mark.heavy
@pytest.mark.parametrize("num_workers", [0, 2])
def test_midepoch_resume_exact_under_prefetch(tmp_path, num_workers):
    """BUG-1 fix must hold with DataLoader prefetch (num_workers>0), not just
    at num_workers=0 — otherwise the 'fix' would only work single-process."""
    recipe = _recipe(tmp_path)
    report = resume_verify(
        recipe,
        phase1_steps=3,
        phase2_steps=3,
        tol=1e-2,
        overrides=[f"data.num_workers={num_workers}"],
    )
    assert report.passed, (
        f"mid-epoch resume must be exact at num_workers={num_workers}: "
        f"max Δ={report.max_abs_delta}"
    )
