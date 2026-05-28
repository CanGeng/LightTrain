"""End-to-end: CLI ``freeze-step`` + ``replay-step`` (M4 — Phase J)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app


# Minimal recipe that runs on CPU with byte tokenizer + tiny_lm + the
# corpus fixture that all other recipes already use.
_RECIPE = """\
mode: lab
seed: 1
exp: m4_freeze_replay
run_root: __ROOT__

model:
  name: tiny_lm
  vocab_size: 260
  d_model: 16
  n_layers: 1
  n_heads: 2
  max_seq_len: 32

data:
  name: simple
  dataset:
    name: line_file_text
    path: __CORPUS__
    max_len: 32
  tokenizer: {name: byte}
  collator: {name: causal_lm, max_len: 32}
  sampler: {name: shuffle, seed: 1}
  batch_size: 2

loss: {name: cross_entropy}
optim: {name: adamw, lr: 1.0e-3, betas: [0.9, 0.95], weight_decay: 0.0}
scheduler: {name: warmup_cosine, warmup_steps: 1, total_steps: 2}

engine: {name: standard, mixed_precision: 'no'}

trainer:
  name: pretrain
  max_steps: 2
  val_every: 0
  ckpt_every: 1
  log_every: 100
  grad_clip: 1.0

callbacks:
  - {name: frozen_step, every: 1}

logger:
  - {name: console, log_every: 100}
"""


def _write_recipe(tmp_path: Path) -> Path:
    corpus = (
        Path(__file__).resolve().parent / "fixtures" / "tiny_corpus.txt"
    )
    assert corpus.exists()
    cfg = tmp_path / "cfg.yaml"
    body = (
        _RECIPE.replace("__ROOT__", str(tmp_path / "runs"))
        .replace("__CORPUS__", str(corpus))
    )
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_train_emits_frozen_step_then_replay_runs(tmp_path):
    cfg = _write_recipe(tmp_path)
    runner = CliRunner()
    res = runner.invoke(app, ["train", "-c", str(cfg)])
    assert res.exit_code == 0, res.stdout
    runs_root = tmp_path / "runs" / "m4_freeze_replay"
    run_dirs = list(runs_root.iterdir())
    assert run_dirs, "no run dir created"
    run = run_dirs[0]
    zips = sorted((run / "frozen_steps").glob("*.zip"))
    assert zips, "frozen_step callback should have emitted at least one bundle"

    res2 = runner.invoke(app, ["replay-step", str(zips[0])])
    assert res2.exit_code == 0, res2.stdout
    assert "replay step" in res2.stdout
