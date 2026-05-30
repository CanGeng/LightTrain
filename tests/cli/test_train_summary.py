"""`lighttrain train --output-summary` / `--eval` (v0.1.8 B1).

Replaces the wall-time accounting + summary.json aggregation that the mamba3
launcher did by hand. One row per `exp`, accumulating across invocations; a
failed run still writes a row with status=error.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lighttrain.cli._app import app

runner = CliRunner()

_CORPUS = "\n".join(f"sample line {i} with a few tokens here" for i in range(24))


def _recipe(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS + "\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        f"""
mode: lab
seed: 7
exp: sumtest
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
  total_steps: 2
  min_lr_ratio: 0.1
engine:
  name: standard
  mixed_precision: "no"
trainer:
  name: pretrain
  max_steps: 2
  val_every: 0
  ckpt_every: 1
  log_every: 1
logger:
  - name: jsonl
""",
        encoding="utf-8",
    )
    return recipe


def test_output_summary_accumulates_and_evals(tmp_path):
    recipe = _recipe(tmp_path)
    summary = tmp_path / "summary.json"

    res = runner.invoke(
        app,
        ["train", "-c", str(recipe), "exp=a", "--eval", "--eval-max-batches", "2",
         "--output-summary", str(summary)],
    )
    assert res.exit_code == 0, res.output

    res = runner.invoke(
        app, ["train", "-c", str(recipe), "exp=b", "--output-summary", str(summary)]
    )
    assert res.exit_code == 0, res.output

    rows = json.loads(summary.read_text())
    assert {r["exp"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["exp"] == "a")
    assert a["status"] == "ok"
    assert a["final_loss"] is not None
    assert a["eval_ppl"] is not None and a["eval_ppl"] > 0
    assert a["last_checkpoint"]
    assert a["wall_seconds"] >= 0


def test_rerun_same_exp_replaces_row(tmp_path):
    recipe = _recipe(tmp_path)
    summary = tmp_path / "summary.json"
    for _ in range(2):
        res = runner.invoke(
            app, ["train", "-c", str(recipe), "exp=x", "--output-summary", str(summary)]
        )
        assert res.exit_code == 0, res.output
    rows = json.loads(summary.read_text())
    assert len([r for r in rows if r["exp"] == "x"]) == 1
