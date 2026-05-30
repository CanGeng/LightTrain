"""Lock the `lighttrain eval` CLI end-to-end (Issue #10 regression guard).

The mamba3 experiment bypassed `lighttrain eval` by calling
`lighttrain.eval.metrics.perplexity` directly. v0.1.8 confirms the CLI works:
it builds the model from the recipe, optionally restores a checkpoint, and
emits a perplexity metric (falling back to the train loader when a recipe has
no dedicated val split — which is the case that made the bypass necessary).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lighttrain.cli._app import app

runner = CliRunner()

_CORPUS = "\n".join(f"the quick brown fox number {i} jumps over the lazy dog" for i in range(24))


def _recipe(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS + "\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        f"""
mode: lab
seed: 7
exp: eval_test
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


def test_eval_emits_perplexity_json(tmp_path):
    recipe = _recipe(tmp_path)
    out = tmp_path / "eval.json"
    res = runner.invoke(
        app, ["eval", "-c", str(recipe), "--json", str(out), "--max-batches", "2"]
    )
    assert res.exit_code == 0, res.output
    report = json.loads(out.read_text())
    assert "perplexity" in report["metrics"]
    ppl = report["metrics"]["perplexity"]
    assert ppl is not None and ppl > 0


def test_eval_with_checkpoint_loads(tmp_path):
    recipe = _recipe(tmp_path)
    # Train first so a checkpoint exists (ckpt_every=1).
    res = runner.invoke(app, ["train", "-c", str(recipe)])
    assert res.exit_code == 0, res.output
    runs = sorted((tmp_path / "runs" / "eval_test").glob("*/checkpoints/step_*"))
    assert runs, "no checkpoint produced"
    ckpt = runs[-1]
    res = runner.invoke(app, ["eval", "-c", str(recipe), "--checkpoint", str(ckpt)])
    assert res.exit_code == 0, res.output
    assert "loaded checkpoint" in res.output.lower() or "perplexity" in res.output.lower()
    # A loaded checkpoint must NOT trigger the untrained-weights warning.
    assert "untrained" not in res.output.lower()


def test_eval_without_checkpoint_warns_untrained(tmp_path):
    """Issue #6: eval with no checkpoint scores random init weights — it must
    say so loudly so the perplexity isn't mistaken for a trained result."""
    recipe = _recipe(tmp_path)
    res = runner.invoke(app, ["eval", "-c", str(recipe), "--max-batches", "1"])
    assert res.exit_code == 0, res.output
    assert "untrained" in res.output.lower()


def test_eval_does_not_mint_run_dir_under_run_root(tmp_path):
    """Issue #6: read-only eval should not accumulate empty run dirs under
    run_root. It runs in a temp dir that is cleaned up afterwards."""
    recipe = _recipe(tmp_path)
    res = runner.invoke(app, ["eval", "-c", str(recipe), "--max-batches", "1"])
    assert res.exit_code == 0, res.output
    run_root = tmp_path / "runs" / "eval_test"
    # No run directory minted by the eval invocation.
    assert not run_root.exists() or not any(run_root.iterdir())
