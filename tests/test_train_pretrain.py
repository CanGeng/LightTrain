"""R1 acceptance smoke (heavy / opt-in).

Marked ``heavy`` so the default ``pytest`` run skips it. To execute::

    pytest -m heavy tests/test_train_pretrain.py

Asserts that loss decreases over a windowed average across a real CPU/GPU
training loop on the committed tiny_corpus fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.heavy


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE = REPO_ROOT / "recipes" / "pretrain_causal.yaml"


def _final_loss_window(jsonl_path: Path, fraction: float = 0.25) -> tuple[float, float]:
    """Return (first_window_mean, last_window_mean) loss across logged steps."""
    losses = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "scalar" and "loss" in rec:
            losses.append(float(rec["loss"]))
    n = len(losses)
    if n < 4:
        raise AssertionError(f"too few loss records to window ({n})")
    win = max(2, int(n * fraction))
    return sum(losses[:win]) / win, sum(losses[-win:]) / win


def test_r1_loss_decreases_after_200_steps(tmp_path: Path):
    pytest.importorskip("torch")
    from lighttrain.cli._runtime import setup_run_from_config

    overrides = [
        f"++run_root={(tmp_path / 'runs').as_posix()}",
        "++trainer.max_steps=200",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
        "++trainer.log_every=10",
    ]
    bundle = setup_run_from_config(RECIPE, overrides=overrides)
    bundle["trainer"].fit()
    bundle["logger"].close()

    jsonl = bundle["run_dir"] / "logs" / "metrics.jsonl"
    first, last = _final_loss_window(jsonl, fraction=0.25)
    # Generous threshold: tiny model + tiny corpus, but loss must clearly drop.
    assert last < first * 0.7, f"R1 smoke: loss did not decrease ({first=}, {last=})"
