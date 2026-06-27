"""image_cls recipe — the reference supervised-vision workflow, end-to-end.

Dry-run resolves the bundle (TinyCNNClassifier + ClassificationLoss on the
generic ``pretrain`` trainer); the heavy variant trains a short run on the
synthetic class-separable dataset and asserts the loss is finite and accuracy
rises. ``run_root`` is redirected to ``tmp_path`` so the suite never writes to
the repo ``runs/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lighttrain.cli._runtime import setup_run_from_config

RECIPE = Path("examples/references/recipes/image_cls.yaml")


@pytest.mark.skipif(not RECIPE.exists(), reason="image_cls recipe missing")
def test_image_cls_dry_run_resolves(tmp_path):
    bundle = setup_run_from_config(
        RECIPE,
        overrides=[
            f"++run_root={tmp_path}",
            "++trainer.max_steps=1",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=1",
        ],
        mode="lab",
    )
    assert bundle["model"].__class__.__name__ == "TinyCNNClassifier"
    # The recipe's ``loss:`` is wrapped in the objective seam; loss_family proves
    # the classification loss (not the LM next-token default) is wired.
    assert bundle["trainer"].objective.loss_family == "classification"


@pytest.mark.heavy
@pytest.mark.skipif(not RECIPE.exists(), reason="image_cls recipe missing")
def test_image_cls_short_run_learns(tmp_path):
    bundle = setup_run_from_config(
        RECIPE,
        overrides=[
            f"++run_root={tmp_path}",
            "++trainer.max_steps=60",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=10",
        ],
        mode="lab",
    )
    metrics = bundle["trainer"].fit()
    assert "loss" in metrics
    assert metrics["loss"] == metrics["loss"]  # NaN check
    # Separable toy data → the CNN learns fast within 60 steps.
    assert float(metrics["acc"]) >= 0.75
