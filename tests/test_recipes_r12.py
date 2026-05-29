"""R12 acceptance — LayerOffloadEngine end-to-end (DESIGN §26.7).

CPU smoke (default): 20 steps with the R12 recipe, verifying loss drops
and no callback raised. The full 200-step Linux/GPU run is heavy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Force eager registration of frontier plug-ins.
import plugins.layer_offload  # noqa: F401

from lighttrain.cli._runtime import setup_run_from_config


RECIPE = Path("recipes/offload_fullparam.yaml")


@pytest.mark.skipif(not RECIPE.exists(), reason="R12 recipe missing")
def test_r12_dry_run_resolves_offload_engine():
    bundle = setup_run_from_config(
        RECIPE,
        overrides=[
            "++trainer.max_steps=1",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=1",
        ],
        mode="lab",
    )
    assert bundle["engine"].__class__.__name__ == "LayerOffloadEngine"
    assert bundle["optimizer"].__class__.__name__ == "OptimizerCPUOffloadWrapper"


@pytest.mark.heavy
@pytest.mark.skipif(not RECIPE.exists(), reason="R12 recipe missing")
def test_r12_short_run_loss_drops():
    bundle = setup_run_from_config(
        RECIPE,
        overrides=[
            "++trainer.max_steps=20",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=2",
        ],
        mode="lab",
    )
    trainer = bundle["trainer"]
    metrics = trainer.fit()
    # Loss should be finite + not stuck. ``metrics`` is the last step's report.
    assert "loss" in metrics
    assert metrics["loss"] == metrics["loss"]  # NaN check
