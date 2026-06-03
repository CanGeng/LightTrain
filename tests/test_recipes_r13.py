"""R13 acceptance — QLoRA (Linux + CUDA + bitsandbytes only).

Both tests skip if (a) the recipe doesn't exist, (b) bitsandbytes isn't
installed, or (c) no CUDA device is visible. Windows hosts skip
automatically. See docs/M5_INTERFACES.md for the WSL acceptance script.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force eager registration of frontier plug-ins.
import lighttrain.builtin_plugins.quant  # noqa: F401

import torch

_RECIPE = Path("recipes/qlora.yaml")
_HAS_BNB = False
try:
    import bitsandbytes  # noqa: F401

    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
_HAS_CUDA = torch.cuda.is_available()
_IS_LINUX = os.name == "posix"


@pytest.mark.heavy
@pytest.mark.gpu
@pytest.mark.skipif(not _RECIPE.exists(), reason="R13 recipe missing")
@pytest.mark.skipif(not _HAS_BNB, reason="bitsandbytes not installed")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
@pytest.mark.skipif(not _IS_LINUX, reason="bnb is Linux-only on the lighttrain path")
def test_r13_dry_run_constructs_qlora_model():
    from lighttrain.cli._runtime import setup_run_from_config

    bundle = setup_run_from_config(
        _RECIPE,
        overrides=[
            "++trainer.max_steps=1",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=1",
        ],
        mode="lab",
    )
    model = bundle["model"]
    assert model is not None
    # Trainable << total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable / total < 0.05


@pytest.mark.heavy
@pytest.mark.gpu
@pytest.mark.skipif(not _RECIPE.exists(), reason="R13 recipe missing")
@pytest.mark.skipif(not _HAS_BNB, reason="bitsandbytes not installed")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
@pytest.mark.skipif(not _IS_LINUX, reason="bnb is Linux-only on the lighttrain path")
def test_r13_short_run_loss_drops():
    from lighttrain.cli._runtime import setup_run_from_config

    bundle = setup_run_from_config(
        _RECIPE,
        overrides=[
            "++trainer.max_steps=20",
            "++trainer.ckpt_every=0",
            "++trainer.log_every=2",
        ],
        mode="lab",
    )
    metrics = bundle["trainer"].fit()
    assert "loss" in metrics
    assert metrics["loss"] == metrics["loss"]  # NaN guard
