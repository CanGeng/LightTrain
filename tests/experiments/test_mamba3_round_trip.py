"""v0.1.8 model_profiles round-trip regression guard.

Runs the self-contained PROFILE-form transformer recipe through the new
`model:` selector + `model_profiles:` resolver and asserts the per-step losses
reproduce the reference recorded BEFORE the schema change (dict-form `model:`).

The comparison is honest, not circular: the reference in
`_fixtures/mamba3_transformer_5step.json` was captured with an equivalent
dict-form recipe before A1/A2 landed. A drift here means either the refactor
changed numerics (it must not — the resolved spec is byte-identical) or the
committed transformer profile was transcribed wrong.

Everything the test needs (recipe + corpus + reference losses) lives under
`tests/experiments/_fixtures/`, so the test does not depend on the gitignored
`experiments/` tree. GPU-gated: the reference is bf16/CUDA and bf16 reduction
order is hardware-dependent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

_FIX = Path(__file__).parent / "_fixtures"


@pytest.mark.gpu
@pytest.mark.heavy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="reference recorded on bf16/CUDA")
def test_transformer_profile_reproduces_reference_losses(tmp_path):
    from lighttrain.cli._runtime import setup_run_from_config

    ref = json.loads((_FIX / "mamba3_transformer_5step.json").read_text())
    recipe = _FIX / "mamba3_transformer_recipe.yaml"
    corpus = (_FIX / "corpus.txt").resolve()

    bundle = setup_run_from_config(
        recipe,
        overrides=[
            f"data.dataset.path={corpus}",
            f"run_root={tmp_path / 'runs'}",
        ],
    )
    trainer = bundle["trainer"]
    try:
        trainer.fit()
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()

    metrics = Path(bundle["run_dir"]) / "logs" / "metrics.jsonl"
    got = [
        json.loads(line)["loss"]
        for line in metrics.read_text().splitlines()
        if line.strip() and "loss" in json.loads(line)
    ]

    expected = ref["losses"]
    assert len(got) == len(expected), (got, expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert math.isclose(g, e, abs_tol=1e-2), f"step {i + 1}: got {g}, expected {e}"
