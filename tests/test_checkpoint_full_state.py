"""Checkpoint persistence covers data_module + full RNG (REVIEW #2)."""

from __future__ import annotations

import torch

from lighttrain.checkpoint.manager import CheckpointManager


def test_save_round_trips_data_module_and_full_rng(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last_n=3)

    state = {
        "model": {"w": torch.randn(2, 2)},
        "optimizer": {"step": 1, "param_groups": []},
        "scheduler": {"base_lrs": [1e-3]},
        # Full RNG: python / numpy / torch / cuda
        "rng": {
            "python": (3, tuple(range(625)), None),
            "numpy": ("MT19937", [0] * 624, 624, 0, 0.0),
            "torch": torch.get_rng_state(),
        },
        "trainer": {"step": 5},
        "data_module": {"sampler": {"epoch": 1, "consumed": 8}},
    }

    target = mgr.save(step=5, state=state, kind="step")

    # data_module file landed on disk and manifest references it
    assert (target / "data_module.pt").exists()
    import json
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["data_module"] == "data_module.pt"
    assert manifest["files"]["rng"] == "rng.pt"

    # Read back & confirm
    loaded = mgr.load(target)
    assert "data_module" in loaded and loaded["data_module"]["sampler"]["consumed"] == 8
    rng = loaded["rng"]
    assert set(rng.keys()) >= {"python", "numpy", "torch"}
