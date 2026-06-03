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


def test_load_is_manifest_driven_not_file_existence(tmp_path):
    """Synthetic repro: re-saving the same step with a smaller state set must
    not let stale on-disk files leak into ``load()``.

    ``load()`` reads components strictly by ``manifest["files"]``; the previous
    ``optimizer.pt`` lingering on disk (a stale file from the first save) must
    be ignored because the second save's manifest no longer lists it.

    This is a defensive contract-hardening test — directly constructing the
    "same dir, heterogeneous manifest" state via the ``save()`` API. (No
    in-repo trainer path omits optimizer on a same-step re-save.)
    """
    mgr = CheckpointManager(tmp_path, keep_last_n=3)

    # First save WITH optimizer.
    mgr.save(
        step=5,
        state={"model": {"w": torch.zeros(2)}, "optimizer": {"OLD": 1}},
        kind="step",
    )
    target = mgr.ckpt_dir / "step_5"
    assert (target / "optimizer.pt").exists()

    # Second save of the SAME step WITHOUT optimizer.
    mgr.save(step=5, state={"model": {"w": torch.ones(2)}}, kind="step")

    import json
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert "optimizer" not in manifest["files"]

    loaded = mgr.load(target)
    # manifest-driven load ignores the stale optimizer.pt still on disk.
    assert "optimizer" not in loaded


def test_save_docstring_pins_not_crash_atomic_contract():
    """Pin the documented same-step-overwrite crash-atomicity limitation.

    Mirrors ``test_pin_checkpoint_manager_is_single_writer``: anyone changing
    the overwrite semantics (e.g. adding a generation-specific layout) must
    consciously update this docstring. We pin only short key phrases to avoid
    brittleness against wording polish.
    """
    save_doc = CheckpointManager.save.__doc__ or ""
    assert "not crash-atomic" in save_doc
    assert "same-step overwrite" in save_doc
