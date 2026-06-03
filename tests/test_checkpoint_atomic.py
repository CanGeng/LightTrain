"""Atomic checkpoint write protocol + Windows JSON pointer fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lighttrain.checkpoint.manager import CheckpointManager


def _toy_state(value: float = 1.0) -> dict:
    m = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return {
        "model": m.state_dict(),
        "optimizer": {"state": {}, "param_groups": [{"lr": 1e-3}]},
        "scheduler": {"last_step": 7},
        "trainer": {"step": 100, "epoch": 0},
        "rng": {"torch": torch.random.get_rng_state()},
    }


def test_save_writes_manifest_last(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    out = mgr.save(step=10, state=_toy_state())
    assert (out / "manifest.json").exists()
    assert (out / "model.safetensors").exists() or (out / "model.pt").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 10
    assert manifest["kind"] == "step"
    assert "model" in manifest["files"]


def test_load_round_trip(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    state = _toy_state(value=3.5)
    saved = mgr.save(step=5, state=state)

    out = mgr.load(saved)
    assert out["step"] == 5
    assert torch.equal(out["model"]["weight"], state["model"]["weight"])
    assert out["scheduler"]["last_step"] == 7


def test_partial_write_recognized_as_incomplete(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    saved = mgr.save(step=1, state=_toy_state())
    # Simulate crash: drop the manifest *after* writing the rest.
    (saved / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        mgr.load(saved)
    # And list_steps must skip it.
    assert mgr.list_steps() == []


def test_last_pointer_resolves_via_json_fallback(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(step=1, state=_toy_state())
    target = mgr.save(step=2, state=_toy_state())

    last_json = tmp_path / "checkpoints" / "last.json"
    assert last_json.exists()
    info = json.loads(last_json.read_text(encoding="utf-8"))
    assert info["target"] == target.name

    resolved = mgr._read_pointer("last")
    assert resolved is not None
    assert resolved.resolve() == target.resolve()


def test_best_pointer_records_metric(tmp_path: Path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(step=3, state=_toy_state(), kind="best", extras={"metric": "loss", "value": 0.42})
    info = json.loads((tmp_path / "checkpoints" / "best.json").read_text(encoding="utf-8"))
    assert info["target"] == "step_3"
    assert info["extras"]["metric"] == "loss"
    assert info["extras"]["value"] == 0.42


def test_prune_keeps_only_n_recent(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, keep_last_n=2)
    for i in range(1, 5):
        mgr.save(step=i, state=_toy_state())
    remaining = [p.name for p in mgr.list_steps()]
    assert remaining == ["step_3", "step_4"]


def test_prune_preserves_last_pointer(tmp_path: Path):
    """If `last` pointer disagrees with list_steps()[-1] (e.g. newer step's
    manifest is missing/corrupt), _prune must still protect what `last`
    points to so the symlink doesn't go dangling."""
    mgr = CheckpointManager(tmp_path, keep_last_n=1)
    # Save 3 valid steps; `last` points to step_30 after the loop.
    mgr.save(step=10, state=_toy_state())
    mgr.save(step=20, state=_toy_state())
    mgr.save(step=30, state=_toy_state())

    last_before = mgr._read_pointer("last")
    assert last_before is not None and last_before.name == "step_30"

    # Simulate corruption: step_30's manifest disappears, so list_steps() no
    # longer sees it; but the `last` pointer still references step_30.
    (last_before / "manifest.json").unlink()

    # list_steps() now returns [step_10, step_20]; latest()-based prune would
    # protect step_20 only, deleting step_10. The `last`-aware prune must keep
    # step_30 (pointed to by `last`) plus the most-recent valid step.
    mgr._prune()
    remaining = {p.name for p in mgr.list_steps()}
    # step_30 has no manifest so list_steps() excludes it; but the directory
    # must still exist on disk so the symlink isn't dangling.
    assert (mgr.ckpt_dir / "step_30").exists(), "last pointer target was pruned"


def test_save_load_tied_weights(tmp_path: Path):
    """CheckpointManager must not crash on tied weights and must preserve values."""
    from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM

    model = TinyCausalLM(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                         max_seq_len=32, tie_weights=True)
    assert model.lm_head.weight is model.tok_emb.weight, "pre-condition: weights are tied"

    mgr = CheckpointManager(tmp_path)
    # Must not raise safetensors shared-storage error.
    saved = mgr.save(step=0, state={"model": model.state_dict()})
    assert (saved / "manifest.json").exists()

    # Load into a fresh tied model and verify values + tied-weight semantics.
    fresh = TinyCausalLM(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                         max_seq_len=32, tie_weights=True)
    ckpt = mgr.load(saved)
    fresh.load_state_dict(ckpt["model"])

    assert torch.equal(fresh.lm_head.weight, model.tok_emb.weight), \
        "loaded lm_head.weight must equal original tok_emb.weight"
    assert fresh.lm_head.weight is fresh.tok_emb.weight, \
        "tied-weight invariant must survive round-trip into a tied model"
