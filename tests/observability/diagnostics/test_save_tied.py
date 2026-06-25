"""save_model_safe must persist models whose weights defeat save_model.

Plain ``safetensors.torch.save_model`` refuses to save aliased storage. Under
ZeRO-2 (DeepSpeed flattens params into one buffer) tiny_lm's tied
``tok_emb.weight`` / ``lm_head.weight`` become non-covering slices, which made
save_model raise and aborted crash-diagnostic snapshots on real-GPU runs.

The exact ZeRO aliasing needs DeepSpeed to reproduce, so the fallback contract
is pinned directly: on *any* RuntimeError from save_model, save_model_safe
clones the state dict (breaking the sharing) and persists every key.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file

import lighttrain.observability.diagnostics._save as _save_mod
from lighttrain.observability.diagnostics._save import save_model_safe


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying (shared storage)


def test_save_model_safe_writes_loadable_file_for_tied_model(tmp_path):
    """A tied model saves without raising and reloads with matching values."""
    model = _TiedModel()
    path = tmp_path / "ok.safetensors"
    save_model_safe(model, str(path))

    assert path.exists()
    loaded = load_file(str(path))
    # save_model keeps at least one of the tied names; whichever survives must
    # carry the correct weight (load_state(strict=False) reloads the tie).
    assert loaded, "no tensors persisted"
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, dict(model.state_dict())[name])


def test_save_model_safe_fallback_clones_every_key(tmp_path, monkeypatch):
    """When save_model raises, the fallback persists *all* state-dict keys."""

    def _boom(*_a, **_k):
        raise RuntimeError("simulated shared-storage rejection")

    monkeypatch.setattr(_save_mod, "save_model", _boom)

    model = _TiedModel()
    path = tmp_path / "fallback.safetensors"
    save_model_safe(model, str(path))

    loaded = load_file(str(path))
    # Fallback clones → both tied names present, values intact.
    assert set(loaded) == set(model.state_dict())
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, dict(model.state_dict())[name])
