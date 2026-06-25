"""Shared safetensors model-save helper for diagnostic snapshots.

``safetensors.torch.save_model`` refuses to persist tied/shared weights (e.g. a
model that ties ``tok_emb.weight`` to ``lm_head.weight``): its dedup pass raises
``RuntimeError`` when no single name covers the shared storage. Diagnostic
snapshots (frozen-step / crash / NaN-repro bundles) don't need the tying
metadata preserved, so on that failure we clone the state dict — breaking the
storage aliasing — and save the raw tensors instead.
"""

from __future__ import annotations

from typing import Any

from safetensors.torch import save_file, save_model


def save_model_safe(model: Any, path: str) -> None:
    """Save ``model`` to ``path`` as safetensors, tolerating tied weights."""
    try:
        save_model(model, path)
    except RuntimeError:
        # Tied/shared tensors defeat save_model's dedup. Clone to break the
        # storage sharing and persist the raw state dict (values preserved;
        # tying metadata dropped — fine for a diagnostic snapshot).
        state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        save_file(state_dict, path)


__all__ = ["save_model_safe"]
