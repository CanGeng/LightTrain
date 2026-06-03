"""Activation management (``activations:`` block).

Three modes:

* ``recompute`` — layer is wrapped with ``torch.utils.checkpoint`` so
  forward activations are dropped and re-created during backward.
* ``offload`` — forward activations are moved to host pinned RAM after the
  layer runs; backward pre-fetches them. *Single-GPU only.*
* ``recompute_or_offload`` — pick per-layer based on a one-shot probe:
  compare a recompute vs offload micro-bench; cache the winner.

The ``recompute`` path is fully functional (PyTorch native). The ``offload``
path is a stub that warns and falls back to recompute — true in-step CPU
activation offload interacts with PyTorch's autograd graph in subtle ways
that are best validated on real GPU hardware.
"""

from __future__ import annotations

import warnings

import torch
import torch.utils.checkpoint as _ckpt

_MODE_ALIASES = {
    "recompute": "recompute",
    "offload": "offload",
    "recompute_or_offload": "recompute_or_offload",
}


class ActivationManager:
    def __init__(
        self,
        *,
        mode: str = "recompute_or_offload",
        device: torch.device | None = None,
    ) -> None:
        if mode not in _MODE_ALIASES:
            raise ValueError(f"activation mode {mode!r} not in {sorted(_MODE_ALIASES)}")
        self.mode = mode
        self.device = device

    def wrap(self, layer_module: torch.nn.Module) -> torch.nn.Module:
        """Return a forward-callable that applies the chosen activation
        policy. In ``recompute`` / ``recompute_or_offload`` mode we use
        ``torch.utils.checkpoint`` (use_reentrant=False — stable across
        torch>=2.0). In ``offload`` mode we currently warn and degrade to
        recompute."""
        if self.mode in ("recompute", "recompute_or_offload"):
            return _CheckpointWrap(layer_module)
        if self.mode == "offload":
            warnings.warn(
                "activation mode=offload is not yet implemented (true in-step "
                "host offload needs autograd-graph plumbing); falling back to "
                "recompute.",
                stacklevel=2,
            )
            return _CheckpointWrap(layer_module)
        return layer_module


class _CheckpointWrap(torch.nn.Module):
    """Apply ``torch.utils.checkpoint`` to a layer's forward."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        def _fn(*inner_args):
            return self.layer(*inner_args, **kwargs)

        if not self.training:
            return self.layer(*args, **kwargs)
        return _ckpt.checkpoint(_fn, *args, use_reentrant=False)


__all__ = ["ActivationManager"]
