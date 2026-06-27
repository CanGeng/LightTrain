"""Activation management (``activations:`` block).

Three modes:

* ``recompute`` — layer is wrapped with ``torch.utils.checkpoint`` so
  forward activations are dropped and re-created during backward.
* ``offload`` — forward saves only the layer's first input, offloaded to
  pinned host RAM after the layer runs; backward pre-fetches it back to
  GPU and recomputes the layer forward to reconstruct intermediates for
  PyTorch's autograd. *GPU-only — constructing an ``_OffloadWrap`` on a
  CPU device raises ``RuntimeError`` (fail-loud, no silent fallback).*
* ``recompute_or_offload`` — pick per-layer based on a one-shot probe:
  compare a recompute vs offload micro-bench; cache the winner. (Current
  implementation falls back to recompute — left intentionally.)
"""

from __future__ import annotations

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
        policy:

        * ``recompute`` / ``recompute_or_offload`` →
          ``torch.utils.checkpoint`` (``use_reentrant=False`` — stable
          across torch>=2.0).
        * ``offload`` → ``_OffloadWrap`` (true host offload via a custom
          ``torch.autograd.Function``). Raises ``RuntimeError`` if the
          configured ``device`` is not CUDA — single-GPU-only contract.
        """
        if self.mode in ("recompute", "recompute_or_offload"):
            return _CheckpointWrap(layer_module)
        if self.mode == "offload":
            # Q8b: GPU-only — silent fallback would re-create the v0.2.3
            # "selectable-but-no-op" footgun (see experience.md #21).
            if self.device is None or self.device.type == "cpu":
                raise RuntimeError(
                    "activation mode=offload requires a CUDA device. "
                    "Use mode='recompute' or 'recompute_or_offload' on CPU."
                )
            return _OffloadWrap(layer_module)
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


class _ActivationOffloadFunction(torch.autograd.Function):
    """Save the layer's first input only; offload to pinned host after
    forward; backward pre-fetches to GPU and recomputes the layer forward
    so autograd can reconstruct intermediates.

    Only the first input ``x`` is offloaded & re-grad'd — other args/kwargs
    are passed through to recompute but receive no gradient. Memory wins
    come from not holding the layer's intermediate activations on GPU
    between forward and backward.
    """

    @staticmethod
    def forward(ctx, run_layer, x, *args, **kwargs):
        ctx.run_layer = run_layer
        ctx.args_for_recompute = args
        ctx.kwargs_for_recompute = kwargs
        ctx.x_device = x.device
        ctx.x_requires_grad = x.requires_grad

        y = run_layer(x, *args, **kwargs)

        if x.requires_grad:
            if not x.is_cuda:
                raise RuntimeError(
                    f"activation mode=offload requires CUDA tensors to offload; "
                    f"got device={x.device}. Use mode='recompute' on CPU."
                )
            ctx.x_pinned = x.detach().cpu().pin_memory()
        else:
            ctx.x_pinned = None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_pinned = ctx.x_pinned
        if x_pinned is None:
            # x didn't need grad → nothing to recompute. Return None for
            # every forward positional input.
            return (None, None, *([None] * len(ctx.args_for_recompute)))

        x_pre = x_pinned.to(ctx.x_device, non_blocking=True)

        # Recompute forward on the original input to build a fresh autograd
        # graph, then backward through it. ``.backward()`` accumulates the
        # parameter gradients into ``.grad`` AND computes ``x.grad`` (the
        # gradient we propagate to the parent graph). This is the standard
        # activation-checkpoint module-style recompute pattern, adapted to
        # offload: the host-pinned input is the only thing we retained.
        with torch.enable_grad():
            x = x_pre.requires_grad_(ctx.x_requires_grad)
            y = ctx.run_layer(
                x, *ctx.args_for_recompute, **ctx.kwargs_for_recompute
            )
            grad_y_local = (
                grad_y.expand_as(y) if y.shape != grad_y.shape else grad_y
            )
            y.backward(grad_y_local)
        grad_x = x.grad if ctx.x_requires_grad else None
        del x_pinned, x_pre, x
        return (None, grad_x, *([None] * len(ctx.args_for_recompute)))


class _OffloadWrap(torch.nn.Module):
    """Apply a layer with activation offload: forward stores the input at
    pinned-host; backward pre-fetches and recomputes the layer to feed
    autograd."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x, *args, **kwargs):
        return _ActivationOffloadFunction.apply(self.layer, x, *args, **kwargs)


__all__ = ["ActivationManager"]
