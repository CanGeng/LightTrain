"""I/O probes used by ``lighttrain estimate``.

Provides one function — :func:`probe_layer_bandwidth` — that the lab
``estimate`` command calls in ``offload`` mode to report a layer-level
"recompute vs transfer" breakdown. The probe is intentionally cheap (one
warmup + a 3-iteration timer); finer profiling can come later.
"""

from __future__ import annotations

import time

import torch

from ._adapters import get_layered_view


def probe_layer_bandwidth(model: torch.nn.Module) -> tuple[float, float, int, int]:
    """Return ``(recompute_us_per_layer, transfer_us_per_layer,
    layer_param_bytes, layer_count)``.

    * recompute time = wall-clock for one layer's forward (no_grad).
    * transfer time  = wall-clock for one ``layer.cpu()`` round-trip.
    * layer_param_bytes = byte count for the layer's parameters.
    * layer_count = number of layers in the layered_view.

    All four numbers are coarse; the goal is "is recompute >> transfer or
    vice versa?", which is enough for the user to pick ``resident_layers``.
    """
    view = get_layered_view(model)
    if not view.layers:
        return 0.0, 0.0, 0, 0
    layer = view.layers[0].module
    layer_param_bytes = sum(
        p.numel() * p.element_size() for p in layer.parameters()
    )

    # Build a representative input for one layer. tiny_lm blocks consume
    # ``(x, mask)`` returning ``(x, attn_probs?)``; HF blocks vary widely.
    # We use a generic strategy: read the first nn.Linear's in_features as
    # the embedding dim, build a (1, 4, dim) tensor.
    dim = 0
    for m in layer.modules():
        if isinstance(m, torch.nn.Linear):
            dim = m.in_features
            break
    if dim == 0:
        dim = 256
    x = torch.zeros(1, 4, dim)
    # Move layer to CPU so transfer timing is a real D2H/H2D (or a noop on CPU).
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        layer = layer.cuda()
        x = x.cuda()
    # 1) recompute
    layer.eval()
    with torch.no_grad():
        try:
            _ = _try_layer(layer, x)
            t0 = time.perf_counter()
            for _ in range(3):
                _ = _try_layer(layer, x)
            recompute_us = ((time.perf_counter() - t0) / 3) * 1e6
        except Exception:  # noqa: BLE001
            recompute_us = 0.0
    # 2) transfer
    try:
        cpu_layer = layer.cpu()
        t0 = time.perf_counter()
        for _ in range(3):
            if has_cuda:
                cpu_layer = cpu_layer.cuda()
                cpu_layer = cpu_layer.cpu()
        transfer_us = ((time.perf_counter() - t0) / 3) * 1e6
    except Exception:  # noqa: BLE001
        transfer_us = 0.0
    return float(recompute_us), float(transfer_us), int(layer_param_bytes), int(len(view.layers))


def _try_layer(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Heuristic invocation that works for tiny_lm blocks and HF blocks
    that accept a positional hidden_states tensor."""
    try:
        out = layer(x, None)
    except TypeError:
        out = layer(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


__all__ = ["probe_layer_bandwidth"]
