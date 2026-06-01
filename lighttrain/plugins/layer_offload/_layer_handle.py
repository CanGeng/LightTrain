"""LayerHandle + LayeredView protocols.

A ``LayeredView`` is a flat decomposition of a model into:

* ``embed`` — token embedding + positional embedding (always resident on GPU)
* ``layers`` — the rotating transformer / SSM / UNet blocks
* ``head``  — the LM head / output projection (always resident on GPU)

Each layer is exposed as a ``LayerHandle`` that carries (a) a stable string
name for indexing into storage backends, (b) the ``nn.Module``, and (c) the
in/out contract so callers know what tensors the layer reads and writes.

Bases that can't be sliced (because the forward fuses cross-layer ops, e.g.
some FlashAttention paths or stateful kernels) should raise
:class:`LayerOffloadNotSupported`. The engine surfaces that error to the
user; we do not attempt to layer-offload a base that says it doesn't
support it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch.nn as nn


class LayerOffloadNotSupported(RuntimeError):
    """Raised when ``layered_view()`` can't slice the model — engine refuses
    to enable layer_offload for this architecture."""


@dataclass
class LayerHandle:
    """A single offload-able layer."""

    name: str
    module: nn.Module
    inputs_contract: tuple[str, ...] = ("hidden_states",)
    outputs_contract: tuple[str, ...] = ("hidden_states",)


@runtime_checkable
class LayeredView(Protocol):
    """Structural view of a model used by ``LayerOffloadEngine``."""

    embed: nn.Module
    layers: list[LayerHandle]
    head: nn.Module


@dataclass
class _DefaultLayeredView:
    embed: nn.Module
    layers: list[LayerHandle]
    head: nn.Module


__all__ = [
    "LayerHandle",
    "LayeredView",
    "LayerOffloadNotSupported",
    "_DefaultLayeredView",
]
