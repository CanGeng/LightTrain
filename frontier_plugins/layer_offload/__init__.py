"""LayerOffload plug-in.

Importing this package registers:

* ``@register("engine", "layer_offload")`` — :class:`LayerOffloadEngine`
* ``@register("optimizer", "cpu_offload")`` — :class:`OptimizerCPUOffloadWrapper`

…and exposes:

* :class:`LayerHandle` / :class:`LayeredView` protocols and the
  ``LayerOffloadNotSupported`` exception.
* :class:`CpuPinnedStorage` / :class:`NvmeStorage` storage backends.
* :func:`get_layered_view` / :func:`register_layered_view` for custom
  architectures to opt in.

Used by ``lighttrain estimate`` to attach an :class:`OffloadEstimate`
breakdown when the recipe's ``engine.name == 'layer_offload'``.
"""

from __future__ import annotations

from ._activation import ActivationManager
from ._adapters import get_layered_view, register_layered_view
from ._engine import LayerOffloadEngine
from ._io import probe_layer_bandwidth
from ._layer_handle import LayerHandle, LayeredView, LayerOffloadNotSupported
from ._optim_offload import OptimizerCPUOffloadWrapper
from ._storage_cpu import CpuPinnedStorage
from ._storage_nvme import IOUringBackend, NvmeStorage
from ._streams import StreamManager

__all__ = [
    "LayerOffloadEngine",
    "OptimizerCPUOffloadWrapper",
    "LayerHandle",
    "LayeredView",
    "LayerOffloadNotSupported",
    "CpuPinnedStorage",
    "NvmeStorage",
    "IOUringBackend",
    "ActivationManager",
    "StreamManager",
    "get_layered_view",
    "register_layered_view",
    "probe_layer_bandwidth",
]
