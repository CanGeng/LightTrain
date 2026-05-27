"""CPU-pinned layer-weight storage.

Holds the master copy of each layer's parameters on host RAM (optionally
``pin_memory=True`` so cuda non-blocking copies are async). The engine's
prefetch path calls :meth:`swap_in` to push a layer onto the device; the
swap-out path calls :meth:`swap_out` to copy any changes back and free
device buffers.

Single-GPU contract: this is **not** a CPU-offload optimizer; that's
:mod:`._optim_offload`. This module owns *weights* only.

On CPU-only runs (no CUDA available) ``swap_in`` / ``swap_out`` are
near-noops; we still copy params into a stash dict so subsequent
``swap_out`` writes can land somewhere and the unit tests can verify
contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CpuPinnedStorage:
    """Host-pinned storage for an entire ``LayeredView``'s layer params."""

    device: torch.device
    pin_memory: bool = True
    # name -> {param_name: cpu_tensor}
    stash: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def init_from_layer(self, name: str, layer: torch.nn.Module) -> None:
        """Take a CPU snapshot of every parameter in ``layer`` and stash it."""
        pinned: dict[str, torch.Tensor] = {}
        for pname, p in layer.named_parameters(recurse=True):
            with torch.no_grad():
                t = p.detach().to("cpu", copy=True)
                if self.pin_memory and torch.cuda.is_available():
                    try:
                        t = t.pin_memory()
                    except Exception:  # noqa: BLE001
                        pass
            pinned[pname] = t
        self.stash[name] = pinned

    def swap_in(self, name: str, layer: torch.nn.Module) -> None:
        """Move host-pinned weights back onto ``self.device`` for ``layer``."""
        if name not in self.stash:
            return
        host = self.stash[name]
        for pname, p in layer.named_parameters(recurse=True):
            src = host.get(pname)
            if src is None:
                continue
            with torch.no_grad():
                if p.data.device != self.device:
                    p.data = src.to(self.device, non_blocking=self.pin_memory).detach()
                else:
                    p.data.copy_(src.to(self.device, non_blocking=self.pin_memory))

    def swap_out(self, name: str, layer: torch.nn.Module) -> None:
        """Copy ``layer``'s current device weights back into host stash and
        free the device buffer. After this the layer has CPU weights only."""
        if name not in self.stash:
            self.init_from_layer(name, layer)
            return
        host = self.stash[name]
        for pname, p in layer.named_parameters(recurse=True):
            with torch.no_grad():
                cpu_t = p.detach().to("cpu")
                if pname in host:
                    host[pname].copy_(cpu_t)
                else:
                    host[pname] = cpu_t
                # Move param storage back to CPU so the device buffer can be freed.
                p.data = host[pname].to("cpu", copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "pin_memory": self.pin_memory,
            "names": list(self.stash.keys()),
        }


__all__ = ["CpuPinnedStorage"]
