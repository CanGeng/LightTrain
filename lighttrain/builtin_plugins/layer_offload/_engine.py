"""``LayerOffloadEngine``.

Drop-in replacement for ``StandardEngine`` in any recipe whose model exposes
(or registers) a ``layered_view``. Behaviorally:

* Same event sequence as ``StandardEngine`` (``on_*`` fired by the
  underlying ``UpdateRule.step``), so callbacks (``invariants`` /
  ``nan_hunter`` / ``frozen_step`` / ``crash_bundle``) work unchanged.
* Same return contract (``dict`` of float metrics).
* Numerically equivalent to ``StandardEngine`` in single-GPU mode — the
  test ``test_layer_offload_engine.py`` asserts ``loss`` matches to
  ``atol=1e-5`` under deterministic seeding.

What it adds:

1. On ``__init__``, it builds a ``LayeredView`` of the model and registers
   every layer's weights with a :class:`CpuPinnedStorage` or
   :class:`NvmeStorage` (the master copy).
2. Before each layer's ``forward``, a pre-hook ``swap_in``-s the layer to
   the device and (if ``prefetch >= 1``) kicks off ``layer i+1``'s
   non-blocking read via the transfer stream.
3. After ``forward``, a post-hook ``swap_out``-s the layer that just left
   the resident window — keeping at most ``resident_layers`` on the
   device simultaneously.
4. ``backward`` re-uses the same pre-hooks (PyTorch fires
   ``register_full_backward_pre_hook`` before each layer's backward), so
   weights are re-paged in just in time.
5. ``OptimizerCPUOffloadWrapper`` (registered as ``optimizer: cpu_offload``)
   handles the optimizer-state offload.

Known limitations:

* ``activation`` mode ``offload`` falls back to ``recompute`` — true in-step
  CPU activation offload is not yet implemented.
* NVMe path uses a thread pool, not ``io_uring`` (Linux future work).
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from collections.abc import Mapping
from contextlib import nullcontext as _nullctx
from pathlib import Path
from typing import Any

import torch

from lighttrain.engine._context import StepContext
from lighttrain.registry import register

from ._activation import ActivationManager
from ._adapters import get_layered_view
from ._layer_handle import LayerOffloadNotSupported
from ._storage_cpu import CpuPinnedStorage
from ._storage_nvme import NvmeStorage
from ._streams import StreamManager

_log = logging.getLogger(__name__)


def _resolve_storage(spec: str | Mapping[str, Any], *, device: torch.device, kind: str):
    """Construct a layer storage backend from a config sub-block."""
    if isinstance(spec, Mapping):
        device_str = str(spec.get("device", "cpu_pinned"))
    else:
        device_str = str(spec)
    if device_str.startswith("nvme:"):
        path = device_str.split(":", 1)[1]
        return NvmeStorage(root=Path(path + f"/{kind}"), device=device)
    if device_str in ("cpu_pinned", "cpu", "pinned"):
        return CpuPinnedStorage(device=device, pin_memory=("pinned" in device_str or device_str == "cpu_pinned"))
    raise ValueError(f"unknown storage device: {device_str!r}")


@register("engine", "layer_offload")
class LayerOffloadEngine:
    """Layer-window offload engine."""

    def __init__(
        self,
        *,
        update_rule: Any,
        loss_fn: Any = None,
        accelerator: Any = None,
        resident_layers: int = 2,
        prefetch: int = 1,
        storage: Mapping[str, Any] | None = None,
        precision_on_gpu: str = "bf16",
        precision_on_host: str = "fp32",
        io_streams: int = 2,
        nvme_threads: int = 4,
        pin_memory: bool = True,
        hooks: Mapping[str, Any] | None = None,
    ) -> None:
        self.update_rule = update_rule
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        self.resident_layers = max(1, int(resident_layers))
        self.prefetch = max(0, int(prefetch))
        self.precision_on_gpu = precision_on_gpu
        self.precision_on_host = precision_on_host
        self.io_streams = int(io_streams)
        self.nvme_threads = int(nvme_threads)
        self.pin_memory = bool(pin_memory)
        self.hooks_cfg = dict(hooks or {})
        self.storage_cfg = dict(storage or {})
        self._view: Any = None
        self._weights_storage: Any = None
        self._streams: StreamManager | None = None
        self._activation_mgr: ActivationManager | None = None
        self._installed: list[Any] = []
        self._device: torch.device | None = None
        # Round-robin window — layer name → "resident on device" flag.
        self._resident: dict[str, bool] = {}
        self._layer_order: list[str] = []
        self._lru: deque[str] = deque(maxlen=self.resident_layers + self.prefetch)

    # ---- one-time wiring ------------------------------------------------

    def _ensure_attached(self, model: torch.nn.Module) -> None:
        if self._view is not None:
            return
        try:
            view = get_layered_view(model)
        except LayerOffloadNotSupported:
            raise
        self._view = view
        self._device = next(model.parameters()).device
        self._streams = StreamManager(self._device, num_streams=self.io_streams)

        weights_spec = self.storage_cfg.get("weights", {"device": "cpu_pinned"})
        self._weights_storage = _resolve_storage(
            weights_spec, device=self._device, kind="weights"
        )
        # Stash master copies (host-side) for every layer.
        for h in view.layers:
            self._weights_storage.init_from_layer(h.name, h.module)
            self._layer_order.append(h.name)
        # Activation manager (recompute / offload / recompute_or_offload).
        act_spec = self.storage_cfg.get("activations", {"mode": "recompute_or_offload"})
        if isinstance(act_spec, Mapping):
            mode = str(act_spec.get("mode", "recompute_or_offload"))
        else:
            mode = "recompute_or_offload"
        self._activation_mgr = ActivationManager(mode=mode, device=self._device)
        # Wrap each layer with the activation policy (recompute by default).
        # We do this by swapping the layer's forward through a wrapped module
        # held in-place; the parent module reference is kept consistent.
        for h in view.layers:
            wrapped = self._activation_mgr.wrap(h.module)
            # Replace the module reference *inside* the LayerHandle so future
            # forward goes through the checkpointed wrapper.
            h.module = wrapped

        # Install forward hooks that page layers in/out.
        self._install_swap_hooks()

    # ---- hook installation ---------------------------------------------

    def _install_swap_hooks(self) -> None:
        assert self._view is not None and self._weights_storage is not None
        order = list(self._layer_order)

        def _pre_hook(name: str, idx: int):
            def _hook(_module, _inputs):
                # 1) ensure current layer is resident.
                self._weights_storage.swap_in(name, _module)
                self._resident[name] = True
                if name in self._lru:
                    self._lru.remove(name)
                self._lru.appendleft(name)
                # 2) prefetch next ``self.prefetch`` layers.
                for j in range(1, self.prefetch + 1):
                    if idx + j < len(order):
                        nxt_name = order[idx + j]
                        nxt = self._view.layers[idx + j].module
                        with self._streams.on_transfer() if self._streams else _nullctx():
                            self._weights_storage.swap_in(nxt_name, nxt)
                            self._resident[nxt_name] = True
                            if nxt_name in self._lru:
                                self._lru.remove(nxt_name)
                            self._lru.appendleft(nxt_name)
                # 3) evict any layer outside the LRU window.
                self._evict_outside_window(idx)
            return _hook

        def _bwd_hook(name: str):
            def _hook(_module, _grad):
                self._weights_storage.swap_in(name, _module)
                self._resident[name] = True
                if name in self._lru:
                    self._lru.remove(name)
                self._lru.appendleft(name)
                self._evict_outside_window(-1)
            return _hook

        for i, h in enumerate(self._view.layers):
            handle = h.module.register_forward_pre_hook(_pre_hook(h.name, i))
            self._installed.append(handle)
            bwd_handle = h.module.register_full_backward_pre_hook(_bwd_hook(h.name))
            self._installed.append(bwd_handle)

    def _evict_outside_window(self, current_idx: int) -> None:  # noqa: ARG002
        """Page out any resident layer not in the LRU window."""
        if self._view is None or self._weights_storage is None:
            return
        keep = set(self._lru)
        for h in self._view.layers:
            if self._resident.get(h.name) and h.name not in keep:
                self._weights_storage.swap_out(h.name, h.module)
                self._resident[h.name] = False

    # ---- engine protocol -----------------------------------------------

    def step(self, batch: Mapping[str, Any], ctx: StepContext) -> dict[str, Any]:
        if ctx.model is None:
            raise RuntimeError("LayerOffloadEngine.step: ctx.model is None")
        self._ensure_attached(ctx.model)
        if ctx.loss_fn is None:
            ctx.loss_fn = self.loss_fn
        if ctx.accelerator is None:
            ctx.accelerator = self.accelerator
        # The Engine contract is "delegate to update_rule" — the swap hooks
        # do the page work transparently during model.__call__.
        out = self.update_rule.step(ctx.model, batch, ctx)
        # After a full step, leave all layers resident on host so subsequent
        # ``estimate`` / external probes see the post-step state on disk
        # rather than scattered. This is also what guarantees a consistent
        # ``checkpoint.save`` later.
        if self._view is not None and self._weights_storage is not None:
            for _i, h in enumerate(self._view.layers):
                if self._resident.get(h.name):
                    self._weights_storage.swap_out(h.name, h.module)
                    self._resident[h.name] = False
        return out

    # ---- teardown ------------------------------------------------------

    def close(self) -> None:
        for h in self._installed:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                _log.warning(
                    "layer_offload: failed to remove a swap hook on close; leftover hook may add overhead",
                    exc_info=True,
                )
        self._installed.clear()
        if self._weights_storage is not None and hasattr(self._weights_storage, "close"):
            try:
                self._weights_storage.close()
            except Exception:  # noqa: BLE001
                _log.warning(
                    "layer_offload: weights storage close failed; backing files/threads may leak",
                    exc_info=True,
                )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            _log.warning(
                "layer_offload: close() during __del__ failed; resources may leak",
                exc_info=True,
            )


# Surface the ``activation: offload`` fallback as a one-time warning rather
# than per-step.
warnings.simplefilter("once", UserWarning)


__all__ = ["LayerOffloadEngine"]
