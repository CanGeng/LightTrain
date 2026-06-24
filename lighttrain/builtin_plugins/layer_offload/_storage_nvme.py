"""NVMe layer storage.

Each layer is materialised as a ``safetensors`` file on disk. Reads and
writes are dispatched through a small thread pool (``concurrent.futures``)
so the engine can overlap I/O with compute on the same CUDA stream-of-
priority. On Linux a future revision can swap this thread pool for an
``io_uring``-backed backend (not yet implemented; falls back to thread pool).

Single-file safetensors per layer keeps shards independently rewritable
when only one layer's weights changed (e.g. after one optimizer step on
that layer). For tiny / debug workloads the same code path works without
``safetensors`` installed — we fall back to ``torch.save``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_log = logging.getLogger(__name__)


def _save_layer(path: Path, state: dict[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file

        save_file({k: v.detach().contiguous().cpu() for k, v in state.items()},
                  str(path))
    except ImportError:
        torch.save({k: v.detach().cpu() for k, v in state.items()}, str(path))


def _load_layer(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file

        return load_file(str(path))
    except ImportError:
        return torch.load(str(path), map_location="cpu", weights_only=True)


@dataclass
class NvmeStorage:
    """Layer-major NVMe storage (one safetensors file per layer)."""

    root: Path
    device: torch.device
    num_threads: int = 4
    layer_paths: dict[str, Path] = field(default_factory=dict)
    _pool: ThreadPoolExecutor | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(self.num_threads)))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            _log.warning(
                "layer_offload: NVMe storage close() during __del__ failed; thread pool may leak",
                exc_info=True,
            )

    # ---- contract mirror of CpuPinnedStorage -----------------------------

    def init_from_layer(self, name: str, layer: torch.nn.Module) -> None:
        path = self.root / f"{name}.safetensors"
        state = {pname: p.detach() for pname, p in layer.named_parameters(recurse=True)}
        # Sync write on init to make the file exist before any swap_in.
        _save_layer(path, state)
        self.layer_paths[name] = path

    def swap_in(self, name: str, layer: torch.nn.Module) -> None:
        path = self.layer_paths.get(name)
        if path is None or not path.exists():
            return
        # Read from disk on the pool thread; we still need the result before
        # forward, so we block on .result() — true async-prefetch happens at
        # the engine level via ``prefetch_async``.
        assert self._pool is not None
        fut = self._pool.submit(_load_layer, path)
        state = fut.result()
        for pname, p in layer.named_parameters(recurse=True):
            src = state.get(pname)
            if src is None:
                continue
            with torch.no_grad():
                if p.data.device != self.device:
                    p.data = src.to(self.device).detach()
                else:
                    p.data.copy_(src.to(self.device))

    def prefetch_async(self, name: str) -> Any:
        """Non-blocking read of ``<root>/<name>.safetensors`` into a future.

        Caller is responsible for ``.result()`` before forward needs the
        weights. Useful for layer i+prefetch overlap.
        """
        assert self._pool is not None
        path = self.layer_paths.get(name)
        if path is None:
            return None
        return self._pool.submit(_load_layer, path)

    def swap_out(self, name: str, layer: torch.nn.Module) -> None:
        path = self.layer_paths.get(name) or self.root / f"{name}.safetensors"
        state = {pname: p.detach().cpu() for pname, p in layer.named_parameters(recurse=True)}
        assert self._pool is not None
        fut = self._pool.submit(_save_layer, path, state)
        fut.result()
        self.layer_paths[name] = path
        # Free device memory by moving the live parameters to CPU view of the
        # state we just wrote (parameters now live on host).
        for pname, p in layer.named_parameters(recurse=True):
            with torch.no_grad():
                if pname in state:
                    p.data = state[pname]


class IOUringBackend:  # noqa: D401
    """Linux io_uring backend (not yet implemented)."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "io_uring NVMe backend is not yet implemented; use NvmeStorage "
            "(thread-pool fallback) for now."
        )


__all__ = ["NvmeStorage", "IOUringBackend"]
