"""Weight tying / untying.

``tie_weights(src, dst)`` makes ``dst.weight`` *share storage* with
``src.weight``; both are written in-place so the change is visible to the
optimizer immediately. ``untie_weights(path)`` is the reverse — it deep-
copies the current shared tensor into a fresh ``Parameter`` so subsequent
updates to either side don't bleed across.
"""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from ._replace import get_submodule


def _has_weight(m: nn.Module) -> bool:
    return hasattr(m, "weight") and isinstance(m.weight, nn.Parameter)


def tie_weights(model: nn.Module, src_path: str, dst_path: str) -> None:
    """Make ``dst.weight`` share storage with ``src.weight``."""
    src = get_submodule(model, src_path)
    dst = get_submodule(model, dst_path)
    if not _has_weight(src) or not _has_weight(dst):
        raise TypeError(
            f"Both src ({src_path}) and dst ({dst_path}) must expose a "
            f"``.weight`` Parameter."
        )
    src_w = cast(Tensor, src.weight)
    dst_w = cast(Tensor, dst.weight)
    if src_w.shape != dst_w.shape:
        raise ValueError(
            f"Shape mismatch: {src_path}.weight {tuple(src_w.shape)} vs "
            f"{dst_path}.weight {tuple(dst_w.shape)}"
        )
    dst.weight = src.weight  # shared storage; preserves Parameter-ness


def untie_weights(model: nn.Module, path: str) -> None:
    """Deep-copy the current weight tensor into a fresh ``Parameter`` so it
    no longer shares storage with anything else."""
    sub = get_submodule(model, path)
    if not _has_weight(sub):
        raise TypeError(f"{path!r} does not expose a ``.weight`` Parameter.")
    sub_w = cast(Tensor, sub.weight)
    cloned = sub_w.detach().clone()
    sub.weight = nn.Parameter(cloned, requires_grad=sub_w.requires_grad)


__all__ = ["tie_weights", "untie_weights"]
