"""Selective re-initialization.

Walks ``named_modules`` and, for each module whose name matches a regex,
re-initializes its parameters with a given distribution. Useful for
ablation studies that ask "what happens if I reset the last two blocks?".
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Pattern

import torch
import torch.nn as nn


def _compile(pattern: str | Iterable[str]) -> list[Pattern[str]]:
    if isinstance(pattern, str):
        return [re.compile(pattern)]
    return [re.compile(p) for p in pattern]


def _apply_dist(tensor: torch.Tensor, dist: Mapping[str, Any]) -> None:
    kind = str(dist.get("kind", "normal"))
    if kind == "normal":
        mean = float(dist.get("mean", 0.0))
        std = float(dist.get("std", 0.02))
        nn.init.normal_(tensor, mean=mean, std=std)
    elif kind == "zeros":
        nn.init.zeros_(tensor)
    elif kind == "ones":
        nn.init.ones_(tensor)
    elif kind == "xavier_uniform":
        if tensor.dim() >= 2:
            nn.init.xavier_uniform_(tensor)
        else:
            nn.init.zeros_(tensor)
    elif kind == "orthogonal":
        if tensor.dim() >= 2:
            nn.init.orthogonal_(tensor)
        else:
            nn.init.zeros_(tensor)
    elif kind == "uniform":
        low = float(dist.get("low", -0.02))
        high = float(dist.get("high", 0.02))
        nn.init.uniform_(tensor, a=low, b=high)
    else:
        raise ValueError(f"Unknown reinit dist kind: {kind!r}")


def reinit_module(
    model: nn.Module,
    pattern: str | Iterable[str],
    *,
    dist: Mapping[str, Any] | None = None,
) -> int:
    """Re-initialize parameters of every submodule whose dotted name matches
    ``pattern``. Returns the number of *modules* touched (not parameters).

    Default ``dist`` is ``{"kind": "normal", "std": 0.02}``. Biases inside
    a matched module are zeroed regardless of ``dist``.
    """
    patterns = _compile(pattern)
    d: Mapping[str, Any] = dict(dist) if dist is not None else {"kind": "normal", "std": 0.02}
    hits = 0
    with torch.no_grad():
        for name, sub in model.named_modules():
            if not name:
                continue
            if not any(p.search(name) for p in patterns):
                continue
            touched = False
            for pname, p in sub.named_parameters(recurse=False):
                if pname == "bias":
                    nn.init.zeros_(p)
                else:
                    _apply_dist(p, d)
                touched = True
            if touched:
                hits += 1
    return hits


__all__ = ["reinit_module"]
