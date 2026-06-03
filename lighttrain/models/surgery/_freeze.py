"""Parameter freezing helpers.

These are pure utilities — they don't know about Registry, configs, or the
training loop. Callers that want declarative freezing via YAML can use the
``param_groups`` DSL on the optimizer (which sets ``requires_grad=False``
when ``freeze: true`` matches). Surgery is the imperative path for the same
underlying primitive: walk ``named_parameters``, set ``requires_grad`` by
regex, count how many were touched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from re import Pattern

import torch.nn as nn


def _compile_patterns(pattern: str | Iterable[str]) -> list[Pattern[str]]:
    if isinstance(pattern, str):
        return [re.compile(pattern)]
    return [re.compile(p) for p in pattern]


def freeze_modules(model: nn.Module, pattern: str | Iterable[str]) -> int:
    """Set ``requires_grad=False`` on every named parameter matching ``pattern``.

    ``pattern`` is a regex (or iterable of regexes); a parameter matches if
    any regex matches via ``re.search`` against its ``named_parameters`` name.
    Returns the count of parameters newly frozen.
    """
    patterns = _compile_patterns(pattern)
    frozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(p.search(name) for p in patterns):
            param.requires_grad = False
            frozen += 1
    return frozen


def unfreeze_modules(model: nn.Module, pattern: str | Iterable[str]) -> int:
    """Inverse of :func:`freeze_modules`: set ``requires_grad=True``."""
    patterns = _compile_patterns(pattern)
    unfrozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        if any(p.search(name) for p in patterns):
            param.requires_grad = True
            unfrozen += 1
    return unfrozen


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_params, all_params)`` summed over ``numel()``."""
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


__all__ = ["freeze_modules", "unfreeze_modules", "count_trainable"]
