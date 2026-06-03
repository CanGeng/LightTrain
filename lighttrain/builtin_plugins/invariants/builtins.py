"""Builtin invariant factories.

Each builtin is a function that takes a ``ctx`` namespace (the same keyword
namespace passed to ``evaluate_check``) and returns ``True`` when the
invariant **holds** (``False`` => violation). They are registered under the
``"invariant"`` registry category so recipes can reference them by short
name in addition to writing inline ``check:`` expressions.

Baseline set: ``loss_finite``, ``grad_norm_bounded``, ``lr_nonneg``,
``label_mask_nonzero``, ``param_count_stable``, ``dtype_stable``,
``batch_nonempty``.
"""

from __future__ import annotations

from typing import Any

import torch

from lighttrain.registry import register

# ----------------------------------------------------------------- registry
# Stash {invariant_name: callable(**ns) -> bool}. The InvariantsCallback
# pulls these in addition to user-declared ``check:`` strings.


@register("invariant", "loss_finite")
def loss_finite(*, loss: Any = None, **_: Any) -> bool:
    """All loss elements are finite (rules out NaN / Inf)."""
    if loss is None:
        return True
    if isinstance(loss, torch.Tensor):
        return bool(torch.isfinite(loss).all().item())
    try:
        return bool(loss == loss) and abs(float(loss)) != float("inf")
    except Exception:  # noqa: BLE001
        return True


@register("invariant", "grad_norm_bounded")
def grad_norm_bounded(*, metrics: Any = None, max: float = 1e3, **_: Any) -> bool:
    """``metrics['grad_norm'] < max`` (default 1000)."""
    if not metrics:
        return True
    gn = metrics.get("grad_norm")
    if gn is None:
        return True
    try:
        return float(gn) < float(max)
    except (TypeError, ValueError):
        return True


@register("invariant", "lr_nonneg")
def lr_nonneg(*, optimizer: Any = None, **_: Any) -> bool:
    """All optimizer param-group LRs are non-negative."""
    if optimizer is None:
        return True
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if not groups:
        return True
    for g in groups:
        try:
            if float(g.get("lr", 0.0)) < 0.0:
                return False
        except (TypeError, ValueError):
            continue
    return True


@register("invariant", "label_mask_nonzero")
def label_mask_nonzero(
    *,
    batch: Any = None,
    ignore_index: int = -100,
    **_: Any,
) -> bool:
    """At least one label position is *not* ``ignore_index``. Catches recipes
    where every label is masked out (training is effectively a no-op)."""
    if not isinstance(batch, dict):
        return True
    labels = batch.get("labels")
    if labels is None:
        return True
    if isinstance(labels, torch.Tensor):
        return bool((labels != int(ignore_index)).any().item())
    try:
        return any(int(x) != int(ignore_index) for x in labels)
    except Exception:  # noqa: BLE001
        return True


@register("invariant", "param_count_stable")
def param_count_stable(*, model: Any = None, metrics: Any = None, **_: Any) -> bool:
    """Number of trainable parameters does not change between steps. Trips on
    accidental ``requires_grad_(False)`` / ``parameters.append(...)`` calls."""
    if model is None or metrics is None:
        return True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prev = metrics.get("_invariant_param_count")
    metrics["_invariant_param_count"] = float(count)
    if prev is None:
        return True
    return int(prev) == int(count)


@register("invariant", "dtype_stable")
def dtype_stable(*, model: Any = None, metrics: Any = None, **_: Any) -> bool:
    """First parameter dtype is unchanged between steps."""
    if model is None or metrics is None:
        return True
    try:
        first = next(model.parameters())
    except StopIteration:
        return True
    dt_str = str(first.dtype)
    prev = metrics.get("_invariant_dtype")
    metrics["_invariant_dtype"] = dt_str  # type: ignore[assignment]
    if prev is None:
        return True
    return str(prev) == dt_str


@register("invariant", "batch_nonempty")
def batch_nonempty(*, batch: Any = None, **_: Any) -> bool:
    """At least one tensor in the batch has a non-zero leading dim."""
    if not isinstance(batch, dict):
        return True
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.numel() > 0 and v.shape[0] > 0:
            return True
    return False


__all__ = [
    "loss_finite",
    "grad_norm_bounded",
    "lr_nonneg",
    "label_mask_nonzero",
    "param_count_stable",
    "dtype_stable",
    "batch_nonempty",
]
