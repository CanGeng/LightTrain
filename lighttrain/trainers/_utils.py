"""Shared trainer utilities — device helpers used across all trainer families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..exceptions import BatchValidationError


def _device_of(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _move_batch(batch: Mapping[str, Any], device: torch.device | None) -> dict[str, Any]:
    if device is None:
        return dict(batch)
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def validate_batch(
    batch: Mapping[str, Any],
    required_keys: list[str],
    trainer_name: str,
) -> None:
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise BatchValidationError(trainer_name, missing, list(batch.keys()))


__all__ = ["_device_of", "_move_batch", "validate_batch"]
