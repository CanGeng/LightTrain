"""Minimal model rebuild used by NaN repro.

This module must not import OmegaConf / Pydantic / Trainer / Callbacks /
Engine. Its job is to let a ~80-line ``repro.py`` script reconstruct a model
from a tiny JSON spec, load its state, and run forward — nothing else.

Usage::

    from lighttrain.minimal import build_minimal_model, load_state
    model = build_minimal_model({"name": "tiny_lm", "params": {...}})
    load_state(model, "model_state.safetensors")
    out = model(**batch)

The spec can also be a path to a JSON file or a dict already.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..registry import get as _registry_get


def build_minimal_model(spec: Mapping[str, Any] | str | Path) -> torch.nn.Module:
    """Reconstruct a model from a spec.

    ``spec`` may be a dict ``{"name": "<short>", "params": {...}}`` or a path
    to a JSON file containing such a dict. Short-name resolution goes through
    the ``"model"`` registry — the same path the full Trainer uses, so any
    model that runs under lighttrain can also be rebuilt by ``repro.py``.

    The repro script is allowed to ``import lighttrain.builtin_plugins.models.adapters`` first
    to populate the registry; this function does not do that itself because
    importing adapters pulls in transformers (heavy) and the repro script
    typically only needs ``tiny_lm`` for inline NaN reproduction.
    """
    data = _coerce_spec(spec)
    if "_target_" in data:
        # Fall back to dotted target if registry isn't pre-loaded.
        target = data["_target_"]
        params = data.get("params", {})
        cls = _import_target(str(target))
        return cls(**params)
    name = data.get("name")
    if not name:
        raise ValueError(f"model spec missing 'name' (or '_target_'): {data!r}")
    params = data.get("params", {})
    cls = _registry_get("model", str(name))
    return cls(**params)


def load_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = False,
) -> torch.nn.Module:
    """Load a safetensors / .pt state dict into ``model`` (in place)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as _load_file

        state = _load_file(str(path))
    else:
        state = torch.load(str(path), weights_only=True)
    model.load_state_dict(state, strict=strict)
    return model


def dump_spec(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe spec dict for later ``build_minimal_model`` calls."""
    return {
        "name": str(name),
        "params": {k: _jsonable(v) for k, v in params.items()},
    }


# ---------------------------------------------------------------- internals


def _coerce_spec(spec: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return dict(spec)
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"model spec file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _import_target(target: str) -> Any:
    if ":" in target:
        mod, _, attr = target.partition(":")
    else:
        mod, _, attr = target.rpartition(".")
    import importlib

    return getattr(importlib.import_module(mod), attr)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


__all__ = ["build_minimal_model", "dump_spec", "load_state"]
