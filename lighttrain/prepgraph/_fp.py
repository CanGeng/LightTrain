"""Fingerprint utilities for PrepGraph — also feeds lineage content hashes.

A node's fingerprint determines cache identity. Same fingerprint ⇔ same on-disk
output. The function is identical to the lineage ``content_hash`` so it can be
re-exported from one place.

Algorithm::

    fp(node) = sha256(
        canonical_config(node.config)
        || code_version_for(node.__class__)
        || node.kind
        || SCHEMA_VERSION[node.schema_kind]
        || sorted(fp(u) for u in node.inputs)
    )

Display form is the first 16 hex chars; the full 64-char digest is what the
manifest stores.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from functools import lru_cache
from typing import Any, Iterable, Mapping

# Bumping a value here invalidates every downstream cache for the matching
# schema_kind. Keep entries terse and prefer additive changes.
SCHEMA_VERSION: dict[str, str] = {
    # PrepGraph row schemas
    "rows": "0.1",
    "tokenized_rows": "0.1",
    "packed_rows": "0.1",
    "validate_report": "0.1",
    "materialized": "0.1",
    "mixed_rows": "0.1",
    "chunked_rows": "0.1",
    # Lineage / artifact / config schemas
    "artifact_header": "0.4",
    "checkpoint_manifest": "0.4",
    "config": "0.4",
    "run_meta": "0.4",
    "frozen_step": "0.4",
}


def canonical_config(value: Any) -> Any:
    """Normalize a config payload so equivalent inputs hash identically.

    Rules:
      * sort dict keys recursively
      * drop ``None`` values
      * quantize floats to 1e-9 to absorb YAML round-trips
      * convert tuples to lists for JSON parity
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k in sorted(value):
            v = canonical_config(value[k])
            if v is None:
                continue
            out[str(k)] = v
        return out
    if isinstance(value, (list, tuple)):
        return [canonical_config(v) for v in value]
    if isinstance(value, float):
        return round(float(value), 9)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    # Fallback: stringified repr — last resort, but stable.
    return repr(value)


@lru_cache(maxsize=512)
def code_version_for(qualname_or_cls: Any) -> str:
    """Return a sha256 digest of a class's source, with module/mtime fallback.

    Cached because ``inspect.getsource`` repeatedly reads the source file.
    Pass either a class object or a fully-qualified ``module:Cls`` string.
    """
    try:
        if isinstance(qualname_or_cls, str):
            mod, _, name = qualname_or_cls.partition(":")
            import importlib

            obj = getattr(importlib.import_module(mod), name)
        else:
            obj = qualname_or_cls
        src = inspect.getsource(obj)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError, AttributeError, ImportError):
        # Fallback: module path + file mtime.
        try:
            obj = qualname_or_cls if not isinstance(qualname_or_cls, str) else None
            mod = inspect.getmodule(obj) if obj is not None else None
            file = getattr(mod, "__file__", None) if mod else None
            if file and os.path.exists(file):
                stamp = f"{file}:{os.path.getmtime(file)}"
            else:
                stamp = repr(qualname_or_cls)
        except Exception:  # noqa: BLE001
            stamp = repr(qualname_or_cls)
        return hashlib.sha256(stamp.encode("utf-8")).hexdigest()


def compose_fingerprint(
    *,
    kind: str,
    schema_kind: str,
    code_version: str,
    config: Mapping[str, Any],
    input_fps: Iterable[str],
) -> str:
    """Compose a node fingerprint. Returns full 64-char sha256 hex digest."""
    payload = {
        "kind": kind,
        "schema_version": SCHEMA_VERSION.get(schema_kind, "0.0"),
        "schema_kind": schema_kind,
        "code_version": code_version,
        "config": canonical_config(dict(config)),
        "inputs": sorted(str(x) for x in input_fps),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def short(fp: str, n: int = 16) -> str:
    return fp[:n]


__all__ = [
    "SCHEMA_VERSION",
    "canonical_config",
    "code_version_for",
    "compose_fingerprint",
    "short",
]
