"""Sample schema.

Includes an optional ``id`` field (stable join key for artifact alignment)
and ``modality_inputs`` (per-modality tensor / path payloads). All extended
fields are ``NotRequired`` — existing collators that only touch
``input_ids`` / ``attention_mask`` / ``labels`` are unaffected.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, NotRequired, TypedDict


class Sample(TypedDict, total=False):
    input_ids: list[int]
    attention_mask: NotRequired[list[int]]
    labels: NotRequired[list[int]]
    meta: NotRequired[dict[str, Any]]
    id: NotRequired[str]
    modality_inputs: NotRequired[dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]


def is_valid_sample(s: Mapping[str, Any]) -> bool:
    return isinstance(s, Mapping) and "input_ids" in s and len(s["input_ids"]) > 0


def derive_sample_id(sample: Mapping[str, Any], *, prefix: str = "s") -> str:
    """Derive a stable, content-addressed sample id.

    The id is the first 16 hex chars of sha256 over a canonical JSON of the
    head-64 token slice + sorted meta. Same content → same id across runs and
    machines, enabling deterministic artifact joins.
    """
    head = list(sample.get("input_ids", []))[:64]
    meta = sample.get("metadata") or sample.get("meta") or {}
    payload = {"head": head, "meta": _canonical(meta)}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _canonical(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


__all__ = ["Sample", "derive_sample_id", "is_valid_sample"]
