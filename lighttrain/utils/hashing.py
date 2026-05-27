"""Stable hashing helpers."""

from __future__ import annotations

import hashlib


def short_hash(text: str | bytes, n: int = 8) -> str:
    """Return the first ``n`` hex characters of sha256(text)."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:n]


__all__ = ["short_hash"]
