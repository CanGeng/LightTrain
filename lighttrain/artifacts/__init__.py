"""Artifact-store abstraction (core).

Header / errors / base class live in :mod:`lighttrain.artifacts.base`; the
structural protocol is in :mod:`lighttrain.protocols`. Concrete store backends
+ producers + the joined dataset are registered impls in
``lighttrain.builtin_plugins.artifacts`` (DESIGN §3.3).
"""

from __future__ import annotations

from lighttrain.protocols import ArtifactStoreProtocol

from .base import (
    ArtifactHeader,
    ArtifactIncompleteError,
    ArtifactStoreBase,
    StaleArtifactError,
)

__all__ = [
    "ArtifactHeader",
    "ArtifactIncompleteError",
    "ArtifactStoreBase",
    "ArtifactStoreProtocol",
    "StaleArtifactError",
]
