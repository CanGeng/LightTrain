"""Artifact-store abstraction — header / errors / base class (core).

The concrete on-disk backends (safetensors-shards / memmap-fixed / parquet-rows)
+ the ``open_artifact_store`` factory are registered impls in
``lighttrain.builtin_plugins.artifacts.store`` (DESIGN §3.3); they subclass
``ArtifactStoreBase`` and carry an :class:`ArtifactHeader`. The structural
``ArtifactStoreProtocol`` lives in :mod:`lighttrain.protocols`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from lighttrain.data.prepgraph._fp import SCHEMA_VERSION


class ArtifactIncompleteError(RuntimeError):
    """Raised when a store dir lacks ``MANIFEST_COMPLETE.json``."""


class StaleArtifactError(RuntimeError):
    """Raised when on-disk header disagrees with expected header."""


_DEFAULT_HEADER_FIELDS = (
    "producer_signature",
    "model_id",
    "model_revision",
    "tokenizer_hash",
    "data_version",
    "preprocess_code_hash",
    "dtype",
    "field_schema",
    "framework_version",
    "schema_version",
)


@dataclass
class ArtifactHeader:
    """Header metadata persisted at ``<root>/header.json``.

    Fields default to empty strings rather than ``None`` to keep equality checks
    straightforward.
    """

    producer_signature: str = ""
    model_id: str = ""
    model_revision: str = ""
    tokenizer_hash: str = ""
    data_version: str = ""
    preprocess_code_hash: str = ""
    dtype: str = ""
    field_schema: dict[str, str] = field(default_factory=dict)
    framework_version: str = f"torch:{torch.__version__}"
    schema_version: str = SCHEMA_VERSION["artifact_header"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactHeader:
        kwargs: dict[str, Any] = {}
        for name, _fld in cls.__dataclass_fields__.items():
            if name in data:
                kwargs[name] = data[name]
        return cls(**kwargs)

    def disagreements(self, other: ArtifactHeader) -> list[str]:
        """Return list of field names where ``self`` and ``other`` differ on
        non-empty values. Empty-vs-anything is considered a 'don't care'."""
        bad: list[str] = []
        for f in _DEFAULT_HEADER_FIELDS:
            a, b = getattr(self, f), getattr(other, f)
            if a and b and a != b:
                bad.append(f)
        return bad


class ArtifactStoreBase:
    """Shared bookkeeping for the on-disk store backends (the core base class).

    Concrete backends (``lighttrain.builtin_plugins.artifacts.store``) subclass
    this and implement ``put / get / contains / iter_keys / finalize``
    (the structural ``ArtifactStoreProtocol`` in ``lighttrain.protocols``).
    """

    backend: str = "_base"

    def __init__(self, root: str | Path, *, header: ArtifactHeader | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = header or ArtifactHeader()
        self._finalized = (self.root / "MANIFEST_COMPLETE.json").exists()

    # ----- header IO -------------------------------------------------------

    def _write_header(self) -> None:
        (self.root / "header.json").write_text(
            json.dumps(self.header.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def _load_header(self) -> ArtifactHeader | None:
        p = self.root / "header.json"
        if not p.exists():
            return None
        try:
            return ArtifactHeader.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return None

    def _write_manifest(self, payload: Mapping[str, Any]) -> Path:
        body = dict(payload)
        body.setdefault("backend", self.backend)
        body.setdefault("finalized_ts", time.time())
        body.setdefault("header", self.header.to_dict())
        tmp = self.root / "MANIFEST_COMPLETE.json.tmp"
        tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.root / "MANIFEST_COMPLETE.json")
        return self.root / "MANIFEST_COMPLETE.json"


__all__ = [
    "ArtifactHeader",
    "ArtifactIncompleteError",
    "ArtifactStoreBase",
    "StaleArtifactError",
]
