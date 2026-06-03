"""Artifact subsystem.

Re-exports the producer / store / joined-dataset / dynamic-producer surfaces.
Concrete classes register against the ``artifact_producer`` / ``artifact_store``
/ ``dataset`` / ``callback`` registry categories on import.
"""

from __future__ import annotations

from .dynamic_producer import DynamicArtifactCallback
from .joined_dataset import ArtifactJoinedDataset, drop_none_collator
from .producer import (
    ArtifactProducerProtocol,
    ModelForwardProducer,
    run_artifact_production,
)
from .store import (
    ArtifactHeader,
    ArtifactIncompleteError,
    ArtifactStoreProtocol,
    MemmapFixedStore,
    ParquetRowStore,
    SafetensorsShardStore,
    StaleArtifactError,
    open_artifact_store,
)

__all__ = [
    "ArtifactHeader",
    "ArtifactIncompleteError",
    "ArtifactJoinedDataset",
    "ArtifactProducerProtocol",
    "ArtifactStoreProtocol",
    "DynamicArtifactCallback",
    "MemmapFixedStore",
    "ModelForwardProducer",
    "ParquetRowStore",
    "SafetensorsShardStore",
    "StaleArtifactError",
    "drop_none_collator",
    "open_artifact_store",
    "run_artifact_production",
]
