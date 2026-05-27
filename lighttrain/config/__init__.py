"""Configuration system — OmegaConf + Pydantic v2."""

from __future__ import annotations

from ._exceptions import ConfigError, ConfigResolveError, ConfigSchemaError
from ._loader import dump_resolved, load_config
from ._resolver import resolve
from ._schema import (
    ComponentSpec,
    EngineSection,
    GradSyncConfig,
    ParallelSection,
    PipelineConfig,
    RootConfig,
    TPConfig,
    TrainerSection,
)

__all__ = [
    "ComponentSpec",
    "ConfigError",
    "ConfigResolveError",
    "ConfigSchemaError",
    "EngineSection",
    "GradSyncConfig",
    "ParallelSection",
    "PipelineConfig",
    "RootConfig",
    "TPConfig",
    "TrainerSection",
    "dump_resolved",
    "load_config",
    "resolve",
]
