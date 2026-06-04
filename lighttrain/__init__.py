"""lighttrain — single-GPU PyTorch LM training framework for research labs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .callbacks.base import EventBus, Signal
from .checkpoint.manager import CheckpointManager
from .config import (
    ComponentSpec,
    ConfigError,
    EngineSection,
    RootConfig,
    TrainerSection,
    dump_resolved,
    load_config,
    resolve,
)
from .distributed import ParallelContext
from .engine._context import StepContext
from .lineage import (
    LineageStore,
    SchemaMigrationError,
    content_hash,
    migrate,
    migrate_file,
    migrate_payload,
)
from .logging._bus import LoggerBus
from .models.extras import ExtraOutputSpec, ExtrasHookManager
from .registry import (
    KNOWN_CATEGORIES,
    NotRegisteredError,
    RegistryConflictError,
    RegistryError,
    UnknownCategoryError,
    categories,
    contains,
    get,
    list_entries,
    register,
    register_category,
    unregister,
)
from .trainers.base import Trainer

try:
    __version__ = _pkg_version("lighttrain")
except PackageNotFoundError:
    __version__ = "0.3.0"

__all__ = [
    "CheckpointManager",
    "ComponentSpec",
    "ConfigError",
    "EngineSection",
    "EventBus",
    "ExtraOutputSpec",
    "ExtrasHookManager",
    "KNOWN_CATEGORIES",
    "LineageStore",
    "LoggerBus",
    "NotRegisteredError",
    "ParallelContext",
    "RegistryConflictError",
    "RegistryError",
    "RootConfig",
    "SchemaMigrationError",
    "Signal",
    "StepContext",
    "Trainer",
    "TrainerSection",
    "UnknownCategoryError",
    "__version__",
    "categories",
    "content_hash",
    "contains",
    "dump_resolved",
    "get",
    "list_entries",
    "load_config",
    "migrate",
    "migrate_file",
    "migrate_payload",
    "register",
    "register_category",
    "resolve",
    "unregister",
]
