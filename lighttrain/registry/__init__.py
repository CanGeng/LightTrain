"""Global multi-category registry."""

from __future__ import annotations

from ._core import (
    KNOWN_CATEGORIES,
    Registry,
    categories,
    contains,
    get,
    get_registry,
    list_entries,
    register,
    register_category,
    unregister,
)
from ._exceptions import (
    NotRegisteredError,
    RegistryConflictError,
    RegistryError,
    UnknownCategoryError,
)

__all__ = [
    "KNOWN_CATEGORIES",
    "NotRegisteredError",
    "Registry",
    "RegistryConflictError",
    "RegistryError",
    "UnknownCategoryError",
    "categories",
    "contains",
    "get",
    "get_registry",
    "list_entries",
    "register",
    "register_category",
    "unregister",
]
