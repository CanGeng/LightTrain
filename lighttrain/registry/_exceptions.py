"""Registry exception hierarchy."""

from __future__ import annotations


class RegistryError(Exception):
    """Base class for registry errors."""


class RegistryConflictError(RegistryError):
    """Raised when registering a duplicate (category, name) without force=True."""


class UnknownCategoryError(RegistryError):
    """Raised when registering or querying an unknown registry category."""


class NotRegisteredError(RegistryError):
    """Raised when looking up a (category, name) that has no registration."""
