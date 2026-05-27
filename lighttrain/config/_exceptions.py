"""Config exception hierarchy."""

from __future__ import annotations


class ConfigError(Exception):
    """Base class for config-system errors."""


class ConfigSchemaError(ConfigError):
    """Pydantic validation error wrapped with config-loader context."""


class ConfigResolveError(ConfigError):
    """Raised when a ComponentSpec cannot be resolved to an object."""
