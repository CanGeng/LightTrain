"""Registry core — global multi-category registry.

A single process-wide :class:`Registry` instance holds named entries grouped by
category (e.g. ``model``, ``loss``, ``callback``). Components are added via the
:func:`register` decorator (or the equivalent function call) and looked up by
short name from configuration files.

Categories are pre-declared in :data:`KNOWN_CATEGORIES`. Plugins that need a
fresh category may call :func:`register_category` before adding entries; this
keeps the registry strict-by-default while still extensible.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ._exceptions import (
    NotRegisteredError,
    RegistryConflictError,
    UnknownCategoryError,
)

KNOWN_CATEGORIES: tuple[str, ...] = (
    # Core
    "model",
    "loss",
    "optimizer",
    "scheduler",
    "dataset",
    "processor",
    "collator",
    "sampler",
    # Training orchestration
    "callback",
    "metric",
    "logger",
    "trainer",
    "engine",
    "update_rule",
    "architecture",
    "objective",
    # Frontier
    "generation_strategy",
    "judge",
    "environment",
    "retriever",
    "chunker",
    "probe",
    # Artifact-related
    "artifact_producer",
    "artifact_store",
    "prep_node",
    # Data plumbing
    "tokenizer",
    "data_module",
    # Failure-first diagnostics
    "invariant",
    # RL backends
    "rl_backend",
    # Distributed strategies (implementations live in frontier_plugins/distributed/)
    "grad_sync_strategy",
    "model_parallel_strategy",
    "pipeline_schedule",
)


class Registry:
    """Multi-category registry. Use the module-level singleton via the public API."""

    def __init__(self, categories: Iterable[str] = KNOWN_CATEGORIES) -> None:
        self._categories: set[str] = set(categories)
        self._store: dict[str, dict[str, Any]] = {c: {} for c in self._categories}

    def register_category(self, category: str) -> None:
        if category in self._categories:
            return
        self._categories.add(category)
        self._store[category] = {}

    def categories(self) -> list[str]:
        return sorted(self._categories)

    def _check_category(self, category: str) -> None:
        if category not in self._categories:
            raise UnknownCategoryError(
                f"Unknown registry category: {category!r}. "
                f"Known: {sorted(self._categories)}. "
                f"Use register_category({category!r}) first if intentional."
            )

    def register(
        self,
        category: str,
        name: str,
        obj: Any | None = None,
        *,
        force: bool = False,
    ) -> Callable[[Any], Any] | Any:
        """Register ``obj`` under (category, name).

        Usable as a decorator (``obj=None``) or function call. Duplicate names
        raise :class:`RegistryConflictError` unless ``force=True``.
        """
        self._check_category(category)

        def _do_register(target: Any) -> Any:
            bucket = self._store[category]
            if name in bucket and not force:
                existing = bucket[name]
                raise RegistryConflictError(
                    f"({category!r}, {name!r}) already registered to {existing!r}. "
                    f"Pass force=True to override (intended for plugin overrides)."
                )
            bucket[name] = target
            return target

        if obj is None:
            return _do_register  # decorator form
        return _do_register(obj)

    def get(self, category: str, name: str) -> Any:
        self._check_category(category)
        bucket = self._store[category]
        if name not in bucket:
            raise NotRegisteredError(
                f"({category!r}, {name!r}) is not registered. "
                f"Available: {sorted(bucket)}"
            )
        return bucket[name]

    def list(self, category: str) -> list[str]:
        self._check_category(category)
        return sorted(self._store[category])

    def contains(self, category: str, name: str) -> bool:
        self._check_category(category)
        return name in self._store[category]

    def unregister(self, category: str, name: str) -> None:
        self._check_category(category)
        bucket = self._store[category]
        if name not in bucket:
            raise NotRegisteredError(f"({category!r}, {name!r}) is not registered.")
        del bucket[name]

    def clear(self, category: str | None = None) -> None:
        """Reset entries. With ``category=None``, clear all categories."""
        if category is None:
            for c in self._store:
                self._store[c].clear()
            return
        self._check_category(category)
        self._store[category].clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy useful for test fixtures."""
        return {c: dict(b) for c, b in self._store.items()}

    def restore(self, snap: dict[str, dict[str, Any]]) -> None:
        """Restore from :meth:`snapshot`. Categories outside the snapshot are kept."""
        for c, b in snap.items():
            if c not in self._categories:
                self.register_category(c)
            self._store[c] = dict(b)


_REGISTRY = Registry()


def get_registry() -> Registry:
    return _REGISTRY


def register(
    category: str,
    name: str,
    obj: Any | None = None,
    *,
    force: bool = False,
) -> Callable[[Any], Any] | Any:
    """Public decorator/function form. See :meth:`Registry.register`."""
    return _REGISTRY.register(category, name, obj, force=force)


def get(category: str, name: str) -> Any:
    return _REGISTRY.get(category, name)


def list_entries(category: str) -> list[str]:
    return _REGISTRY.list(category)


def categories() -> list[str]:
    return _REGISTRY.categories()


def register_category(category: str) -> None:
    _REGISTRY.register_category(category)


def unregister(category: str, name: str) -> None:
    _REGISTRY.unregister(category, name)


def contains(category: str, name: str) -> bool:
    return _REGISTRY.contains(category, name)
