"""Lineage subsystem.

Public surface (re-exports):
  * :class:`LineageStore` — SQLite persistence per run.
  * :func:`cycle_check` / :func:`apply_cycle_policy` — self-feeding detection.
  * :func:`to_mermaid` / :func:`to_dot` — graph export.
  * :func:`migrate` / :func:`migrate_payload` / :func:`migrate_file` /
    :class:`SchemaMigrationError` — schema migration registry + file driver.
  * :class:`RetentionPolicy` / :func:`gc_artifacts` / :func:`prune_orphans`.

The lineage ``content_hash`` is identical to the PrepGraph fingerprint —
both are :func:`lighttrain.data.prepgraph._fp.compose_fingerprint`.
"""

from __future__ import annotations

from lighttrain.data.prepgraph._fp import compose_fingerprint as content_hash

from .dag import CycleHit, apply_cycle_policy, cycle_check, to_dot, to_mermaid
from .migration import (
    SchemaMigrationError,
    find_path,
    migrate,
    migrate_file,
    migrate_payload,
    registered_migrations,
)
from .retention import GCReport, RetentionPolicy, gc_artifacts, prune_orphans
from .store import LineageStore

__all__ = [
    "CycleHit",
    "GCReport",
    "LineageStore",
    "RetentionPolicy",
    "SchemaMigrationError",
    "apply_cycle_policy",
    "content_hash",
    "cycle_check",
    "find_path",
    "gc_artifacts",
    "migrate",
    "migrate_file",
    "migrate_payload",
    "prune_orphans",
    "registered_migrations",
    "to_dot",
    "to_mermaid",
]
