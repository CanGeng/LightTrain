"""Schema migration registry.

Migrations are pure-function patches keyed by ``(schema_kind, from_version,
to_version)``. The registry forms a DAG; :func:`find_path` does a BFS over it
to find the shortest chain of migrations from ``from_version`` to ``to_version``
(e.g. ``0.2 -> 0.3 -> 0.4``). Each migration must produce a payload whose
``schema_version`` equals its declared ``to_``.

File-level migration is built on top of :func:`migrate_payload` and additionally
writes ``<path>.pre-migration-bak`` plus a lineage ``migrated_from`` edge when
a :class:`LineageStore` is supplied.

Schema versions ``CURRENT`` come from ``lighttrain.prepgraph._fp.SCHEMA_VERSION``
so PrepGraph and lineage share one source of truth.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ..prepgraph._fp import SCHEMA_VERSION


MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

_REGISTRY: dict[tuple[str, str, str], MigrationFn] = {}


class SchemaMigrationError(RuntimeError):
    pass


def migrate(schema_kind: str, *, from_: str, to_: str) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a migration ``(schema_kind, from_, to_) -> fn``.

    Example::

        @migrate("config", from_="0.3", to_="0.4")
        def _bump_config(old):
            new = dict(old)
            if "ema" in new and "start" in new["ema"]:
                new["ema"]["start_step"] = new["ema"].pop("start")
            new["schema_version"] = "0.4"
            return new
    """
    def deco(fn: MigrationFn) -> MigrationFn:
        key = (schema_kind, from_, to_)
        if key in _REGISTRY:
            raise ValueError(f"duplicate migration registration: {key}")
        _REGISTRY[key] = fn
        return fn
    return deco


def registered_migrations() -> dict[tuple[str, str, str], MigrationFn]:
    return dict(_REGISTRY)


def find_path(schema_kind: str, from_version: str, to_version: str) -> list[tuple[str, str]]:
    """BFS shortest migration chain. Returns list of (from, to) hops.

    Empty list means ``from_version == to_version``. Raises
    :class:`SchemaMigrationError` when no path exists.
    """
    if from_version == to_version:
        return []
    seen = {from_version}
    queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(from_version, [])])
    while queue:
        cur, chain = queue.popleft()
        for (sk, src, dst), _ in _REGISTRY.items():
            if sk != schema_kind or src != cur or dst in seen:
                continue
            new_chain = chain + [(src, dst)]
            if dst == to_version:
                return new_chain
            seen.add(dst)
            queue.append((dst, new_chain))
    raise SchemaMigrationError(
        f"no migration path for schema {schema_kind!r}: {from_version} → {to_version}; "
        f"registered: {sorted(k for k in _REGISTRY if k[0] == schema_kind)}"
    )


def migrate_payload(
    payload: Mapping[str, Any],
    *,
    schema_kind: str,
    target: str | None = None,
) -> dict[str, Any]:
    """Apply the shortest migration chain to ``payload`` in memory.

    ``target=None`` resolves to ``SCHEMA_VERSION[schema_kind]``.
    """
    if target is None:
        target = SCHEMA_VERSION.get(schema_kind)
        if target is None:
            raise SchemaMigrationError(
                f"no CURRENT schema_version for {schema_kind!r}; "
                "register it in lighttrain/prepgraph/_fp.py SCHEMA_VERSION."
            )
    current = str(payload.get("schema_version") or "0.0")
    if current == target:
        return dict(payload)
    chain = find_path(schema_kind, current, target)
    out = dict(payload)
    for src, dst in chain:
        fn = _REGISTRY[(schema_kind, src, dst)]
        out = fn(dict(out))
        if out.get("schema_version") != dst:
            out["schema_version"] = dst
    return out


def migrate_file(
    path: str | Path,
    *,
    schema_kind: str,
    target: str | None = None,
    backup: bool = True,
    lineage_store: Any = None,
    in_place: bool = True,
) -> dict[str, Any]:
    """Migrate a YAML / JSON file and optionally write a lineage edge.

    Backup goes to ``<path>.pre-migration-bak`` (preserves extension info in name).
    Returns the migrated payload dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        payload = yaml.safe_load(raw) or {}
    elif suffix == ".json":
        payload = json.loads(raw)
    else:
        # Try JSON first, then YAML.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise SchemaMigrationError(f"file {path} did not parse to a mapping")
    migrated = migrate_payload(payload, schema_kind=schema_kind, target=target)
    if migrated == payload:
        return migrated

    if backup and in_place:
        bak = path.with_suffix(path.suffix + ".pre-migration-bak")
        shutil.copy2(path, bak)

    if in_place:
        out_text = (
            yaml.safe_dump(migrated, sort_keys=False)
            if suffix in (".yaml", ".yml")
            else json.dumps(migrated, indent=2)
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(out_text, encoding="utf-8")
        os.replace(tmp, path)

    if lineage_store is not None:
        try:
            old_id = lineage_store.upsert_node(
                kind="config" if schema_kind == "config" else "artifact",
                name=str(path),
                version=str(payload.get("schema_version") or "0.0"),
                schema_kind=schema_kind,
                schema_version=str(payload.get("schema_version") or "0.0"),
                payload_path=str(path),
                payload={"note": "pre-migration"},
            )
            new_id = lineage_store.upsert_node(
                kind="config" if schema_kind == "config" else "artifact",
                name=str(path),
                version=str(migrated.get("schema_version")),
                schema_kind=schema_kind,
                schema_version=str(migrated.get("schema_version")),
                payload_path=str(path),
                payload={"migrated_at": time.time()},
            )
            lineage_store.add_edge(
                old_id, new_id, "migrated_from",
                {"from": payload.get("schema_version"), "to": migrated.get("schema_version")},
            )
        except Exception:
            pass  # lineage is soft — never block migration on a DB hiccup
    return migrated


# ---------------------------------------------------------------- seed migrations
# These are the smallest possible migrations that prove the F4 acceptance path:
# legacy ``0.3`` payloads → current ``0.4``. Real future migrations register
# alongside, never bump CURRENT in SCHEMA_VERSION without registering a hop.


# ---------------------------------------------------------------- model_profiles
# v0.1.8 structural rewrite: a bare-dict ``model:`` block becomes a
# ``model_profiles: {<name>: {...}}`` group plus a ``model: <name>`` selector.
# This is NOT a schema_version hop — it is orthogonal to the version DAG above
# (recipes carry no ``schema_version``), so it gets its own text-level transform
# rather than going through ``migrate_payload``. The transform is surgical: it
# preserves comments, blank lines, and OmegaConf ``${...}`` interpolations,
# touching only the model block.

import re as _re


_MODEL_BLOCK_RE = _re.compile(r"^model:[ \t]*(#.*)?$")
_MODEL_FLOW_RE = _re.compile(r"^model:[ \t]*(\{.*\})[ \t]*(#.*)?$")


def migrate_model_to_profiles_text(
    raw: str, *, profile_name: str = "default"
) -> tuple[str, bool]:
    """Rewrite a top-level bare-dict ``model:`` block to ``model_profiles:``.

    Returns ``(new_text, changed)``. Idempotent: a recipe whose ``model:`` is
    already a string selector (or which has no top-level ``model:``) is returned
    unchanged with ``changed=False``. Handles both block form::

        model:
          name: tiny_lm
          d_model: 128

    and single-line flow form (``model: {name: tiny_lm, d_model: 128}``).
    """
    lines = raw.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    changed = False
    while i < n:
        line = lines[i]
        if not changed:
            flow = _MODEL_FLOW_RE.match(line)
            if flow:
                out.append(f"model: {profile_name}")
                out.append("model_profiles:")
                out.append(f"  {profile_name}: {flow.group(1)}")
                changed = True
                i += 1
                continue
            if _MODEL_BLOCK_RE.match(line):
                # Collect the indented block body (blank lines included) until
                # the next column-0 non-blank line.
                body: list[str] = []
                j = i + 1
                while j < n:
                    bl = lines[j]
                    if bl.strip() == "" or bl[:1] in (" ", "\t"):
                        body.append(bl)
                        j += 1
                        continue
                    break
                out.append(f"model: {profile_name}")
                out.append("model_profiles:")
                out.append(f"  {profile_name}:")
                for bl in body:
                    out.append(bl if bl.strip() == "" else "  " + bl)
                changed = True
                i = j
                continue
        out.append(line)
        i += 1
    result = "\n".join(out)
    if raw.endswith("\n"):
        result += "\n"
    return result, changed


def rewrite_model_to_profiles_file(
    path: str | Path,
    *,
    profile_name: str = "default",
    in_place: bool = True,
    backup: bool = True,
) -> bool:
    """Apply :func:`migrate_model_to_profiles_text` to a YAML file.

    Returns ``True`` if the file was changed. Writes a ``.pre-migration-bak``
    backup before an in-place rewrite, and writes atomically via a tmp + rename.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    new_text, changed = migrate_model_to_profiles_text(raw, profile_name=profile_name)
    if not changed or not in_place:
        return changed
    if backup:
        bak = path.with_suffix(path.suffix + ".pre-migration-bak")
        shutil.copy2(path, bak)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)
    return changed


@migrate("config", from_="0.3", to_="0.4")
def _migrate_config_03_to_04(old: dict[str, Any]) -> dict[str, Any]:
    new = dict(old)
    if "ema" in new and isinstance(new["ema"], dict) and "start" in new["ema"]:
        new["ema"]["start_step"] = new["ema"].pop("start")
    new.setdefault("mode", "lab")
    new["schema_version"] = "0.4"
    return new


@migrate("artifact_header", from_="0.3", to_="0.4")
def _migrate_artifact_header_03_to_04(old: dict[str, Any]) -> dict[str, Any]:
    new = dict(old)
    new.setdefault("framework_version", "torch:unknown")
    new["schema_version"] = "0.4"
    return new


@migrate("checkpoint_manifest", from_="0.3", to_="0.4")
def _migrate_checkpoint_manifest_03_to_04(old: dict[str, Any]) -> dict[str, Any]:
    new = dict(old)
    new["schema_version"] = "0.4"
    return new


__all__ = [
    "MigrationFn",
    "SchemaMigrationError",
    "find_path",
    "migrate",
    "migrate_file",
    "migrate_model_to_profiles_text",
    "migrate_payload",
    "registered_migrations",
    "rewrite_model_to_profiles_file",
]
