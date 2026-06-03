"""Atomic IO primitives for PrepGraph node outputs.

Every node writes into ``<store_root>/tmp/<fp>/`` first, fsyncs each shard,
then drops a ``MANIFEST_COMPLETE`` last, and finally ``os.replace``s the temp
dir into ``<store_root>/<kind>/<name>/<fp>/``. A crash anywhere in that path
leaves the cache table at a clean state — either the final dir exists with a
manifest, or it doesn't and gets retried next run.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MANIFEST_COMPLETE.json"
SHARD_COMPLETE_NAME = "complete.json"


def staging_dir(store_root: Path, fp: str) -> Path:
    """Per-fingerprint staging directory under ``<store_root>/tmp``."""
    return Path(store_root) / "tmp" / fp


def final_dir(store_root: Path, kind: str, name: str, fp: str) -> Path:
    """Final cache directory for a node's output."""
    return Path(store_root) / kind / name / fp


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def fsync_file(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems (e.g. on Windows tmp) don't support fsync on dirs;
        # we still flushed the file's writer, which is enough for our needs.
        pass


def write_manifest(target_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write the ``MANIFEST_COMPLETE`` file *last* — this is the commit signal."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    fsync_file(path)
    return path


def read_manifest(target_dir: Path) -> dict[str, Any] | None:
    path = Path(target_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def is_complete(target_dir: Path) -> bool:
    """A directory is complete iff its MANIFEST_COMPLETE is present and parseable."""
    return read_manifest(target_dir) is not None


def write_shard_state(target_dir: Path, shards: Iterable[Mapping[str, Any]]) -> Path:
    """Persist a per-shard ``complete.json`` map for crash-resume."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / SHARD_COMPLETE_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(list(shards), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def read_shard_state(target_dir: Path) -> list[dict[str, Any]]:
    path = Path(target_dir) / SHARD_COMPLETE_NAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
    except json.JSONDecodeError:
        pass
    return []


def commit(staging: Path, final: Path) -> None:
    """Atomically promote ``staging`` to ``final``.

    Pre-conditions:
      * ``staging`` exists and contains MANIFEST_COMPLETE
      * ``final`` does not exist (or is empty); we ``rmtree`` it to be safe

    Uses ``os.replace`` so the swap is atomic on POSIX and best-effort on
    Windows (which has its own atomic-rename semantics for empty targets).
    """
    staging = Path(staging)
    final = Path(final)
    if not is_complete(staging):
        raise RuntimeError(f"Refusing to commit incomplete staging dir: {staging}")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        # Replace requires the target to not exist on Windows; clean it up.
        shutil.rmtree(final, ignore_errors=True)
    os.replace(str(staging), str(final))


def cleanup_staging(store_root: Path) -> int:
    """Remove orphaned tmp/<fp>/ dirs without a manifest. Returns count removed."""
    tmp = Path(store_root) / "tmp"
    if not tmp.exists():
        return 0
    removed = 0
    for child in tmp.iterdir():
        if not child.is_dir():
            continue
        if not is_complete(child):
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


__all__ = [
    "MANIFEST_NAME",
    "SHARD_COMPLETE_NAME",
    "cleanup_staging",
    "commit",
    "ensure_dir",
    "final_dir",
    "fsync_file",
    "is_complete",
    "read_manifest",
    "read_shard_state",
    "staging_dir",
    "write_manifest",
    "write_shard_state",
]
