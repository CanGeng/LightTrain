"""Code snapshot — compact source provenance for run directories.

The snapshot is best-effort and controlled by environment variables, so the
training config/YAML schema does not need to grow storage-policy fields.

Modes:

* ``cas`` (default): store each source file once in a content-addressed blob
  store, and write a small per-run ``manifest.json``.
* ``archive``: write a self-contained ``code.zip`` plus ``manifest.json``.
* ``off``: do not create ``code.snapshot/``; frozen-step bundles fall back to
  pointing at the run directory.

Environment variables:

* ``LIGHTTRAIN_CODE_SNAPSHOT_MODE=cas|archive|off``
* ``LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR=/path/to/store`` (CAS mode only)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


MODE_ENV = "LIGHTTRAIN_CODE_SNAPSHOT_MODE"
STORE_DIR_ENV = "LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR"
VALID_MODES = frozenset({"cas", "archive", "off"})


@dataclass(frozen=True)
class _SnapshotFile:
    src: Path
    rel: str
    sha256: str
    size: int


def _package_root() -> Path:
    import lighttrain  # local import to avoid cycle at module-load time

    return Path(lighttrain.__file__).resolve().parent


def _snapshot_mode() -> str:
    mode = (os.getenv(MODE_ENV) or "cas").strip().lower() or "cas"
    if mode not in VALID_MODES:
        warnings.warn(
            f"code_snapshot: invalid {MODE_ENV}={mode!r}; falling back to 'cas'"
        )
        return "cas"
    return mode


def _default_store_dir(run_dir: Path) -> Path:
    configured = (os.getenv(STORE_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    return run_dir.parent / ".code_snapshot_store"


def _matches_exclude(name: str, excludes: tuple[str, ...]) -> bool:
    for pat in excludes:
        if pat.startswith("*.") and name.endswith(pat[1:]):
            return True
        if name == pat:
            return True
    return False


def _is_excluded(path: Path, root: Path, excludes: tuple[str, ...]) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(_matches_exclude(part, excludes) for part in rel.parts)


def _iter_tree_files(
    root: Path,
    rel_prefix: str,
    *,
    excludes: tuple[str, ...],
) -> Iterator[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if _is_excluded(path, root, excludes):
            continue
        if path.is_file():
            rel = Path(rel_prefix) / path.relative_to(root)
            yield path, rel.as_posix()


def _iter_snapshot_sources(
    package_root: Path,
    *,
    user_modules: Iterable[str] | None,
    excludes: tuple[str, ...],
) -> Iterator[tuple[Path, str]]:
    yield from _iter_tree_files(package_root, "lighttrain", excludes=excludes)

    if not user_modules:
        return

    for entry in user_modules:
        src = Path(entry)
        if not src.exists():
            warnings.warn(f"code_snapshot: user_module {src} not found, skipping")
            continue
        if src.is_file():
            if not _matches_exclude(src.name, excludes):
                yield src, (Path("user_modules") / src.name).as_posix()
            continue
        yield from _iter_tree_files(
            src,
            (Path("user_modules") / src.name).as_posix(),
            excludes=excludes,
        )


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def _collect_files(
    package_root: Path,
    *,
    user_modules: Iterable[str] | None,
    excludes: tuple[str, ...],
) -> list[_SnapshotFile]:
    files: list[_SnapshotFile] = []
    seen: set[str] = set()
    for src, rel in _iter_snapshot_sources(
        package_root, user_modules=user_modules, excludes=excludes
    ):
        if rel in seen:
            warnings.warn(f"code_snapshot: duplicate snapshot path {rel}, skipping")
            continue
        try:
            digest, size = _hash_file(src)
        except OSError as e:
            warnings.warn(f"code_snapshot: cannot read {src}: {e}")
            continue
        files.append(_SnapshotFile(src=src, rel=rel, sha256=digest, size=size))
        seen.add(rel)
    return files


def _manifest(
    *,
    mode: str,
    package_root: Path,
    files: list[_SnapshotFile],
    store_dir: Path | None = None,
    archive: str | None = None,
) -> dict:
    data: dict = {
        "schema_version": 2,
        "format": "lighttrain-code-snapshot",
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root.resolve()),
        "files": [
            {
                "path": f.rel,
                "sha256": f.sha256,
                "size": f.size,
                "source": str(f.src.resolve()),
            }
            for f in sorted(files, key=lambda item: item.rel)
        ],
    }
    if store_dir is not None:
        data["store_dir"] = str(store_dir.resolve())
    if archive is not None:
        data["archive"] = archive
    return data


def _write_manifest(snap_dir: Path, data: dict) -> None:
    (snap_dir / "manifest.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (snap_dir / "SNAPSHOT.txt").write_text(
        "lighttrain code snapshot\n", encoding="utf-8"
    )


def _ensure_blob(src: Path, blob: Path) -> None:
    if blob.exists():
        return
    blob.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_name(f".{blob.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, blob)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _capture_cas(
    snap_dir: Path,
    *,
    package_root: Path,
    files: list[_SnapshotFile],
    store_dir: Path,
) -> None:
    blob_root = store_dir / "blobs"
    blob_root.mkdir(parents=True, exist_ok=True)
    for f in files:
        blob = blob_root / f.sha256[:2] / f.sha256
        _ensure_blob(f.src, blob)
    _write_manifest(
        snap_dir,
        _manifest(
            mode="cas",
            package_root=package_root,
            files=files,
            store_dir=store_dir,
        ),
    )


def _capture_archive(
    snap_dir: Path,
    *,
    package_root: Path,
    files: list[_SnapshotFile],
) -> None:
    archive_name = "code.zip"
    with zipfile.ZipFile(
        snap_dir / archive_name, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for f in sorted(files, key=lambda item: item.rel):
            zf.write(f.src, f.rel)
    _write_manifest(
        snap_dir,
        _manifest(
            mode="archive",
            package_root=package_root,
            files=files,
            archive=archive_name,
        ),
    )


def _new_tmp_dir(run_dir: Path) -> Path:
    base = run_dir / f".code.snapshot.{os.getpid()}.tmp"
    candidate = base
    i = 0
    while candidate.exists():
        i += 1
        candidate = run_dir / f"{base.name}.{i}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def capture_code_snapshot(
    run_dir: Path | str,
    *,
    package_root: Path | None = None,
    user_modules: Iterable[str] | None = None,
    excludes: tuple[str, ...] = (
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".git",
    ),
) -> Path:
    """Capture compact code provenance under ``<run_dir>/code.snapshot``.

    Returns the snapshot directory on success, or ``Path(run_dir)`` when mode
    is ``off`` or a best-effort failure occurs. Existing snapshots are reused
    as-is so resume keeps pointing at the original code provenance.
    """
    run_path = Path(run_dir)
    snap_dir = run_path / "code.snapshot"
    if snap_dir.exists():
        return snap_dir

    mode = _snapshot_mode()
    if mode == "off":
        return run_path

    pkg = Path(package_root) if package_root is not None else _package_root()
    if not pkg.exists():
        warnings.warn(f"code_snapshot: package root {pkg} not found")
        return run_path

    tmp_dir: Path | None = None
    try:
        files = _collect_files(pkg, user_modules=user_modules, excludes=excludes)
        tmp_dir = _new_tmp_dir(run_path)
        if mode == "archive":
            _capture_archive(tmp_dir, package_root=pkg, files=files)
        else:
            _capture_cas(
                tmp_dir,
                package_root=pkg,
                files=files,
                store_dir=_default_store_dir(run_path),
            )

        try:
            tmp_dir.rename(snap_dir)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return snap_dir
        return snap_dir
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"code_snapshot: failed to capture snapshot: {e}")
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return run_path


__all__ = ["capture_code_snapshot"]
