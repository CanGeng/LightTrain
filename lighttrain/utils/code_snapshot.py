"""Code snapshot — captures every ``.py`` file the run depended on.

At ``make_run_dir`` time we capture the resolved config, environment metadata,
and a copy of every ``.py`` file the run depended on for full reproducibility.

We snapshot:

* the ``lighttrain`` package source tree (everything under
  ``Path(lighttrain.__file__).parent`` excluding ``__pycache__`` / ``*.pyc``);
* every entry in ``cfg.user_modules`` (one path per entry; files are copied
  verbatim into ``code.snapshot/user_modules/<basename>``).

We deliberately do **not** snapshot ``frontier_plugins/`` here — those are
plugins installed alongside lighttrain and follow the same registry contract;
their source paths are written into ``env.json`` so the snapshot stays small
(typical lighttrain tree ≈ 1 MB; frontier_plugins could balloon to many MB
once 4-bit kernels and friends land).

Failure mode: this function is best-effort. If it can't copy something it
emits a ``warnings.warn`` and returns ``run_dir`` itself, so frozen-step
bundles fall back to the "pointer = run_dir" behavior.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Iterable


def _package_root() -> Path:
    import lighttrain  # local import to avoid cycle at module-load time

    return Path(lighttrain.__file__).resolve().parent


def _safe_copy_tree(src: Path, dst: Path, *, excludes: tuple[str, ...]) -> None:
    """Copy ``src`` to ``dst`` skipping any segment matching ``excludes``."""

    def _ignore(_dir: str, names: list[str]) -> list[str]:
        skip = []
        for n in names:
            if n in excludes:
                skip.append(n)
                continue
            for pat in excludes:
                if pat.startswith("*.") and n.endswith(pat[1:]):
                    skip.append(n)
                    break
        return skip

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=False)


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
    """Snapshot ``lighttrain`` (and ``user_modules``) into ``<run_dir>/code.snapshot``.

    Returns the absolute snapshot directory on success, or ``Path(run_dir)``
    on any failure (with a ``warnings.warn``). Idempotent across re-runs in
    the same dir: existing ``code.snapshot/`` is preserved as-is (no
    overwrite, no exception). Resume reuses the original snapshot.
    """
    run_path = Path(run_dir)
    snap_dir = run_path / "code.snapshot"
    if snap_dir.exists():
        return snap_dir

    pkg = Path(package_root) if package_root is not None else _package_root()
    if not pkg.exists():
        warnings.warn(f"code_snapshot: package root {pkg} not found")
        return run_path

    try:
        snap_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return snap_dir
    except OSError as e:
        warnings.warn(f"code_snapshot: cannot create {snap_dir}: {e}")
        return run_path

    # 1) lighttrain package source.
    try:
        _safe_copy_tree(pkg, snap_dir / "lighttrain", excludes=excludes)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"code_snapshot: failed to copy lighttrain package: {e}")

    # 2) user_modules — one file (or directory) per entry.
    if user_modules:
        user_dir = snap_dir / "user_modules"
        try:
            user_dir.mkdir(exist_ok=True)
        except OSError as e:
            warnings.warn(f"code_snapshot: cannot create user_modules dir: {e}")
            return snap_dir
        for entry in user_modules:
            src = Path(entry)
            if not src.exists():
                warnings.warn(f"code_snapshot: user_module {src} not found, skipping")
                continue
            try:
                if src.is_file():
                    shutil.copy2(src, user_dir / src.name)
                else:
                    _safe_copy_tree(src, user_dir / src.name, excludes=excludes)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"code_snapshot: failed to copy user_module {src}: {e}"
                )

    # 3) tiny marker so the frozen_step pointer can verify the directory
    # is a valid code snapshot.
    try:
        (snap_dir / "SNAPSHOT.txt").write_text(
            "lighttrain code snapshot\n", encoding="utf-8"
        )
    except OSError:
        pass

    return snap_dir


__all__ = ["capture_code_snapshot"]
