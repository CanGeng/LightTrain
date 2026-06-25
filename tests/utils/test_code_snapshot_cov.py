"""Coverage-extension tests for ``lighttrain.utils.code_snapshot``.

Pins every uncovered branch not reached by tests/utils/test_code_snapshot.py:

* ``_snapshot_mode`` — invalid env-var value → warning + fallback to "cas"
  (lines 55, 58)
* ``_matches_exclude`` — wildcard extension pattern branch  (line 72)
* ``_is_excluded`` — path not relative to root → ValueError catch  (lines 81-82)
* ``_iter_snapshot_sources`` — user_module path does not exist → warn + skip
  (lines 114-115); user_module is a directory → recurse (line 120)
* ``_collect_files`` — duplicate rel path → warn + skip  (lines 149-150);
  OSError from _hash_file → warn + continue  (lines 153-155)
* ``_new_tmp_dir`` — collision loop (candidate exists → bump suffix)
  (lines 268-269)
* ``capture_code_snapshot`` — package root not found → warn + return run_path
  (lines 304-305); FileExistsError on rename → cleanup + return snap_dir
  (lines 323-325); generic Exception → warn + cleanup tmp + return run_path
  (lines 327-331)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from lighttrain.utils.code_snapshot import (
    _collect_files,
    _is_excluded,
    _iter_snapshot_sources,
    _matches_exclude,
    _new_tmp_dir,
    _snapshot_mode,
    capture_code_snapshot,
)

# ---------------------------------------------------------------------------
# _snapshot_mode — invalid env value
# ---------------------------------------------------------------------------


def test_invariant_snapshot_mode_invalid_warns_and_falls_back_to_cas(monkeypatch):
    """An unrecognised MODE_ENV value emits a UserWarning and returns 'cas'."""
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "invalid_xyz")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _snapshot_mode()
    assert result == "cas"
    assert any("invalid" in str(w.message).lower() for w in caught), caught


def test_pin_current_behavior_snapshot_mode_whitespace_invalid(monkeypatch):
    """Pin: env value that is only whitespace around an invalid token still
    hits the invalid-mode branch (strip+lower leaves a non-VALID string).

    Docstring flag: this pins the CURRENT fallback behaviour; if the env
    parsing logic is tightened this test may need updating.
    """
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "  BOGUS  ")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _snapshot_mode()
    assert result == "cas"
    assert any("bogus" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# _matches_exclude — wildcard extension branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, pat, expected",
    [
        ("module.pyc", "*.pyc", True),   # wildcard extension match
        ("module.pyo", "*.pyo", True),   # wildcard extension match
        ("module.py", "*.pyc", False),   # extension mismatch
        ("__pycache__", "__pycache__", True),  # exact match
        ("other_dir", "__pycache__", False),   # exact miss
        ("file.txt", "*.pyc", False),    # extension mismatch
    ],
)
def test_invariant_matches_exclude(name, pat, expected):
    """``_matches_exclude`` correctly handles wildcard and exact patterns."""
    assert _matches_exclude(name, (pat,)) is expected


def test_invariant_matches_exclude_wildcard_returns_true_short_circuits(monkeypatch):
    """When a wildcard pattern matches, True is returned immediately (line 72)."""
    result = _matches_exclude("foo.pyc", ("*.pyc", "*.pyo"))
    assert result is True


# ---------------------------------------------------------------------------
# _is_excluded — ValueError branch (path not relative to root)
# ---------------------------------------------------------------------------


def test_invariant_is_excluded_path_outside_root(tmp_path):
    """When ``path`` is not under ``root``, ValueError is caught and the path
    itself is used directly for part matching (lines 81-82).
    """

    root = tmp_path / "pkg"
    root.mkdir()
    outside = tmp_path / "other" / "__pycache__" / "foo.cpython-312.pyc"
    outside.parent.mkdir(parents=True)
    outside.touch()

    # "__pycache__" is a part of `outside` so it should be excluded
    assert _is_excluded(outside, root, ("__pycache__",)) is True


def test_invariant_is_excluded_path_outside_root_no_match(tmp_path):
    """Path outside root with no matching part → not excluded."""

    root = tmp_path / "pkg"
    root.mkdir()
    outside = tmp_path / "other" / "clean_dir" / "module.py"
    outside.parent.mkdir(parents=True)
    outside.touch()

    assert _is_excluded(outside, root, ("__pycache__",)) is False


# ---------------------------------------------------------------------------
# _iter_snapshot_sources — user_module not found (lines 114-115)
# ---------------------------------------------------------------------------


def test_invariant_user_module_not_found_warns_and_skips(tmp_path):
    """A non-existent user_module path emits a warning and is skipped."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    missing = str(tmp_path / "does_not_exist.py")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = list(
            _iter_snapshot_sources(pkg, user_modules=[missing], excludes=())
        )
    # The missing module should not appear in results
    user_paths = [r for _, r in results if r.startswith("user_modules/")]
    assert user_paths == []
    # A warning must have been issued
    assert any("not found" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# _iter_snapshot_sources — user_module is a directory (line 120)
# ---------------------------------------------------------------------------


def test_invariant_user_module_directory_is_recursed(tmp_path):
    """A user_module entry that is a directory causes ``_iter_tree_files`` to
    be called for it (line 120), yielding files under user_modules/<dir>/.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    user_dir = tmp_path / "myext"
    user_dir.mkdir()
    (user_dir / "a.py").write_text("A = 1\n", encoding="utf-8")
    (user_dir / "b.py").write_text("B = 2\n", encoding="utf-8")

    results = list(
        _iter_snapshot_sources(pkg, user_modules=[str(user_dir)], excludes=())
    )
    rel_paths = [r for _, r in results]
    assert any("user_modules/myext/a.py" in p for p in rel_paths)
    assert any("user_modules/myext/b.py" in p for p in rel_paths)


# ---------------------------------------------------------------------------
# _collect_files — duplicate rel path (lines 149-150)
# ---------------------------------------------------------------------------


def test_invariant_collect_files_duplicate_warns_and_skips(tmp_path, monkeypatch):
    """When two sources map to the same rel path, the second is warned and
    skipped (lines 149-150).

    We achieve the duplicate by monkey-patching ``_iter_snapshot_sources``
    to yield the same (src, rel) pair twice.
    """
    src_file = tmp_path / "dummy.py"
    src_file.write_text("X = 1\n", encoding="utf-8")

    def _fake_iter(pkg, *, user_modules, excludes):
        yield src_file, "lighttrain/dummy.py"
        yield src_file, "lighttrain/dummy.py"  # duplicate

    with patch(
        "lighttrain.utils.code_snapshot._iter_snapshot_sources", side_effect=_fake_iter
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            files = _collect_files(
                tmp_path, user_modules=None, excludes=()
            )

    # Only one entry survives
    assert len(files) == 1
    # A duplicate warning was emitted
    assert any("duplicate" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# _collect_files — OSError from _hash_file (lines 153-155)
# ---------------------------------------------------------------------------


def test_invariant_collect_files_oserror_warns_and_skips(tmp_path, monkeypatch):
    """When ``_hash_file`` raises OSError, the file is warned about and
    skipped; other files proceed normally (lines 153-155).
    """
    good_file = tmp_path / "good.py"
    good_file.write_text("GOOD = 1\n", encoding="utf-8")
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("BAD = 1\n", encoding="utf-8")

    # Import the real implementation before patching to avoid recursion.
    from lighttrain.utils.code_snapshot import _hash_file as _real_hash

    def _selective_hash(path: Path):
        if path == bad_file:
            raise OSError("permission denied (test)")
        return _real_hash(path)

    def _fake_iter(pkg, *, user_modules, excludes):
        yield good_file, "lighttrain/good.py"
        yield bad_file, "lighttrain/bad.py"

    with patch(
        "lighttrain.utils.code_snapshot._iter_snapshot_sources", side_effect=_fake_iter
    ):
        with patch(
            "lighttrain.utils.code_snapshot._hash_file", side_effect=_selective_hash
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                files = _collect_files(tmp_path, user_modules=None, excludes=())

    # Only the good file survives
    assert len(files) == 1
    assert files[0].rel == "lighttrain/good.py"
    # Warning emitted for bad file
    assert any("cannot read" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# _new_tmp_dir — collision loop (lines 268-269)
# ---------------------------------------------------------------------------


def test_invariant_new_tmp_dir_collision_creates_suffixed_candidate(tmp_path):
    """When the base candidate already exists, ``_new_tmp_dir`` bumps the
    suffix until it finds a free name (lines 268-269).
    """
    pid = os.getpid()
    base_name = f".code.snapshot.{pid}.tmp"

    # Pre-create the base candidate so the while-loop fires at least once.
    collision = tmp_path / base_name
    collision.mkdir()

    result = _new_tmp_dir(tmp_path)
    # Must be a different directory (the suffixed .1 variant)
    assert result != collision
    assert result.exists()
    assert result.name.startswith(base_name)


# ---------------------------------------------------------------------------
# capture_code_snapshot — package root not found (lines 304-305)
# ---------------------------------------------------------------------------


def test_invariant_capture_nonexistent_package_root_warns_and_returns_run_dir(
    tmp_path, monkeypatch
):
    """When ``package_root`` does not exist, a warning is issued and
    ``run_dir`` is returned (lines 304-305).
    """
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "cas")
    ghost_root = tmp_path / "nonexistent_pkg"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = capture_code_snapshot(tmp_path, package_root=ghost_root)

    assert result == tmp_path
    assert not (tmp_path / "code.snapshot").exists()
    assert any("not found" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# capture_code_snapshot — FileExistsError on rename (lines 323-325)
# ---------------------------------------------------------------------------


def test_invariant_capture_file_exists_error_on_rename_returns_snap_dir(
    tmp_path, monkeypatch
):
    """When another process races to create the snapshot and ``rename``
    raises FileExistsError, the tmp dir is cleaned up and the (now-existing)
    snap_dir is returned (lines 323-325).
    """
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "archive")

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# pkg\n", encoding="utf-8")

    snap_dir = tmp_path / "code.snapshot"


    def _racing_rename(self, target):
        # Simulate the race: another process just created snap_dir right
        # before our rename.
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "manifest.json").write_text("{}", encoding="utf-8")
        raise FileExistsError("race condition (test)")

    with patch.object(Path, "rename", _racing_rename):
        result = capture_code_snapshot(tmp_path, package_root=pkg)

    assert result == snap_dir
    # Tmp dirs must have been cleaned up (no stale .tmp dirs)
    tmp_leftovers = list(tmp_path.glob(".code.snapshot.*.tmp*"))
    assert tmp_leftovers == []


# ---------------------------------------------------------------------------
# capture_code_snapshot — generic exception → warn + cleanup (lines 327-331)
# ---------------------------------------------------------------------------


def test_invariant_capture_generic_exception_warns_and_returns_run_dir(
    tmp_path, monkeypatch
):
    """Any unexpected exception during capture is caught, a warning is
    issued, the tmp dir is cleaned up, and ``run_dir`` is returned
    (lines 327-331).
    """
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "cas")

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# pkg\n", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("injected failure for test")

    with patch("lighttrain.utils.code_snapshot._collect_files", side_effect=_boom):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = capture_code_snapshot(tmp_path, package_root=pkg)

    assert result == tmp_path
    assert not (tmp_path / "code.snapshot").exists()
    assert any("failed to capture" in str(w.message).lower() for w in caught), caught
    # No stale tmp dirs left
    tmp_leftovers = list(tmp_path.glob(".code.snapshot.*.tmp*"))
    assert tmp_leftovers == []


def test_invariant_capture_exception_with_tmp_dir_already_created_cleans_up(
    tmp_path, monkeypatch
):
    """When the exception fires AFTER ``_new_tmp_dir`` creates the tmp dir,
    ``shutil.rmtree`` still cleans it up (lines 329-330).
    """
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "cas")

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# pkg\n", encoding="utf-8")


    def _new_tmp_then_boom(run_dir):
        from lighttrain.utils.code_snapshot import _new_tmp_dir as _real
        d = _real(run_dir)
        raise RuntimeError(f"boom after tmp created at {d}")

    with patch(
        "lighttrain.utils.code_snapshot._new_tmp_dir", side_effect=_new_tmp_then_boom
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = capture_code_snapshot(tmp_path, package_root=pkg)

    assert result == tmp_path
    # No stale tmp dirs should remain
    tmp_leftovers = list(tmp_path.glob(".code.snapshot.*.tmp*"))
    assert tmp_leftovers == []
    assert any("failed to capture" in str(w.message).lower() for w in caught), caught


# ---------------------------------------------------------------------------
# _default_store_dir — STORE_DIR_ENV configured vs default
# ---------------------------------------------------------------------------


def test_invariant_default_store_dir_uses_env_when_set(tmp_path, monkeypatch):
    """When STORE_DIR_ENV is set, ``_default_store_dir`` returns that path."""
    from lighttrain.utils.code_snapshot import _default_store_dir

    custom = tmp_path / "custom_store"
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", str(custom))
    result = _default_store_dir(tmp_path / "run")
    assert result == custom


def test_invariant_default_store_dir_falls_back_to_sibling(tmp_path, monkeypatch):
    """When STORE_DIR_ENV is unset, returns ``run_dir.parent / .code_snapshot_store``."""
    from lighttrain.utils.code_snapshot import _default_store_dir

    monkeypatch.delenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", raising=False)
    run_dir = tmp_path / "run"
    result = _default_store_dir(run_dir)
    assert result == tmp_path / ".code_snapshot_store"


# ---------------------------------------------------------------------------
# User module excluded by pattern (line 117-119 — excluded file not yielded)
# ---------------------------------------------------------------------------


def test_invariant_user_module_file_excluded_by_pattern_not_yielded(tmp_path):
    """A user_module that is a single file AND matches an exclude pattern is
    silently skipped (line 117 branch where _matches_exclude returns True).
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    pyc_file = tmp_path / "compiled.pyc"
    pyc_file.write_bytes(b"\x00" * 4)

    results = list(
        _iter_snapshot_sources(
            pkg, user_modules=[str(pyc_file)], excludes=("*.pyc",)
        )
    )
    user_paths = [r for _, r in results if r.startswith("user_modules/")]
    assert user_paths == []
