"""Tests for ``lighttrain.cli._runtime`` helpers.

Covers v0.1.7 fixes:

* Issue #9 — ``_import_user_modules`` must be idempotent within a process so
  resume / multi-stage scripts don't re-run @register decorators and raise
  ``RegistryConflictError``.
* Issue #2 + #10 — ``setup_run_from_config`` accepts either a config path or
  an already-parsed ``RootConfig``; combining ``overrides`` with a parsed
  RootConfig is rejected.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lighttrain.cli._runtime import (
    _IMPORTED_USER_MODULES,
    _import_user_modules,
    setup_run_from_config,
)
from lighttrain.config import load_config
from lighttrain.registry import RegistryConflictError, get_registry


# ===========================================================================
# Issue #9 — _import_user_modules idempotency
# ===========================================================================


@pytest.fixture
def _clean_imported_cache():
    """Snapshot the module-level cache so tests don't poison each other."""
    snap = set(_IMPORTED_USER_MODULES)
    try:
        yield
    finally:
        _IMPORTED_USER_MODULES.clear()
        _IMPORTED_USER_MODULES.update(snap)


def test_import_user_modules_idempotent(
    tmp_path: Path, clean_registry, _clean_imported_cache
):
    """Goal (Issue #9): a second call for the same file path must be a no-op,
    not a second ``spec.loader.exec_module`` that re-runs @register and
    raises ``RegistryConflictError``.

    Setup: write a temp ``.py`` that registers a new optimizer (force=False,
    so a duplicate registration WOULD raise). Call _import_user_modules
    twice. The second call must not raise.
    """
    plugin = tmp_path / "v017_idempotency_plugin.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from lighttrain.registry import register

            @register("optimizer", "v017_idempotent_dummy")
            class DummyOpt:
                def __init__(self, **kw): pass
            """
        ),
        encoding="utf-8",
    )

    _import_user_modules([str(plugin)])
    # Pre-fix, this second call would re-execute the decorator and raise.
    _import_user_modules([str(plugin)])

    reg = get_registry()
    assert reg.get("optimizer", "v017_idempotent_dummy") is not None


def test_import_user_modules_dedupes_via_resolved_path(
    tmp_path: Path, clean_registry, _clean_imported_cache
):
    """Goal (Issue #9): the cache key is the *resolved* absolute path, so
    relative + absolute references to the same file collapse to one import.
    """
    plugin = tmp_path / "v017_resolved_dedup_plugin.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from lighttrain.registry import register

            @register("optimizer", "v017_resolved_dedup_dummy")
            class DummyOpt:
                def __init__(self, **kw): pass
            """
        ),
        encoding="utf-8",
    )

    abs_path = str(plugin.resolve())
    _import_user_modules([abs_path])
    _import_user_modules([abs_path])  # exact same key — covered by primary test
    _import_user_modules([str(plugin)])  # same file, possibly non-canonical form

    reg = get_registry()
    assert reg.get("optimizer", "v017_resolved_dedup_dummy") is not None


# ===========================================================================
# Issue #2 + #10 — setup_run_from_config dispatch on Path vs RootConfig
# ===========================================================================


def test_setup_run_from_config_rejects_overrides_with_rootconfig(tmp_path: Path):
    """Goal (Issue #2 + #10): an already-parsed RootConfig already had its
    overrides baked in at load_config time; passing fresh overrides
    alongside it is semantically ambiguous and must raise ``ValueError``
    rather than be silently ignored.
    """
    cfg_path = tmp_path / "recipe.yaml"
    cfg_path.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    cfg = load_config(cfg_path)

    with pytest.raises(ValueError, match="overrides"):
        setup_run_from_config(cfg, overrides=["++seed=42"])


def test_setup_run_from_config_rejects_wrong_type(tmp_path: Path):
    """Goal (Issue #2): passing something that's neither a path nor a
    RootConfig raises a clear TypeError naming the bad type — not the
    pre-fix opaque ``argument should be a str or an os.PathLike object``.
    """
    with pytest.raises(TypeError, match="RootConfig"):
        setup_run_from_config(42)  # type: ignore[arg-type]
