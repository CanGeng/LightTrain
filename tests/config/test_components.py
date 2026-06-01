"""Failure contract for component auto-discovery (#11).

``import_all_components`` replaced a hand list of ``try/except ImportError`` that
blindly swallowed *everything* — including internal breakage. The replacement is
loud on internal failure and silent only on a missing optional third-party dep.
These pin that contract so a future stray ``except Exception`` can't quietly
revert it to blind-swallow.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lighttrain.config._components import _safe_import


def _write_module(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_non_importerror_at_import_propagates(tmp_path, monkeypatch):
    """A built-in module raising a non-ImportError (e.g. RuntimeError) at import
    time is loud — never swallowed."""
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_module(tmp_path, "lt_boom_mod", "raise RuntimeError('boom at import')")
    with pytest.raises(RuntimeError, match="boom at import"):
        _safe_import("lt_boom_mod")


def test_missing_third_party_dep_is_skipped(tmp_path, monkeypatch):
    """A module that can't import an absent optional third-party dependency is
    skipped silently (the optional-backend case)."""
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_module(
        tmp_path, "lt_optdep_mod", "import a_third_party_dep_that_is_absent_xyz"
    )
    _safe_import("lt_optdep_mod")  # must NOT raise


def test_missing_internal_lighttrain_module_propagates():
    """A missing ``lighttrain.*`` module is internal breakage — loud."""
    with pytest.raises(ImportError):
        _safe_import("lighttrain.this_submodule_does_not_exist_xyz")
