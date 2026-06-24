"""Test that user_modules listed in config are actually imported before component build.

Covers both dotted-module-name and ./file-path forms.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from lighttrain.cli._runtime import _import_user_modules
from lighttrain.registry import get as _reg_get
from lighttrain.registry import unregister as _unregister

# ---------------------------------------------------------------------------
# Dotted-module-name form
# ---------------------------------------------------------------------------

def test_import_user_modules_dotted_name(tmp_path):
    mod_file = tmp_path / "lt_test_optim_dot.py"
    mod_file.write_text(textwrap.dedent("""
        import torch
        from lighttrain.registry import register

        @register("optimizer", "_test_optim_dot")
        class _TestOptimDot:
            def build(self, model):
                self._o = torch.optim.SGD(model.parameters(), lr=1e-3)
                return self
            def step(self): self._o.step()
            def zero_grad(self, set_to_none=True):
                self._o.zero_grad(set_to_none=set_to_none)
    """))
    sys.path.insert(0, str(tmp_path))
    try:
        _import_user_modules(["lt_test_optim_dot"])
        cls = _reg_get("optimizer", "_test_optim_dot")
        assert cls is not None
    finally:
        sys.path.remove(str(tmp_path))
        _unregister("optimizer", "_test_optim_dot")


# ---------------------------------------------------------------------------
# File-path form (./path.py)
# ---------------------------------------------------------------------------

def test_import_user_modules_file_path(tmp_path):
    mod_file = tmp_path / "lt_test_optim_fp.py"
    mod_file.write_text(textwrap.dedent("""
        import torch
        from lighttrain.registry import register

        @register("optimizer", "_test_optim_fp")
        class _TestOptimFp:
            def build(self, model):
                self._o = torch.optim.SGD(model.parameters(), lr=1e-3)
                return self
            def step(self): self._o.step()
            def zero_grad(self, set_to_none=True):
                self._o.zero_grad(set_to_none=set_to_none)
    """))
    _import_user_modules([str(mod_file)])  # absolute path
    cls = _reg_get("optimizer", "_test_optim_fp")
    assert cls is not None
    _unregister("optimizer", "_test_optim_fp")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_import_user_modules_bad_dotted_name():
    with pytest.raises(ImportError, match="user_modules"):
        _import_user_modules(["_nonexistent_module_xyz_lighttrain_test"])


def test_import_user_modules_bad_file_path(tmp_path):
    with pytest.raises(ImportError, match="user_modules"):
        _import_user_modules([str(tmp_path / "does_not_exist.py")])


def test_import_user_modules_empty_list():
    _import_user_modules([])  # no-op, must not raise
