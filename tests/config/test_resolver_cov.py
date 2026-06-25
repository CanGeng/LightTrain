"""Edge-case tests for ``lighttrain.config._resolver`` — coverage supplement.

Pins uncovered branches in the existing 93%-covered suite:

* Line 51-52  : ``_filter_kwargs`` — ``inspect.signature`` raises ValueError/TypeError
                (C-extension / builtin factory) → passthrough without filtering.
* Line 142    : ``_coerce`` — spec is already a ``ComponentSpec`` → returned as-is.
* Line 144    : ``_coerce`` — spec is neither ComponentSpec nor Mapping → ConfigResolveError.
* Lines 173-174: ``_import_target`` colon form — attribute chain not found after
                 module import → ConfigResolveError.
* Line 199    : ``_import_target`` dotted-only form — ``ModuleNotFoundError`` where
                 ``e.name != mod_str`` (broken internal dependency) → ConfigResolveError.
* Lines 203,205: ``_import_target`` dotted-only form — generic ``ImportError`` (not
                 ``ModuleNotFoundError``) → ConfigResolveError immediately.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

from lighttrain.config import ConfigResolveError, resolve
from lighttrain.config._resolver import _coerce, _filter_kwargs, _import_target
from lighttrain.config._schema import ComponentSpec

# ===========================================================================
# _filter_kwargs — lines 51-52: inspect.signature raises ValueError/TypeError
# ===========================================================================


def test_invariant_filter_kwargs_passthrough_when_signature_raises_valueerror():
    """``_filter_kwargs`` returns kwargs unchanged when ``inspect.signature``
    raises ``ValueError``.

    Some special callables (e.g. certain built-in C extensions) raise
    ``ValueError`` from ``inspect.signature``. The function must pass the
    kwargs through untouched so callers are not silently broken.
    """
    kwargs = {"a": 1, "b": 2}

    def _bad_factory():
        pass

    with patch("lighttrain.config._resolver.inspect.signature",
               side_effect=ValueError("no signature")):
        result = _filter_kwargs(_bad_factory, kwargs)

    assert result is kwargs or result == kwargs


def test_invariant_filter_kwargs_passthrough_when_signature_raises_typeerror():
    """``_filter_kwargs`` returns kwargs unchanged when ``inspect.signature``
    raises ``TypeError``.

    Certain callable objects (e.g. native slot wrappers) raise ``TypeError``.
    The resolver must gracefully degrade to pass-through rather than crashing.
    """
    kwargs = {"x": 10, "y": 20}

    class _CExtLike:
        pass

    with patch("lighttrain.config._resolver.inspect.signature",
               side_effect=TypeError("unsupported")):
        result = _filter_kwargs(_CExtLike, kwargs)

    assert result == kwargs


def test_pin_filter_kwargs_passthrough_preserves_all_keys():
    """Pin: passthrough on ValueError returns the *same* dict contents — no
    keys are dropped even if they would ordinarily be filtered.
    """
    kwargs = {"unrelated": True, "another": "value", "third": 3}

    with patch("lighttrain.config._resolver.inspect.signature",
               side_effect=ValueError("introspect failure")):
        result = _filter_kwargs(lambda: None, kwargs)

    assert set(result.keys()) == {"unrelated", "another", "third"}


# ===========================================================================
# _coerce — line 142: spec already a ComponentSpec
# ===========================================================================


def test_invariant_coerce_returns_componentspec_unchanged():
    """``_coerce`` short-circuits and returns the same ``ComponentSpec`` object
    when the input is already a ``ComponentSpec``.

    Identity (``is``) check ensures no unnecessary copying occurs.
    """
    cs = ComponentSpec(name="dummy")
    result = _coerce(cs)
    assert result is cs


def test_invariant_coerce_componentspec_with_target_returned_unchanged():
    """``_coerce`` returns a ``ComponentSpec`` with a ``_target_`` unchanged."""
    cs = ComponentSpec(_target_="os.path.join")
    result = _coerce(cs)
    assert result is cs
    assert result.target == "os.path.join"


# ===========================================================================
# _coerce — line 143-145: spec is not ComponentSpec and not Mapping
# ===========================================================================


def test_invariant_coerce_raises_for_non_mapping_non_componentspec():
    """``_coerce`` raises ``ConfigResolveError`` when the input is neither a
    ``ComponentSpec`` nor a ``Mapping``.

    This guards against accidental pass of plain strings/ints as specs.
    """
    with pytest.raises(ConfigResolveError, match="Spec must be a mapping"):
        _coerce("just_a_string")


def test_invariant_coerce_raises_for_integer_input():
    """Integer input to ``_coerce`` raises ``ConfigResolveError`` with the
    type name in the error message.
    """
    with pytest.raises(ConfigResolveError, match="int"):
        _coerce(42)


def test_invariant_coerce_raises_for_list_input():
    """List input to ``_coerce`` raises ``ConfigResolveError``."""
    with pytest.raises(ConfigResolveError, match="list"):
        _coerce([{"name": "x"}])


def test_invariant_coerce_raises_for_none_input():
    """``None`` input to ``_coerce`` raises ``ConfigResolveError`` (NoneType
    is not a Mapping).
    """
    with pytest.raises(ConfigResolveError, match="NoneType"):
        _coerce(None)


def test_invariant_resolve_raises_for_non_mapping_spec():
    """``resolve()`` surfaces the ConfigResolveError when spec is not a mapping."""
    with pytest.raises(ConfigResolveError, match="Spec must be a mapping"):
        resolve(123)


# ===========================================================================
# _import_target — lines 173-174: colon form, attribute chain missing
# ===========================================================================


def test_invariant_import_target_colon_bad_attr_raises():
    """Colon-form ``_import_target`` raises ``ConfigResolveError`` when the
    attribute chain after the module name does not exist.

    Input: ``'os:path.nonexistent_xyz_attr'`` — ``os`` is importable, but
    ``path.nonexistent_xyz_attr`` does not exist.
    Expected: ConfigResolveError wrapping the AttributeError.
    """
    with pytest.raises(ConfigResolveError, match="Cannot resolve"):
        _import_target("os:path.nonexistent_xyz_attr")


def test_invariant_import_target_colon_first_attr_missing_raises():
    """Colon-form raises ConfigResolveError when even the first attribute is
    absent from the module.

    Input: ``'os:_zzz_no_such_attr'``.
    """
    with pytest.raises(ConfigResolveError, match="Cannot resolve"):
        _import_target("os:_zzz_no_such_attr")


def test_invariant_import_target_colon_bad_module_raises():
    """Colon-form raises ConfigResolveError when the module itself is missing.

    Input: ``'_no_such_pkg_xyz:Foo'``.
    Expected: ConfigResolveError wrapping the ImportError.
    """
    with pytest.raises(ConfigResolveError, match="Cannot import"):
        _import_target("_no_such_pkg_xyz:Foo")


def test_invariant_import_target_colon_two_part_attr_chain_bad_second():
    """Colon-form with two-part attribute chain raises when the second part
    is missing.

    Input: ``'decimal:Decimal.nonexistent_method_zz'``.
    """
    with pytest.raises(ConfigResolveError, match="Cannot resolve"):
        _import_target("decimal:Decimal.nonexistent_method_zz")


# ===========================================================================
# _import_target — line 199: dotted form, broken internal import (missing dep)
# ===========================================================================


def _make_synthetic_module(mod_name: str, *, raises: Exception) -> None:
    """Insert a synthetic module under ``sys.modules`` that raises on import.

    The loader re-raises ``raises`` during ``exec_module``; this simulates a
    package whose internal dependency is missing.
    """
    spec = importlib.util.spec_from_loader(mod_name, loader=None)
    if spec is None:
        spec = importlib.machinery.ModuleSpec(mod_name, None)

    class _BrokenLoader:
        def create_module(self, s):
            return None

        def exec_module(self, module):
            raise raises

    spec.loader = _BrokenLoader()  # type: ignore[assignment]
    sys.modules.pop(mod_name, None)


def test_invariant_import_target_broken_internal_dep_raises():
    """When a module exists but raises ``ModuleNotFoundError`` with a *different*
    ``e.name`` (i.e., a missing internal dependency, not the module itself),
    ``_import_target`` raises ``ConfigResolveError`` immediately with a helpful
    message instead of continuing to peel.

    Line 199 path: ``missing != mod_str`` → raise.
    """
    # We need: import succeeds for the prefix but the module itself raises
    # ModuleNotFoundError(name=<some_other_dep>).
    # Strategy: inject a fake top-level module "os" alias that raises with
    # e.name pointing at a side dependency.  But "os" is already cached.
    # Instead, create a fresh synthetic package.

    fake_pkg = "_lighttrain_test_brkn_dep_pkg"
    inner_dep = "_lighttrain_missing_internal_dep"

    # Build a real ModuleNotFoundError with e.name set to inner_dep (NOT fake_pkg)
    internal_err = ModuleNotFoundError(
        f"No module named {inner_dep!r}", name=inner_dep
    )

    # Patch importlib.import_module so that importing fake_pkg triggers the error
    original_import = importlib.import_module

    def _patched_import(name: str, *args: Any, **kwargs: Any):
        if name == fake_pkg:
            raise internal_err
        return original_import(name, *args, **kwargs)

    with patch("lighttrain.config._resolver.importlib.import_module",
               side_effect=_patched_import):
        with pytest.raises(ConfigResolveError, match="missing dependency"):
            # Two parts so peeling yields mod_str = fake_pkg (split=1)
            _import_target(f"{fake_pkg}.SomeClass")


def test_pin_import_target_broken_internal_dep_error_contains_missing_name():
    """Pin: the ConfigResolveError message for a broken internal dependency
    includes the *name* of the missing internal dependency (not just the
    module prefix).
    """
    fake_pkg = "_lighttrain_test_brkn_dep_pkg2"
    inner_dep = "_missing_dep_xyz"

    internal_err = ModuleNotFoundError(
        f"No module named {inner_dep!r}", name=inner_dep
    )

    original_import = importlib.import_module

    def _patched_import(name: str, *args: Any, **kwargs: Any):
        if name == fake_pkg:
            raise internal_err
        return original_import(name, *args, **kwargs)

    with patch("lighttrain.config._resolver.importlib.import_module",
               side_effect=_patched_import):
        with pytest.raises(ConfigResolveError) as exc_info:
            _import_target(f"{fake_pkg}.SomeClass")

    msg = str(exc_info.value)
    assert inner_dep in msg


# ===========================================================================
# _import_target — lines 203, 205: generic ImportError (not ModuleNotFoundError)
# ===========================================================================


def test_invariant_import_target_generic_importerror_raises_immediately():
    """When the dotted-form right-peel triggers a generic ``ImportError``
    (not a ``ModuleNotFoundError``), the resolver must raise ``ConfigResolveError``
    immediately without peeling further.

    Lines 203-205 path: ``except ImportError``.
    """
    fake_pkg = "_lighttrain_test_generic_ie_pkg"
    generic_err = ImportError(f"cannot import due to binary mismatch in {fake_pkg!r}")

    original_import = importlib.import_module

    def _patched_import(name: str, *args: Any, **kwargs: Any):
        if name == fake_pkg:
            raise generic_err
        return original_import(name, *args, **kwargs)

    with patch("lighttrain.config._resolver.importlib.import_module",
               side_effect=_patched_import):
        with pytest.raises(ConfigResolveError, match="failed to import"):
            _import_target(f"{fake_pkg}.SomeClass")


def test_pin_import_target_generic_importerror_message_names_module():
    """Pin: the ConfigResolveError message for a generic ImportError names the
    module prefix that triggered the error, not just the full dotted path.

    Regression: if peeling continues past a real ImportError, the user gets a
    confusing 'Cannot resolve' message instead of actionable module info.
    """
    fake_pkg = "_lighttrain_test_generic_ie_pkg2"
    generic_err = ImportError("binary incompatibility in extension module")

    original_import = importlib.import_module

    def _patched_import(name: str, *args: Any, **kwargs: Any):
        if name == fake_pkg:
            raise generic_err
        return original_import(name, *args, **kwargs)

    with patch("lighttrain.config._resolver.importlib.import_module",
               side_effect=_patched_import):
        with pytest.raises(ConfigResolveError) as exc_info:
            _import_target(f"{fake_pkg}.SomeClass")

    msg = str(exc_info.value)
    assert fake_pkg in msg


def test_invariant_import_target_generic_importerror_does_not_peel_further():
    """When a generic ``ImportError`` occurs at the first peel level, the
    resolver must NOT continue peeling to shorter prefixes.

    If it silently continued peeling, a binary incompatibility error would
    masquerade as 'Cannot resolve', hiding the real cause.

    We verify by checking the error message says 'failed to import' (the
    branch at line 205), not 'Cannot resolve' (the fallthrough at line 224).
    """
    fake_pkg = "_lighttrain_test_no_peel_pkg"
    generic_err = ImportError("ABI mismatch in shared library")

    original_import = importlib.import_module

    def _patched_import(name: str, *args: Any, **kwargs: Any):
        if name == fake_pkg:
            raise generic_err
        return original_import(name, *args, **kwargs)

    with patch("lighttrain.config._resolver.importlib.import_module",
               side_effect=_patched_import):
        with pytest.raises(ConfigResolveError) as exc_info:
            _import_target(f"{fake_pkg}.Attr")

    # The message must come from the immediate-raise branch (line 205), not
    # from the exhausted-peeling fallback at the bottom of _import_target.
    assert "failed to import" in str(exc_info.value)


# ===========================================================================
# Integration: resolve() through _coerce(ComponentSpec) fast-path
# ===========================================================================


def test_invariant_resolve_accepts_componentspec_directly():
    """``resolve()`` accepts a pre-built ``ComponentSpec`` and constructs the
    target. The ``_coerce`` fast-path (line 142) must return it unchanged.
    """
    cs = ComponentSpec(_target_="decimal.Decimal", params={"value": "7.77"})
    from decimal import Decimal

    result = resolve(cs)
    assert isinstance(result, Decimal)
    assert str(result) == "7.77"


def test_invariant_resolve_componentspec_instantiate_false():
    """``resolve(ComponentSpec(...), instantiate=False)`` returns the factory
    without constructing, exercising the fast-path through _coerce.
    """
    from decimal import Decimal

    cs = ComponentSpec(_target_="decimal.Decimal")
    factory = resolve(cs, instantiate=False)
    assert factory is Decimal
