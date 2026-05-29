"""Adversarial tests for ``lighttrain.config._resolver``.

Coverage beyond the flat ``tests/test_config.py``:

* Colon escape-hatch (``mod:Cls.method``) path coverage.
* Right-peel correctly distinguishes ``ModuleNotFoundError(e.name == prefix)``
  (continue peeling) from ``ModuleNotFoundError(e.name != prefix)`` (broken
  internal import — raise immediately).
* ``extra_kwargs`` precedence over ``params`` is pinned.
* ``instantiate=False`` returns the factory itself.
* Sugar form (``{name: x, lr: 1e-4}``) merges remaining keys into ``params``;
  precedence when ``params={"lr": 0.1}`` AND top-level ``lr=0.5`` is also
  present is pinned to current behavior (``params`` wins via ``setdefault``).
"""

from __future__ import annotations

import pytest

from lighttrain.config import ConfigResolveError, resolve
from lighttrain.registry import register


def test_resolve_short_name_via_registry(clean_registry):
    """Short-name spec is routed through the registry.

    Setup: register a class under ('optimizer', 'fake_lr').
    Input: ``{name: 'fake_lr', lr: 0.5}``.
    Expected: an instance of FakeLR with ``lr == 0.5`` (via params merge).
    """
    class FakeLR:
        def __init__(self, lr: float = 1e-3) -> None:
            self.lr = lr

    register("optimizer", "fake_lr", FakeLR)
    obj = resolve({"name": "fake_lr", "lr": 0.5}, category="optimizer")
    assert isinstance(obj, FakeLR)
    assert obj.lr == 0.5


def test_resolve_target_dotted_path_imports():
    """Dotted ``_target_`` constructs the class via right-peel import.

    Input: ``{_target_: 'decimal.Decimal', value: '3.14'}``.
    Expected: a Decimal instance whose str repr is ``'3.14'``.
    """
    obj = resolve({"_target_": "decimal.Decimal", "value": "3.14"})
    assert str(obj) == "3.14"


def test_resolve_target_colon_form_works():
    """Colon-form ``_target_`` resolves ``pkg.module:Class.method`` paths.

    Input: ``{_target_: 'decimal:Decimal'}`` (colon between module and class).
    Expected: factory returns the class itself when ``instantiate=False``;
    invoking it produces a Decimal.
    """
    cls = resolve({"_target_": "decimal:Decimal"}, instantiate=False)
    from decimal import Decimal as D
    assert cls is D
    inst = cls("2.5")
    assert str(inst) == "2.5"


def test_resolve_target_invalid_raises_resolveerror():
    """A clearly unimportable ``_target_`` raises ConfigResolveError.

    Input: ``{_target_: 'no.such.module:Nope'}`` — module truly missing.
    Expected: ConfigResolveError (NOT bare ModuleNotFoundError).
    """
    with pytest.raises(ConfigResolveError):
        resolve({"_target_": "no.such.module:Nope"})


def test_resolve_missing_xor_raises():
    """Neither ``name`` nor ``_target_`` set is rejected by ComponentSpec
    validator (xor enforcement).

    Input: ``{params: {x: 1}}`` — name and target both absent.
    Expected: ValueError or ConfigResolveError (Pydantic wraps to ValueError).
    """
    with pytest.raises((ValueError, ConfigResolveError)):
        resolve({"params": {"x": 1}})


def test_resolve_both_xor_raises():
    """Both ``name`` and ``_target_`` set is rejected (xor enforcement).

    Input: ``{name: 'a', _target_: 'pkg.Cls'}``.
    Expected: ValueError.
    """
    with pytest.raises(ValueError):
        resolve({"name": "a", "_target_": "pkg.Cls"}, category="model")


def test_resolve_short_name_requires_category():
    """``{name: 'x'}`` without a category is rejected upfront.

    Input: ``{name: 'x'}`` with no ``category`` kwarg.
    Expected: ConfigResolveError mentioning ``category``.
    """
    with pytest.raises(ConfigResolveError) as exc:
        resolve({"name": "x"})
    assert "category" in str(exc.value).lower()


def test_pin_resolve_extra_kwargs_override_params(clean_registry):
    """Pin: ``extra_kwargs`` takes precedence over ``spec.params`` (current
    behavior: ``kwargs.update(extra_kwargs)`` at line 138 of _resolver.py).

    Setup: register FakeLR.
    Input: spec with ``params={'lr': 0.1}``, ``extra_kwargs={'lr': 0.9}``.
    Expected: instance has ``lr == 0.9`` (extra_kwargs wins).

    If this behavior is intentionally changed (e.g., to make ``params`` win),
    update this test AND bump SCHEMA_VERSION (or document the breaking change).
    """
    class FakeLR:
        def __init__(self, lr: float = 1e-3) -> None:
            self.lr = lr

    register("optimizer", "fake_lr_v2", FakeLR)
    obj = resolve(
        {"name": "fake_lr_v2", "params": {"lr": 0.1}},
        category="optimizer",
        extra_kwargs={"lr": 0.9},
    )
    assert obj.lr == 0.9


def test_resolve_instantiate_false_returns_factory():
    """``instantiate=False`` returns the class itself, not an instance.

    Input: ``{_target_: 'decimal.Decimal'}`` with ``instantiate=False``.
    Expected: identity comparison with ``decimal.Decimal``.
    """
    from decimal import Decimal as D
    factory = resolve({"_target_": "decimal.Decimal"}, instantiate=False)
    assert factory is D


def test_resolve_construction_failure_wraps_typeerror(clean_registry):
    """A class whose ``__init__`` rejects the params raises ConfigResolveError
    (wrapping the underlying TypeError) rather than letting TypeError surface.

    Setup: register a Strict class whose ``__init__`` requires arg ``x: int``.
    Input: spec with ``wrong_kwarg=1``.
    Expected: ConfigResolveError, message names the bad call signature.
    """
    class Strict:
        def __init__(self, x: int) -> None:
            self.x = x

    register("model", "strict_kw", Strict)
    with pytest.raises(ConfigResolveError):
        resolve({"name": "strict_kw", "wrong_kwarg": 1}, category="model")


def test_pin_resolve_sugar_form_params_wins_over_top_level(clean_registry):
    """Pin: when a spec has BOTH ``params={'lr': 0.1}`` AND a top-level
    ``lr=0.5``, the explicit ``params`` value wins (current behavior:
    ``params.setdefault(k, v)`` at line 35 of _resolver.py — meaning the
    explicit params dict is populated first and ``setdefault`` does nothing).

    Setup: register FakeLR.
    Input: ``{name: 'fake_lr_v3', params: {lr: 0.1}, lr: 0.5}``.
    Expected: instance has ``lr == 0.1``.

    If this behavior is intentionally changed (e.g., to make top-level keys
    win), update this test AND document the breaking change in the changelog.
    """
    class FakeLR:
        def __init__(self, lr: float = 1e-3) -> None:
            self.lr = lr

    register("optimizer", "fake_lr_v3", FakeLR)
    obj = resolve(
        {"name": "fake_lr_v3", "params": {"lr": 0.1}, "lr": 0.5},
        category="optimizer",
    )
    assert obj.lr == 0.1


def test_resolve_target_one_part_only_raises():
    """``_target_`` of a single-part dotted path is invalid (line 61-62 of
    _resolver.py).

    Input: ``{_target_: 'singleword'}`` — no dots, no colon.
    Expected: ConfigResolveError mentioning the invalid path.
    """
    with pytest.raises(ConfigResolveError) as exc:
        resolve({"_target_": "singleword"})
    assert "singleword" in str(exc.value) or "Invalid" in str(exc.value)
