"""Invariants plugins — the InvariantsCallback runtime + predicate library.

Mirrors core ``lighttrain.callbacks.invariants`` (the DSL framework:
``InvariantError`` + ``evaluate_check``). This bundled package holds the
concrete impls:

* ``callback.py`` — :class:`InvariantsCallback` (registers ``callback/invariants``)
* ``builtins.py`` — the baseline invariant predicates (register ``invariant/*``)
* ``regression_gate.py`` — ``RegressionGate`` (registers ``invariant/regression_gate``)

``InvariantsCallback`` is re-exported here so existing
``from lighttrain.builtin_plugins.callbacks.invariants import InvariantsCallback``
(previously resolving to the ``invariants.py`` module) keeps working now that
this is a package.
"""

from __future__ import annotations

from . import (  # noqa: F401 — import for @register side-effects
    builtins,
    regression_gate,
)
from .callback import InvariantsCallback

__all__ = ["InvariantsCallback"]
