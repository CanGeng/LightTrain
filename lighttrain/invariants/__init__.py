"""Invariants subsystem.

A small DSL + registry of factory functions that the
:class:`InvariantsCallback` runs every step. See ``builtins.py`` for the
seven baseline invariants ``loss_finite / grad_norm_bounded / lr_nonneg /
label_mask_nonzero / param_count_stable / dtype_stable / batch_nonempty``.
"""

from __future__ import annotations

from . import builtins as _builtins  # noqa: F401 — registers via decorators
from ._dsl import InvariantError, evaluate_check

__all__ = ["InvariantError", "evaluate_check"]
