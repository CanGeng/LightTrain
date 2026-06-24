"""Invariants subsystem.

A small DSL (``InvariantError`` / ``evaluate_check``) that the
:class:`InvariantsCallback` runs every step. The seven baseline invariant impls
(``loss_finite / grad_norm_bounded / lr_nonneg / label_mask_nonzero /
param_count_stable / dtype_stable / batch_nonempty``) are registered impls living
in ``lighttrain.builtin_plugins.invariants.builtins`` (DESIGN §3.3), picked up by
auto-discovery.
"""

from __future__ import annotations

from ._dsl import InvariantError, evaluate_check

__all__ = ["InvariantError", "evaluate_check"]
