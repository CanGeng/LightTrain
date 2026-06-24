"""Tests for the built-in registered invariants in
``lighttrain.builtin_plugins.callbacks.invariants.builtins``.

Layered alongside ``tests/callbacks/invariants/test_dsl_sandbox.py`` (which
covers the DSL sandbox / ``evaluate_check``). This module instead pins the
seven default invariant *functions* registered under the ``("invariant", ...)``
registry namespace (DESIGN §18.6): they return ``True`` when the invariant
holds and ``False`` on violation.

Relocated from ``tests/test_invariants_dsl.py`` (builtin-invariants half; the
DSL/sandbox half is already subsumed by ``test_dsl_sandbox.py``).
"""

from __future__ import annotations

import torch

# Importing the builtins module registers the seven default invariants under
# the ``("invariant", ...)`` namespace as an import side effect. Import it
# explicitly so this module is self-contained regardless of plugin-load order.
import lighttrain.builtin_plugins.callbacks.invariants.builtins  # noqa: F401
from lighttrain.registry import get as registry_get


def test_seven_default_invariants_registered():
    """All seven documented default invariants are registered and callable."""
    names = (
        "loss_finite",
        "grad_norm_bounded",
        "lr_nonneg",
        "label_mask_nonzero",
        "param_count_stable",
        "dtype_stable",
        "batch_nonempty",
    )
    for n in names:
        assert callable(registry_get("invariant", n))


def test_loss_finite_builtin():
    """``loss_finite`` holds for finite loss, fails for inf/nan."""
    fn = registry_get("invariant", "loss_finite")
    assert fn(loss=torch.tensor([1.0])) is True
    assert fn(loss=torch.tensor([float("inf")])) is False


def test_grad_norm_bounded_builtin():
    """``grad_norm_bounded`` holds when ``metrics['grad_norm'] < max``."""
    fn = registry_get("invariant", "grad_norm_bounded")
    assert fn(metrics={"grad_norm": 5.0}, max=10) is True
    assert fn(metrics={"grad_norm": 50.0}, max=10) is False


def test_label_mask_nonzero_builtin():
    """``label_mask_nonzero`` holds when at least one non-(-100) label exists."""
    fn = registry_get("invariant", "label_mask_nonzero")
    good = {"labels": torch.tensor([[1, 2, -100], [3, -100, -100]])}
    bad = {"labels": torch.tensor([[-100, -100], [-100, -100]])}
    assert fn(batch=good) is True
    assert fn(batch=bad) is False


def test_batch_nonempty_builtin():
    """``batch_nonempty`` holds when ``input_ids`` has a nonzero leading dim."""
    fn = registry_get("invariant", "batch_nonempty")
    assert fn(batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)}) is True
    assert fn(batch={"input_ids": torch.zeros(0, 3, dtype=torch.long)}) is False
