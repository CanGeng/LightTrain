"""Invariants DSL — sandbox + builtin invariant coverage (DESIGN §18.6)."""

from __future__ import annotations

import pytest
import torch

from lighttrain.invariants import InvariantError, evaluate_check
from lighttrain.registry import get as registry_get


def test_finite_torch_loss():
    loss = torch.tensor([1.0, 2.0, 3.0])
    assert evaluate_check("torch.isfinite(loss).all()", loss=loss) is True


def test_nan_loss_violates():
    loss = torch.tensor([1.0, float("nan"), 3.0])
    assert evaluate_check("torch.isfinite(loss).all()", loss=loss) is False


def test_metric_dict_access():
    assert evaluate_check(
        "metrics['grad_norm'] < 100", metrics={"grad_norm": 50.0}
    ) is True
    assert evaluate_check(
        "metrics['grad_norm'] < 100", metrics={"grad_norm": 500.0}
    ) is False


def test_sandbox_blocks_import():
    with pytest.raises(InvariantError):
        evaluate_check("__import__('os')")


def test_sandbox_blocks_open():
    with pytest.raises(InvariantError):
        evaluate_check("open('foo')")


def test_sandbox_blocks_eval():
    with pytest.raises(InvariantError):
        evaluate_check("eval('1+1')")


def test_seven_default_invariants_registered():
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
    fn = registry_get("invariant", "loss_finite")
    assert fn(loss=torch.tensor([1.0])) is True
    assert fn(loss=torch.tensor([float("inf")])) is False


def test_grad_norm_bounded_builtin():
    fn = registry_get("invariant", "grad_norm_bounded")
    assert fn(metrics={"grad_norm": 5.0}, max=10) is True
    assert fn(metrics={"grad_norm": 50.0}, max=10) is False


def test_label_mask_nonzero_builtin():
    fn = registry_get("invariant", "label_mask_nonzero")
    good = {"labels": torch.tensor([[1, 2, -100], [3, -100, -100]])}
    bad = {"labels": torch.tensor([[-100, -100], [-100, -100]])}
    assert fn(batch=good) is True
    assert fn(batch=bad) is False


def test_batch_nonempty_builtin():
    fn = registry_get("invariant", "batch_nonempty")
    assert fn(batch={"input_ids": torch.zeros(2, 3, dtype=torch.long)}) is True
    assert fn(batch={"input_ids": torch.zeros(0, 3, dtype=torch.long)}) is False


# ---- dunder sandbox fix (bug fix verification) ------------------------------

def test_sandbox_blocks_dunder_mro_chain():
    """Classic sandbox escape via __class__.__mro__ must raise InvariantError."""
    with pytest.raises(InvariantError):
        evaluate_check("().__class__.__mro__[1].__subclasses__()")


def test_sandbox_blocks_dunder_class():
    with pytest.raises(InvariantError):
        evaluate_check("loss.__class__", loss=torch.tensor(1.0))


def test_sandbox_blocks_dunder_dict():
    with pytest.raises(InvariantError):
        evaluate_check("loss.__dict__", loss=torch.tensor(1.0))


def test_sandbox_allows_normal_expression():
    """Non-dunder expressions must still work after the AST check."""
    assert evaluate_check("loss < 5.0", loss=3.0) is True
    assert evaluate_check("torch.isfinite(loss).all()", loss=torch.tensor([1.0, 2.0])) is True
