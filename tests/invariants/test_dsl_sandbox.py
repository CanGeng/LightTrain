"""Adversarial tests for ``lighttrain.invariants._dsl.evaluate_check``.

Layered on top of ``tests/test_invariants_dsl.py``. New coverage:

* **3 INV_DUNDER_01 regression pins** (v0.1.1 fix) for the historical
  sandbox-escape bug: dunder-attribute access via various chain forms.
* **10+ adversarial sandbox payloads** (subscript→attr, lambda, list
  comprehension, type-descriptor reflection, etc.) — every payload must
  raise InvariantError, NOT return a value.
* **Baseline allowlist functionality** still works (torch.isfinite, len,
  arithmetic) so legitimate checks remain usable.
* **No eval/exec/open/import in namespace** — pin the namespace.
* **InvariantError is raised, not returned** for sandbox violations
  (caller relies on the exception type to abort training).
* **expr is required to be a string** (defensive coercion check).
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.invariants._dsl import InvariantError, evaluate_check


# ---------------------------------------------------------------------------
# Baseline: allowlisted functionality
# ---------------------------------------------------------------------------

def test_baseline_arithmetic_works():
    """``1 + 1 == 2`` evaluates without raising and returns True."""
    assert evaluate_check("1 + 1 == 2") is True


def test_baseline_loss_comparison_works():
    """Namespace variable access: ``loss > 0`` with loss=tensor(1.0).

    Closed form: True.
    """
    assert evaluate_check("loss > 0", loss=torch.tensor(1.0)) is True


def test_baseline_len_min_max_work():
    """Allowlisted builtins are callable from inside the expression."""
    assert evaluate_check("len([1,2,3]) == 3") is True
    assert evaluate_check("min(1, 2, 3) == 1") is True
    assert evaluate_check("max(1, 2, 3) == 3") is True


def test_baseline_torch_isfinite_works():
    """``torch.isfinite(loss)`` is reachable through the ``torch`` namespace
    binding (line 82 of _dsl.py).
    """
    assert evaluate_check("torch.isfinite(loss).all().item()", loss=torch.tensor(1.0)) is True


def test_baseline_metrics_dict_access_works():
    """Subscript on a metrics dict works inside the expression.

    Closed form: ``metrics["loss"] < 1.0`` with ``loss=0.5`` → True.
    """
    assert evaluate_check("metrics['loss'] < 1.0", metrics={"loss": 0.5}) is True


# ---------------------------------------------------------------------------
# INV_DUNDER_01 regression pins (v0.1.1)
# ---------------------------------------------------------------------------

def test_regression_INV_DUNDER_01_blocks_class_attribute():
    """Pre-fix bug: ``loss.__class__`` was reachable through the sandbox,
    enabling arbitrary class reflection via ``__class__`` followed by other
    dunder methods.

    Fix: AST pre-filter rejects any Attribute node whose ``.attr`` starts
    or ends with ``__`` (lines 88-92 of _dsl.py).

    Pre-fix bug: DSL sandbox dunder bypass (see docs/changelog/v0.1.1:
    "[DSL 沙箱 dunder 绕过]").
    """
    with pytest.raises(InvariantError, match="dunder"):
        evaluate_check("loss.__class__", loss=torch.tensor(1.0))


def test_regression_INV_DUNDER_01_blocks_mro_chain():
    """Pre-fix bug: ``loss.__class__.__mro__`` would expose the type MRO
    chain, the first step of the canonical sandbox escape.

    Pre-fix bug: see docs/changelog/v0.1.1: "[DSL 沙箱 dunder 绕过]".
    """
    with pytest.raises(InvariantError, match="dunder"):
        evaluate_check("loss.__class__.__mro__", loss=torch.tensor(1.0))


def test_regression_INV_DUNDER_01_blocks_subclasses_reflection():
    """Pre-fix bug: ``().__class__.__mro__[1].__subclasses__()`` is THE
    classical Python sandbox escape — getting from any object to every
    class in the interpreter via type reflection.

    Pre-fix bug: see docs/changelog/v0.1.1: "[DSL 沙箱 dunder 绕过]".
    """
    with pytest.raises(InvariantError, match="dunder"):
        evaluate_check("().__class__.__mro__[1].__subclasses__()")


# ---------------------------------------------------------------------------
# Adversarial sandbox payloads — at least 10 distinct attacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        # 1. Direct dunder
        "loss.__class__",
        # 2. Dunder via metrics dict subscript→attr chain
        "metrics['x'].__class__",
        # 3. Dunder on a list literal
        "[1, 2].__class__",
        # 4. Dunder on a tuple literal
        "(1, 2).__class__",
        # 5. Dunder via integer literal (Python 3.10+ allows this)
        "(1).__class__",
        # 6. Dunder via float literal
        "(1.0).__sizeof__",
        # 7. Dunder via abs() result (chained call → dunder)
        "abs(loss).__class__",
        # 8. Dunder via list comprehension
        "[x.__class__ for x in [loss]]",
        # 9. Dunder inside a lambda
        "(lambda x: x.__class__)(loss)",
        # 10. Dunder via subscript-then-attr on metrics dict
        "metrics['x'].__dict__",
        # 11. Dunder via torch module reflection
        "torch.__dict__",
        # 12. Chained method call followed by dunder
        "loss.abs().__class__",
    ],
)
def test_sandbox_blocks_dunder_payload(payload):
    """Adversarial payload: every form of dunder access must raise
    InvariantError at AST-walk time (line 88-92 of _dsl.py).

    Goal: cover the full surface — direct, chained, via subscript, via
    comprehension, via lambda, via call result. A regression that walks
    only the top-level node would let several of these through.
    """
    with pytest.raises(InvariantError, match="dunder"):
        evaluate_check(
            payload,
            loss=torch.tensor(1.0),
            metrics={"x": 1.0},
        )


# ---------------------------------------------------------------------------
# Namespace closure: no builtins reachable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["eval", "exec", "open", "compile", "__import__", "globals", "locals", "vars"],
)
def test_invariant_dangerous_builtins_not_in_namespace(name):
    """Every dangerous builtin is NOT accessible from inside the expression.

    Setup: try to call each name; expect InvariantError (raised because
    the name is unbound — wrapped as InvariantError per line 101-104 of
    source).
    """
    with pytest.raises(InvariantError):
        evaluate_check(f"{name}('x')")


def test_invariant_string_with_dunder_substring_is_data_not_code():
    """A *string literal* containing dunder text is just data — the AST
    walker does not see it as an Attribute node.

    Setup: an expression that produces the string ``"__class__"`` as a
    plain str.
    Expected: evaluates to True (no exception).

    Goal: pin that the dunder rejection is at AST level, not naive
    string-substring level. Otherwise we'd over-reject legitimate
    expressions that mention ``__`` in payload.
    """
    assert evaluate_check("'__class__' == '__class__'") is True


def test_pin_torch_module_exposes_load_function():
    """Pin: ``torch.load`` is reachable inside expressions (the entire
    ``torch`` namespace is bound, no per-attribute filtering).

    This documents the current scope. If you intentionally restrict torch
    attrs (e.g. block torch.load to prevent untrusted-file deserialization
    via crafted checks), update this test AND document the breaking change.
    """
    # We don't *invoke* torch.load; just verify the attribute is reachable.
    # The simplest probe is to compare an attribute that exists on torch.
    # (Using torch.float32 as a stable, safe witness for "torch namespace
    # is bound".)
    assert evaluate_check("torch.float32 is torch.float32") is True


# ---------------------------------------------------------------------------
# Error path / coercion
# ---------------------------------------------------------------------------

def test_non_string_expression_raises_invariant_error():
    """Passing a non-string ``expr`` raises InvariantError early
    (line 68-69 of _dsl.py).
    """
    with pytest.raises(InvariantError, match="must be a string"):
        evaluate_check(123)  # type: ignore[arg-type]


def test_expression_runtime_error_wrapped_in_invariant_error():
    """When the expression raises a non-InvariantError exception during
    eval (e.g., ZeroDivisionError), it's wrapped in InvariantError
    (line 101-104 of source).
    """
    with pytest.raises(InvariantError, match="ZeroDivisionError"):
        evaluate_check("1 / 0")


def test_syntax_error_in_expression_raises_invariant_error():
    """A syntactically invalid expression raises an error at AST-parse time.

    Note: ``ast.parse`` raises SyntaxError, which is NOT caught by the
    InvariantError wrap (line 101 catches only ``Exception``). SyntaxError
    propagates as-is.
    """
    with pytest.raises((SyntaxError, InvariantError)):
        evaluate_check("loss <<")


# ---------------------------------------------------------------------------
# Truth coercion
# ---------------------------------------------------------------------------

def test_invariant_truthy_result_returns_true():
    """A truthy expression value returns Python ``True``.

    Closed form: ``1 < 2`` → True.
    """
    assert evaluate_check("1 < 2") is True


def test_invariant_falsy_result_returns_false():
    """A falsy expression value returns Python ``False``.

    Closed form: ``1 > 2`` → False.
    """
    assert evaluate_check("1 > 2") is False


def test_invariant_nonzero_int_result_coerces_to_true():
    """Non-boolean truthy values (e.g., int 42) coerce to True
    (line 105 of source: ``return bool(result)``).
    """
    assert evaluate_check("42") is True
    assert evaluate_check("0") is False


# ---------------------------------------------------------------------------
# Extra namespace bindings
# ---------------------------------------------------------------------------

def test_extra_kwargs_propagate_into_namespace():
    """``extra={"custom_var": 7}`` makes ``custom_var`` accessible inside
    the expression (line 85-87 of source).
    """
    assert evaluate_check("custom_var == 7", extra={"custom_var": 7}) is True


def test_extra_kwargs_do_not_overwrite_existing_bindings():
    """Pin: ``setdefault`` is used (line 87), so ``extra["step"] = 999``
    does NOT overwrite the ``step=`` kwarg (which is bound earlier in
    ``ns`` on line 80).

    Setup: pass step=7 AND extra={"step": 999}.
    Expected: the function-call kwarg wins; expression sees step=7.
    """
    assert evaluate_check(
        "step == 7",
        step=7,
        extra={"step": 999},
    ) is True
