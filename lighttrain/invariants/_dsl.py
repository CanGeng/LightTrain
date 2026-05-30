"""Invariant DSL.

A *guarded* expression evaluator for short ``check:`` strings declared in
recipes. The expressions are evaluated against a controlled namespace that
contains only:

* ``loss``, ``outputs``, ``batch``, ``model``, ``optimizer``, ``scheduler``,
  ``metrics``, ``step``, ``epoch`` — the per-step variables a check might
  inspect.
* ``torch`` (the module itself; useful for ``torch.isfinite``).
* ``len``, ``min``, ``max``, ``sum``, ``abs``, ``any``, ``all`` — a small
  bounded set of pure builtins.

The implementation goes through :func:`eval` with ``{"__builtins__": {}}`` and
rejects dunder attribute access at the AST level, so the classic
``().__class__.__subclasses__()`` breakout and ``__import__`` / ``open`` /
``exec`` are unreachable from inside the expression (see
``tests/invariants/test_dsl_sandbox.py``).

**This is NOT a security boundary for untrusted input.** The *entire* ``torch``
module is bound, so primitives such as ``torch.load`` (pickle → arbitrary code
execution) and ``torch.save`` (arbitrary file write) are reachable with plain,
dunder-free expressions. The guard exists to stop accidental footguns and the
generic Python escapes — it assumes the ``check:`` strings come from a
**trusted recipe author**, the same trust already required by ``user_modules``
/ registered components. Do NOT evaluate ``check:`` strings submitted by
untrusted parties (e.g. a multi-tenant service) on the strength of this guard.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

import torch


class InvariantError(RuntimeError):
    """Raised when an invariant violation should abort training."""


_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
}


def evaluate_check(
    expr: str,
    *,
    loss: Any = None,
    outputs: Any = None,
    batch: Any = None,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    metrics: Mapping[str, float] | None = None,
    step: int = 0,
    epoch: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> bool:
    """Evaluate ``expr`` in a restricted namespace; return a Python truth value.

    The expression has read access to the named arguments and to ``torch``;
    no other globals are available. Mutation through ``expr`` would only
    affect Python-level local copies (e.g. dict subscript writes), not the
    underlying StepContext, but we still treat the namespace as read-only by
    convention.
    """
    if not isinstance(expr, str):
        raise InvariantError(f"check expression must be a string, got {type(expr)!r}")
    ns: dict[str, Any] = dict(_SAFE_BUILTINS)
    ns.update(
        {
            "loss": loss,
            "outputs": outputs,
            "batch": batch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "metrics": dict(metrics) if metrics else {},
            "step": int(step),
            "epoch": int(epoch),
            "torch": torch,
        }
    )
    if extra:
        for k, v in extra.items():
            ns.setdefault(k, v)
    for _node in ast.walk(ast.parse(expr, mode="eval")):
        if isinstance(_node, ast.Attribute) and (
            _node.attr.startswith("__") or _node.attr.endswith("__")
        ):
            raise InvariantError(f"invariant check: dunder attribute access not allowed: {_node.attr!r}")
    try:
        result = eval(  # noqa: S307 — guarded: builtins stripped, dunders AST-rejected; trusted authors only
            expr,
            {"__builtins__": {}},
            ns,
        )
    except InvariantError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvariantError(
            f"invariant check {expr!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    return bool(result)


__all__ = ["InvariantError", "evaluate_check"]
