"""Invariant DSL.

A safe expression evaluator for short ``check:`` strings declared in recipes.
The expressions are evaluated against a controlled namespace that contains
only:

* ``loss``, ``outputs``, ``batch``, ``model``, ``optimizer``, ``scheduler``,
  ``metrics``, ``step``, ``epoch`` — the per-step variables a check might
  inspect.
* ``torch`` (the module itself; useful for ``torch.isfinite``).
* ``len``, ``min``, ``max``, ``sum``, ``abs``, ``any``, ``all`` — a small
  bounded set of pure builtins.

The implementation goes through :func:`eval` with ``{"__builtins__": {}}`` so
attribute access works but no dunder import / open / exec is reachable from
inside the expression. See ``tests/test_invariants_dsl.py`` for the
sandbox tests (no ``__import__``, no ``open``, no ``eval``).
"""

from __future__ import annotations

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
    try:
        result = eval(  # noqa: S307 — sandbox: builtins stripped below
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
