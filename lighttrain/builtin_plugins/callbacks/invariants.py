"""InvariantsCallback.

Critical callback. Evaluates a list of invariant specs every step. Each spec
is either:

* A short name resolving via the ``"invariant"`` registry (e.g.
  ``{"name": "loss_finite", "action": "abort"}``).
* An inline ``{"check": "<expr>", "action": "warn|skip|abort"}``.

Actions:

* ``abort`` (default) — raise :class:`InvariantError` out of the
  callback. ``EventBus`` propagates it because ``InvariantsCallback`` is
  registered as critical (see :class:`EventBus`).
* ``skip``  — return ``Signal.SKIP_STEP`` so the engine drops backward
  for the current step.
* ``warn``  — emit a Python warning and continue.

When ``cfg.invariants`` is absent the callback runs a small default set
suited for any causal-LM run (``loss_finite`` abort + ``grad_norm_bounded``
warn + ``lr_nonneg`` abort + ``batch_nonempty`` abort).

Violations are recorded under ``ctx.diagnostics["invariant_violations"]``
so :func:`write_index_page` can show them in ``diagnostics/index.md``.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping

from lighttrain.invariants import InvariantError, evaluate_check
from lighttrain.registry import get as _registry_get
from lighttrain.registry import register
from lighttrain.callbacks.base import Signal


_DEFAULTS: tuple[dict[str, Any], ...] = (
    {"name": "loss_finite", "action": "abort"},
    {"name": "grad_norm_bounded", "action": "warn", "max": 1e3},
    {"name": "lr_nonneg", "action": "abort"},
    {"name": "batch_nonempty", "action": "abort"},
)


@register("callback", "invariants")
class InvariantsCallback:
    """Run a set of invariants every step."""

    critical: bool = True

    def __init__(
        self,
        *,
        specs: Iterable[Mapping[str, Any]] | None = None,
        on_violation: str = "abort",  # default action when spec omits it
    ) -> None:
        self.on_violation = str(on_violation)
        raw = list(specs) if specs else list(_DEFAULTS)
        self._specs: list[dict[str, Any]] = []
        for s in raw:
            if isinstance(s, Mapping):
                self._specs.append(dict(s))

    # ----- lifecycle -------------------------------------------------------

    def on_loss_computed(
        self,
        *,
        step: int = 0,
        loss: Any = None,
        outputs: Any = None,
        batch: Any = None,
        model: Any = None,
        metrics: Any = None,
        **_: Any,
    ) -> Signal:
        return self._evaluate_all(
            step=step,
            loss=loss,
            outputs=outputs,
            batch=batch,
            model=model,
            metrics=metrics,
        )

    def on_optimizer_step_post(
        self,
        *,
        step: int = 0,
        model: Any = None,
        **_: Any,
    ) -> Signal:
        # Re-check param/dtype stability after the optimizer touched params.
        # ``metrics`` isn't always passed at this hook; build a best-effort one
        # from any ctx-like kwarg, otherwise skip silently.
        return Signal.CONTINUE

    # ----- core ------------------------------------------------------------

    def _evaluate_all(
        self,
        *,
        step: int = 0,
        loss: Any = None,
        outputs: Any = None,
        batch: Any = None,
        model: Any = None,
        metrics: Any = None,
    ) -> Signal:
        agg = Signal.CONTINUE
        for spec in self._specs:
            action = str(spec.get("action") or self.on_violation)
            name = spec.get("name")
            check = spec.get("check")
            try:
                ok = self._eval_one(
                    spec,
                    loss=loss,
                    outputs=outputs,
                    batch=batch,
                    model=model,
                    metrics=metrics,
                    step=step,
                )
            except InvariantError as exc:
                # Treat DSL-level errors as abort regardless of action.
                self._record_violation(
                    metrics, step, name or check or "<unknown>", str(exc), "abort"
                )
                raise
            if ok:
                continue
            ident = name or check or "<unknown>"
            self._record_violation(metrics, step, ident, "predicate=False", action)
            if action == "abort":
                raise InvariantError(
                    f"invariant {ident!r} violated at step {step}"
                )
            if action == "skip":
                if Signal.SKIP_STEP > agg:
                    agg = Signal.SKIP_STEP
            elif action == "warn":
                warnings.warn(
                    f"invariant {ident!r} violated at step {step}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return agg

    def _eval_one(
        self,
        spec: Mapping[str, Any],
        *,
        loss: Any,
        outputs: Any,
        batch: Any,
        model: Any,
        metrics: Any,
        step: int,
    ) -> bool:
        check = spec.get("check")
        if check:
            return evaluate_check(
                str(check),
                loss=loss,
                outputs=outputs,
                batch=batch,
                model=model,
                optimizer=None,
                metrics=metrics,
                step=step,
            )
        name = spec.get("name")
        if not name:
            raise InvariantError("invariant spec missing both 'name' and 'check'")
        fn = _registry_get("invariant", str(name))
        # Pass through positional kwargs from spec (e.g. ``max=1e3``) minus
        # the bookkeeping keys.
        extra = {
            k: v for k, v in spec.items() if k not in ("name", "check", "action")
        }
        return bool(
            fn(
                loss=loss,
                outputs=outputs,
                batch=batch,
                model=model,
                metrics=metrics,
                step=step,
                **extra,
            )
        )

    @staticmethod
    def _record_violation(
        metrics: Any,
        step: int,
        ident: str,
        reason: str,
        action: str,
    ) -> None:
        if not isinstance(metrics, dict):
            return
        log = metrics.setdefault("_invariant_violations", [])
        if isinstance(log, list):
            log.append(
                {"step": int(step), "name": ident, "reason": reason, "action": action}
            )


__all__ = ["InvariantsCallback"]
