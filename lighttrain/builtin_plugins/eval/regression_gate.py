"""RegressionGate — metric regression gate for CI / sweep early-stopping.

Registered under the ``invariant`` category (reusing the invariants DSL) so
``InvariantsCallback`` can pick it up by name. The eval framework
(``Evaluator`` / ``EvalTask`` / ``EvalReport``) stays in ``lighttrain.eval.suite``
(DESIGN §3.3: framework in core, this concrete impl in builtin_plugins).
"""

from __future__ import annotations

from lighttrain.eval.suite import EvalReport
from lighttrain.registry import register


@register("invariant", "regression_gate")
class RegressionGate:
    """Metric regression gate for CI / sweep early-stopping.

    On each call to :meth:`check`, evaluates ``metric <op> threshold``. When the
    condition is violated, raises :class:`lighttrain.invariants.InvariantError`
    (``action="abort"``), warns (``action="warn"``), or skips (``action="skip"``).

    Parameters
    ----------
    metric_name : str
        The metric key to watch (e.g. ``"val_loss"``).
    threshold : float
        Boundary value.
    op : str
        Comparison operator — ``"<"``, ``"<="``, ``">"``, ``">="``, ``"=="``, ``"!="``
        applied as ``metric_value <op> threshold``.  Use ``"<"`` to ensure the
        metric stays below the threshold (i.e. block regression when metric goes up).
    action : str
        ``"abort"`` (raise), ``"warn"`` (log warning only), or ``"skip"`` (no-op).
    history_window : int
        Only trigger if the metric exceeds the threshold for ``history_window``
        consecutive checks (0 = trigger immediately).
    """

    def __init__(
        self,
        *,
        metric_name: str,
        threshold: float,
        op: str = "<",
        action: str = "abort",
        history_window: int = 0,
    ) -> None:
        self.metric_name = str(metric_name)
        self.threshold = float(threshold)
        self.op = str(op)
        self.action = str(action)
        self.history_window = int(history_window)
        self._fail_count = 0
        self._last_value: float | None = None

    def check(
        self,
        report: EvalReport | dict[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Evaluate the gate against a report or metric dict.

        Raises
        ------
        InvariantError
            When ``action="abort"`` and the condition is violated.
        """
        metrics = report.metrics if isinstance(report, EvalReport) else report
        if self.metric_name not in metrics:
            return  # metric absent — skip silently

        value = float(metrics[self.metric_name])
        self._last_value = value
        passed = self._evaluate(value)

        if passed:
            self._fail_count = 0
            return

        self._fail_count += 1
        if self._fail_count <= self.history_window:
            return  # not yet enough consecutive failures

        msg = (
            f"RegressionGate [{self.metric_name} {self.op} {self.threshold}] "
            f"FAILED: value={value:.6g} at step={step}"
        )

        if self.action == "abort":
            try:
                from lighttrain.invariants import InvariantError
                raise InvariantError(msg)
            except ImportError:
                raise RuntimeError(msg) from None
        elif self.action == "warn":
            import warnings
            warnings.warn(msg, stacklevel=2)
        # "skip" — no-op

    def _evaluate(self, value: float) -> bool:
        t = self.threshold
        if self.op == "<":
            return value < t
        elif self.op == "<=":
            return value <= t
        elif self.op == ">":
            return value > t
        elif self.op == ">=":
            return value >= t
        elif self.op == "==":
            return value == t
        elif self.op == "!=":
            return value != t
        raise ValueError(f"RegressionGate: unknown op {self.op!r}.")

    @property
    def last_value(self) -> float | None:
        return self._last_value


__all__ = ["RegressionGate"]
