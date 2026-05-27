"""EvalSuite — Evaluator / EvalTask / EvalReport / RegressionGate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from ..registry import register


# ---------------------------------------------------------------------------
# Protocols & data classes
# ---------------------------------------------------------------------------


@runtime_checkable
class EvalTask(Protocol):
    """An evaluation task that can be run against a model at a given step."""

    name: str

    def run(
        self,
        model: Any,
        *,
        device: Any | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class EvalReport:
    """Result of running one or more EvalTasks at a given training step.

    Attributes
    ----------
    task_name : str
        Name of the task (or ``"suite"`` for multi-task aggregate).
    metrics : dict
        Flat metric name → scalar float mapping.
    step : int or None
        Training step at which the report was generated.
    timestamp : float
        Unix timestamp (``time.time()``).
    """

    task_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    step: int | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Manages multiple :class:`EvalTask` instances and runs them periodically.

    Parameters
    ----------
    tasks :
        List of EvalTask instances.
    eval_every_n_steps : int
        Run all tasks every N training steps (0 = never).
    on_report :
        Optional callback ``(EvalReport) -> None`` called after each run.
    """

    def __init__(
        self,
        tasks: list[Any],
        *,
        eval_every_n_steps: int = 500,
        on_report: Callable[[EvalReport], None] | None = None,
    ) -> None:
        self.tasks = list(tasks)
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.on_report = on_report
        self._last_eval_step: int = -1

    def should_eval(self, step: int) -> bool:
        if self.eval_every_n_steps <= 0:
            return False
        return step % self.eval_every_n_steps == 0 and step != self._last_eval_step

    def run(
        self,
        model: Any,
        step: int,
        *,
        device: Any | None = None,
        force: bool = False,
    ) -> EvalReport | None:
        """Run all tasks if ``should_eval(step)`` or ``force=True``.

        Returns
        -------
        :class:`EvalReport` if tasks were run, ``None`` otherwise.
        """
        if not force and not self.should_eval(step):
            return None

        self._last_eval_step = step
        combined_metrics: dict[str, float] = {}

        for task in self.tasks:
            try:
                result = task.run(model, device=device, step=step)
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(f"EvalTask {getattr(task, 'name', '?')} failed: {exc}", stacklevel=2)
                continue

            task_name = getattr(task, "name", "?")
            # Prefix metrics with task name if multiple tasks.
            if len(self.tasks) > 1:
                for k, v in result.items():
                    if k != "task_name" and isinstance(v, (int, float)):
                        combined_metrics[f"{task_name}/{k}"] = float(v)
            else:
                for k, v in result.items():
                    if k != "task_name" and isinstance(v, (int, float)):
                        combined_metrics[k] = float(v)

        report = EvalReport(
            task_name="suite" if len(self.tasks) != 1 else getattr(self.tasks[0], "name", "eval"),
            metrics=combined_metrics,
            step=step,
        )
        if self.on_report is not None:
            try:
                self.on_report(report)
            except Exception:  # noqa: BLE001
                pass
        return report


# ---------------------------------------------------------------------------
# RegressionGate
# ---------------------------------------------------------------------------


@register("invariant", "regression_gate")
class RegressionGate:
    """Metric regression gate for CI / sweep early-stopping.

    Reuses the invariants DSL — registered as an ``"invariant"``
    so ``InvariantCallback`` can pick it up by name.

    On each call to :meth:`check`, evaluates ``metric <op> threshold``.  When
    the condition is violated, raises :class:`~lighttrain.diagnostics.invariants.InvariantError`
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
        report: "EvalReport | dict[str, float]",
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
                from ..invariants import InvariantError
                raise InvariantError(msg)
            except ImportError:
                raise RuntimeError(msg)
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


__all__ = ["EvalReport", "EvalTask", "Evaluator", "RegressionGate"]
