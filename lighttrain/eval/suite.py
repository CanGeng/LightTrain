"""EvalSuite framework — Evaluator / EvalTask / EvalReport.

``RegressionGate`` (the registered ``invariant`` gate) moved to
``lighttrain.builtin_plugins.eval.regression_gate`` (DESIGN §3.3: framework in
core, concrete impl in builtin_plugins).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


__all__ = ["EvalReport", "EvalTask", "Evaluator"]
