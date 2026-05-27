"""Optuna TPE sweep backend.

Registered as ``@register("sweep_backend", "optuna")``.

Wraps Optuna's ``TPESampler`` so ``SweepRunner(strategy="optuna")`` can use
Bayesian optimisation instead of grid / random search.

Install:
    pip install -e '.[sweep]'   # adds optuna to your env

Usage (sweep spec):
    strategy: optuna   # passed to lighttrain sweep --strategy optuna
    n_trials: 20
    params:
      optim.lr: {low: 1e-5, high: 1e-2, log: true}
      optim.weight_decay: {low: 0.0, high: 0.1}
"""

from __future__ import annotations

from typing import Any

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as _import_err:
    raise ImportError(
        "frontier_plugins.sweep_backends.optuna requires optuna. "
        "Install with: pip install -e '.[sweep]'"
    ) from _import_err

from lighttrain.registry import register


@register("sweep_backend", "optuna")
class OptunaSearcher:
    """Bayesian hyperparameter search via Optuna TPE sampler.

    ``SweepRunner`` calls :meth:`all_suggestions` to get a pre-computed list
    of *n_trials* configs upfront (eager mode).  This means Optuna's adaptive
    sampling is only as adaptive as the number of completed trials before the
    next batch — for single-GPU sequential sweeps this is equivalent to
    running one study with ``n_trials`` calls.
    """

    def __init__(
        self,
        params: dict[str, Any],
        n_trials: int,
        direction: str = "minimize",
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.n_trials = n_trials
        self.direction = direction
        self.seed = seed

        sampler = optuna.samplers.TPESampler(seed=seed)
        self._study = optuna.create_study(direction=direction, sampler=sampler)
        self._trial_map: dict[int, optuna.Trial] = {}

    # ---------------------------------------------------------------- suggest

    def suggest(self, trial_id: int) -> dict[str, Any]:
        """Return a config dict for the next trial."""
        trial = self._study.ask()
        self._trial_map[trial_id] = trial
        return self._trial_to_config(trial)

    def _trial_to_config(self, trial: "optuna.Trial") -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                cfg[k] = trial.suggest_categorical(k, v)
            elif isinstance(v, dict):
                low = float(v["low"])
                high = float(v["high"])
                typ = v.get("type", "float")
                log = bool(v.get("log", False))
                if typ == "int":
                    cfg[k] = trial.suggest_int(k, int(low), int(high), log=log)
                else:
                    cfg[k] = trial.suggest_float(k, low, high, log=log)
            else:
                cfg[k] = v
        return cfg

    # ---------------------------------------------------------------- report / prune

    def report(self, trial_id: int, metric: float) -> None:
        """Tell Optuna the result for *trial_id*."""
        trial = self._trial_map.get(trial_id)
        if trial is not None:
            self._study.tell(trial, metric)

    def should_prune(self, trial_id: int) -> bool:
        """Return True if Optuna's MedianPruner wants to stop this trial."""
        trial = self._trial_map.get(trial_id)
        if trial is None:
            return False
        return trial.should_prune()

    # ---------------------------------------------------------------- batch mode

    def all_suggestions(self) -> list[dict[str, Any]]:
        """Pre-generate *n_trials* configs (eager mode for sequential sweeps)."""
        configs: list[dict[str, Any]] = []
        for i in range(self.n_trials):
            configs.append(self.suggest(i))
        return configs

    @property
    def best_params(self) -> dict[str, Any] | None:
        try:
            return self._study.best_params
        except Exception:
            return None

    @property
    def best_value(self) -> float | None:
        try:
            return self._study.best_value
        except Exception:
            return None


__all__ = ["OptunaSearcher"]
