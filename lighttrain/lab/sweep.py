"""Hyperparameter sweep.

Orchestrates grid, random, and median-stop strategies.  Each trial runs as a
``lighttrain train`` subprocess so the registry and GPU state are fully
isolated.  Optuna is available as an opt-in plugin at
``lighttrain.plugins.sweep_backends.optuna``.

Sweep spec YAML schema::

    name: lr_sweep
    metric: loss              # key in logs/metrics.jsonl
    direction: minimize       # or maximize
    n_trials: 12              # random strategy only
    seed: 42
    trial_timeout_s: 3600     # optional per-trial wall-clock cap
    params:
      optim.lr: [1e-4, 3e-4, 1e-3]             # grid: list of values
      optim.weight_decay: {low: 0.0, high: 0.1} # random: {low, high[, log, type]}
    stop:
      type: median   # or "none"
      grace: 3       # completed trials before pruning kicks in
"""

from __future__ import annotations

import itertools
import json
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _grid_configs(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of all list-valued params."""
    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for k, v in params.items():
        if isinstance(v, list):
            keys.append(k)
            value_lists.append(v)
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _random_configs(
    params: dict[str, Any],
    n_trials: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample *n_trials* configs uniformly from the param space."""
    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(n_trials):
        cfg: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, list):
                cfg[k] = rng.choice(v)
            elif isinstance(v, Mapping):
                low = float(v.get("low", 0.0))
                high = float(v.get("high", 1.0))
                typ = v.get("type", "float")
                log_scale: bool = bool(v.get("log", False))
                if typ == "int":
                    cfg[k] = rng.randint(int(low), int(high))
                elif log_scale:
                    import math

                    cfg[k] = math.exp(rng.uniform(math.log(low), math.log(high)))
                else:
                    cfg[k] = rng.uniform(low, high)
            else:
                cfg[k] = v
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Run-dir discovery & metric reading
# ---------------------------------------------------------------------------


def _find_run_dir(trial_root: Path) -> Path | None:
    """Return the most recently-created run dir inside *trial_root*."""
    if not trial_root.exists():
        return None
    subdirs = sorted(trial_root.iterdir())
    for d in reversed(subdirs):
        if d.is_dir():
            return d
    return None


def _read_final_metric(run_dir: Path, metric_key: str) -> float | None:
    """Return the last occurrence of *metric_key* in ``logs/metrics.jsonl``."""
    for candidate in (run_dir / "logs" / "metrics.jsonl", run_dir / "metrics.jsonl"):
        if not candidate.exists():
            continue
        last_val: float | None = None
        try:
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if metric_key in entry:
                            last_val = float(entry[metric_key])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
        except OSError:
            continue
        if last_val is not None:
            return last_val
    return None


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def _compute_sensitivity(
    trials: list["TrialResult"],
    params: dict[str, Any],
) -> dict[str, float]:
    """Estimate each param's impact as a normalised absolute correlation."""
    ok = [t for t in trials if t.metric is not None]
    if len(ok) < 2:
        return {}
    metrics = [t.metric for t in ok]  # type: ignore[misc]
    mean_m = sum(metrics) / len(metrics)
    var_m = sum((m - mean_m) ** 2 for m in metrics) / len(metrics)
    if var_m < 1e-14:
        return {k: 0.0 for k in params}

    result: dict[str, float] = {}
    for pname in params:
        pvals = [t.config_overrides.get(pname) for t in ok]
        try:
            numeric = [float(v) for v in pvals]  # type: ignore[arg-type]
            mean_p = sum(numeric) / len(numeric)
            var_p = sum((p - mean_p) ** 2 for p in numeric) / len(numeric)
            if var_p < 1e-14:
                result[pname] = 0.0
                continue
            cov = sum(
                (p - mean_p) * (m - mean_m)
                for p, m in zip(numeric, metrics)
            ) / len(metrics)
            r = abs(cov / (var_p**0.5 * var_m**0.5))
            result[pname] = round(min(1.0, r), 4)
        except (TypeError, ValueError):
            # Categorical: between-group vs total variance
            groups: dict[Any, list[float]] = {}
            for v, m in zip(pvals, metrics):
                groups.setdefault(v, []).append(m)
            if len(groups) < 2:
                result[pname] = 0.0
                continue
            group_means = [sum(g) / len(g) for g in groups.values()]
            between_var = sum((gm - mean_m) ** 2 for gm in group_means) / len(group_means)
            result[pname] = round(min(1.0, between_var / var_m), 4)
    return result


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    trial_id: int
    config_overrides: dict[str, Any]
    metric: float | None
    status: str  # "ok" | "pruned" | "failed"
    run_dir: Path | None


@dataclass
class SweepReport:
    sweep_name: str
    strategy: str
    trials: list[TrialResult]
    best_config: dict[str, Any]
    best_metric: float | None
    direction: str
    sensitivity: dict[str, float]
    report_path: Path | None = None


# ---------------------------------------------------------------------------
# SweepRunner
# ---------------------------------------------------------------------------


class SweepRunner:
    """Orchestrate hyperparameter trials against a lighttrain recipe.

    Each trial is a ``lighttrain train`` subprocess with config overrides
    injected as OmegaConf ``++key=value`` arguments.  Metrics are read from
    the trial's ``logs/metrics.jsonl`` after the process exits.
    """

    def __init__(
        self,
        base_cfg_path: Path,
        sweep_cfg_path: Path,
        strategy: str = "grid",
    ) -> None:
        self.base_cfg_path = Path(base_cfg_path).resolve()
        self.sweep_cfg = _load_yaml(Path(sweep_cfg_path))
        self.strategy = strategy

        self.params: dict[str, Any] = self.sweep_cfg.get("params", {})
        self.metric_key: str = self.sweep_cfg.get("metric", "loss")
        self.direction: str = self.sweep_cfg.get("direction", "minimize")
        self.n_trials: int = int(self.sweep_cfg.get("n_trials", 10))
        self.stop_cfg: dict[str, Any] = self.sweep_cfg.get("stop") or {}
        self.sweep_name: str = self.sweep_cfg.get("name", "sweep")
        self.seed: int = int(self.sweep_cfg.get("seed", 42))
        self.trial_timeout: float | None = (
            float(self.sweep_cfg["trial_timeout_s"])
            if "trial_timeout_s" in self.sweep_cfg
            else None
        )

        base_cfg = _load_yaml(self.base_cfg_path)
        self.run_root = Path(base_cfg.get("run_root", "runs")).resolve()

    # ---------------------------------------------------------------- config generation

    def _generate_configs(self) -> list[dict[str, Any]]:
        if self.strategy == "grid":
            return _grid_configs(self.params)
        if self.strategy == "random":
            return _random_configs(self.params, self.n_trials, self.seed)
        if self.strategy == "optuna":
            return self._optuna_configs()
        raise ValueError(
            f"unknown sweep strategy {self.strategy!r}; "
            "expected 'grid', 'random', or 'optuna'"
        )

    def _optuna_configs(self) -> list[dict[str, Any]]:
        try:
            from lighttrain.config._components import import_all_components
            from lighttrain.registry import RegistryError
            from lighttrain.registry import get as _get

            # The sweep CLI does not call load_config(), so the optuna plugin
            # may not be registered yet — populate the registry before lookup.
            import_all_components()
            searcher_cls = _get("sweep_backend", "optuna")
            searcher = searcher_cls(
                params=self.params,
                n_trials=self.n_trials,
                direction=self.direction,
                seed=self.seed,
            )
            return searcher.all_suggestions()
        except (ImportError, RegistryError) as exc:
            raise RuntimeError(
                f"Optuna sweep backend unavailable: {exc}. "
                "Install with: pip install -e '.[sweep]' and ensure "
                "lighttrain.plugins.sweep_backends.optuna is importable."
            ) from exc

    # ---------------------------------------------------------------- trial execution

    def _trial_exp(self, trial_id: int) -> str:
        return f"{self.sweep_name}_trial_{trial_id:03d}"

    def _run_trial(self, trial_id: int, overrides: dict[str, Any]) -> TrialResult:
        trial_exp = self._trial_exp(trial_id)
        sweep_run_root = self.run_root / f"sweep_{self.sweep_name}"

        cmd = [
            sys.executable, "-m", "lighttrain", "train",
            "-c", str(self.base_cfg_path),
        ]
        for k, v in overrides.items():
            cmd.append(f"++{k}={v}")
        cmd.append(f"++exp={trial_exp}")
        cmd.append(f"++run_root={sweep_run_root}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.trial_timeout,
            )
            succeeded = proc.returncode == 0
        except subprocess.TimeoutExpired:
            return TrialResult(trial_id, overrides, None, "failed", None)
        except Exception:
            return TrialResult(trial_id, overrides, None, "failed", None)

        run_dir = _find_run_dir(sweep_run_root / trial_exp)
        metric = _read_final_metric(run_dir, self.metric_key) if run_dir else None
        status = "ok" if succeeded else "failed"
        return TrialResult(trial_id, overrides, metric, status, run_dir)

    # ---------------------------------------------------------------- stopping

    def _apply_median_stop(self, trials: list[TrialResult]) -> None:
        """Retroactively mark below-median trials as 'pruned'."""
        stop_type = self.stop_cfg.get("type", "none")
        if stop_type not in ("median", "asha"):
            return
        grace = int(self.stop_cfg.get("grace", 3))
        ok = [t for t in trials if t.status == "ok" and t.metric is not None]
        if len(ok) < grace:
            return
        median_val = statistics.median(t.metric for t in ok)  # type: ignore[misc]
        threshold = 1.2 if self.direction == "minimize" else 0.8
        for t in ok:
            assert t.metric is not None
            if self.direction == "minimize" and t.metric > median_val * threshold:
                t.status = "pruned"
            elif self.direction != "minimize" and t.metric < median_val * threshold:
                t.status = "pruned"

    # ---------------------------------------------------------------- main entry

    def run(self) -> SweepReport:
        """Execute all trials and return a :class:`SweepReport`."""
        configs = self._generate_configs()
        trials: list[TrialResult] = []
        for i, overrides in enumerate(configs):
            result = self._run_trial(i, overrides)
            trials.append(result)

        self._apply_median_stop(trials)

        ok = [t for t in trials if t.status == "ok" and t.metric is not None]
        if ok:
            best = (
                min(ok, key=lambda t: t.metric)  # type: ignore[arg-type]
                if self.direction == "minimize"
                else max(ok, key=lambda t: t.metric)  # type: ignore[arg-type]
            )
            best_config = best.config_overrides
            best_metric: float | None = best.metric
        else:
            best_config, best_metric = {}, None

        sensitivity = _compute_sensitivity(trials, self.params)
        return SweepReport(
            sweep_name=self.sweep_name,
            strategy=self.strategy,
            trials=trials,
            best_config=best_config,
            best_metric=best_metric,
            direction=self.direction,
            sensitivity=sensitivity,
        )


__all__ = ["SweepRunner", "SweepReport", "TrialResult"]
