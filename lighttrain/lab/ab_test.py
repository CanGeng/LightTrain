"""A/B testing with identical RNG seeds.

Runs two config variants under the same RNG seeds so any metric difference
is attributable to the config change rather than stochastic variance.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .compare import CompareReport, compare
from .sweep import _find_run_dir, _read_final_metric


@dataclass
class ABReport:
    run_a: Path | None
    run_b: Path | None
    metric_a: float | None
    metric_b: float | None
    delta: float | None        # B - A  (positive = B better when maximising)
    compare: CompareReport | None


def ab_test(
    config_a: Path,
    config_b: Path,
    *,
    seed: int = 42,
    metric_key: str = "loss",
    run_root: str = "runs",
    trial_timeout_s: float | None = None,
) -> ABReport:
    """Run two recipe variants with the same seed and compare results.

    Args:
        config_a: Path to recipe YAML for variant A.
        config_b: Path to recipe YAML for variant B.
        seed: RNG seed to inject into both runs (``++seed=<seed>``).
        metric_key: Metric to extract from ``logs/metrics.jsonl``.
        run_root: Where to store both runs.
        trial_timeout_s: Per-run wall-clock cap in seconds.

    Returns:
        :class:`ABReport` with run dirs, metrics, delta, and full compare.
    """
    run_root_path = Path(run_root).resolve()

    def _launch(cfg_path: Path, variant: str) -> tuple[Path | None, float | None]:
        exp = f"ab_{variant}_{cfg_path.stem}"
        cmd = [
            sys.executable, "-m", "lighttrain", "train",
            "-c", str(cfg_path),
            f"++seed={seed}",
            f"++exp={exp}",
            f"++run_root={run_root_path}",
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=trial_timeout_s,
            )
        except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
            pass
        rd = _find_run_dir(run_root_path / exp)
        metric = _read_final_metric(rd, metric_key) if rd else None
        return rd, metric

    run_a_dir, metric_a = _launch(Path(config_a), "a")
    run_b_dir, metric_b = _launch(Path(config_b), "b")

    delta: float | None = None
    if metric_a is not None and metric_b is not None:
        delta = metric_b - metric_a

    compare_report: CompareReport | None = None
    if run_a_dir is not None and run_b_dir is not None:
        try:
            compare_report = compare([run_a_dir, run_b_dir])
        except Exception:  # noqa: BLE001
            pass

    return ABReport(
        run_a=run_a_dir,
        run_b=run_b_dir,
        metric_a=metric_a,
        metric_b=metric_b,
        delta=delta,
        compare=compare_report,
    )


__all__ = ["ab_test", "ABReport"]
