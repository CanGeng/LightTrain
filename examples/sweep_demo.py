"""Hyperparameter sweep — programmatic API example (R15).

Equivalent CLI:
    lighttrain sweep -c recipes/sweep_demo.yaml -s recipes/sweep_r15.yaml
"""

from pathlib import Path

from lighttrain.lab.auto_report import write_sweep_report
from lighttrain.lab.sweep import SweepRunner

runner = SweepRunner(
    Path("recipes/sweep_demo.yaml"),
    Path("recipes/sweep_r15.yaml"),
    strategy="grid",
)
report = runner.run()

print(f"Best metric: {report.best_metric:.4f}")
print(f"Best config: {report.best_config}")
print(f"Sensitivity: {report.sensitivity}")

path = write_sweep_report(report, top_k=5)
print(f"Report written → {path}")
