"""Markdown report generation for sweep and compare results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compare import CompareReport
    from .sweep import SweepReport


# ---------------------------------------------------------------------------
# Sweep report
# ---------------------------------------------------------------------------


def render_sweep_markdown(report: SweepReport, top_k: int = 5) -> str:
    """Return a Markdown string summarising a :class:`~.sweep.SweepReport`."""
    lines: list[str] = []
    lines.append(f"# Sweep report: {report.sweep_name}")
    lines.append("")
    lines.append(f"**Strategy:** {report.strategy}  ")
    lines.append(f"**Metric:** {report.direction} `{_guess_metric_key(report)}`  ")
    n_ok = sum(1 for t in report.trials if t.status == "ok")
    n_pruned = sum(1 for t in report.trials if t.status == "pruned")
    n_failed = sum(1 for t in report.trials if t.status == "failed")
    lines.append(
        f"**Trials:** {len(report.trials)} total"
        f" ({n_ok} ok, {n_pruned} pruned, {n_failed} failed)  "
    )
    if report.best_metric is not None:
        lines.append(f"**Best metric:** `{report.best_metric:.6g}`  ")
    lines.append("")

    # Top-K results table
    ok_trials = sorted(
        [t for t in report.trials if t.status == "ok" and t.metric is not None],
        key=lambda t: t.metric,  # type: ignore[arg-type]
        reverse=(report.direction != "minimize"),
    )
    lines.append(f"## Top-{min(top_k, len(ok_trials))} trials")
    lines.append("")
    if ok_trials:
        param_keys = sorted({k for t in ok_trials for k in t.config_overrides})
        header = "| Rank | Trial | Metric | " + " | ".join(param_keys) + " |"
        sep = "|------|-------|--------|" + "|".join("---" for _ in param_keys) + "|"
        lines.append(header)
        lines.append(sep)
        for rank, t in enumerate(ok_trials[:top_k], 1):
            metric_str = f"{t.metric:.6g}"
            param_vals = " | ".join(
                str(t.config_overrides.get(k, "—")) for k in param_keys
            )
            lines.append(
                f"| {rank} | {t.trial_id} | `{metric_str}` | {param_vals} |"
            )
    else:
        lines.append("_No successful trials._")
    lines.append("")

    # All trials summary
    if len(report.trials) > top_k:
        lines.append("## All trials")
        lines.append("")
        lines.append("| Trial | Status | Metric |")
        lines.append("|-------|--------|--------|")
        for t in report.trials:
            metric_str = f"{t.metric:.6g}" if t.metric is not None else "—"
            lines.append(f"| {t.trial_id} | {t.status} | `{metric_str}` |")
        lines.append("")

    # Sensitivity table
    if report.sensitivity:
        lines.append("## Parameter sensitivity")
        lines.append("")
        lines.append("_Absolute correlation with metric (0 = no impact, 1 = high impact)._")
        lines.append("")
        lines.append("| Parameter | Sensitivity |")
        lines.append("|-----------|-------------|")
        for pname, sens in sorted(
            report.sensitivity.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"| `{pname}` | {sens:.4f} |")
        lines.append("")

    # Best config
    if report.best_config:
        lines.append("## Best configuration overrides")
        lines.append("")
        lines.append("```yaml")
        for k, v in report.best_config.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _guess_metric_key(report: SweepReport) -> str:
    ok = [t for t in report.trials if t.metric is not None]
    if not ok:
        return "metric"
    return "metric"


def write_sweep_report(
    report: SweepReport,
    out_path: Path | None = None,
    top_k: int = 5,
) -> Path:
    """Render and write a sweep report; returns the path written."""
    md = render_sweep_markdown(report, top_k=top_k)
    if out_path is None:
        sweep_run_root = Path("runs") / f"sweep_{report.sweep_name}"
        sweep_run_root.mkdir(parents=True, exist_ok=True)
        out_path = sweep_run_root / "sweep_report.md"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Compare report
# ---------------------------------------------------------------------------


def render_compare_markdown(report: CompareReport) -> str:
    """Return a Markdown string summarising a :class:`~.compare.CompareReport`."""
    lines: list[str] = []
    lines.append("# Compare report")
    lines.append("")
    lines.append(f"**Runs compared:** {len(report.runs)}")
    lines.append("")
    for i, r in enumerate(report.runs):
        parent = report.fork_ancestry.get(str(r))
        ancestry = f" ← fork of `{parent}`" if parent else ""
        lines.append(f"- Run {i}: `{r}`{ancestry}")
    lines.append("")

    # Config diff
    if report.config_diff:
        lines.append("## Configuration differences")
        lines.append("")
        lines.append("_Only fields that differ across runs are shown._")
        lines.append("")
        run_labels = [f"Run {i}" for i in range(len(report.runs))]
        header = "| Key | " + " | ".join(run_labels) + " |"
        sep = "|-----|" + "|".join("---" for _ in run_labels) + "|"
        lines.append(header)
        lines.append(sep)
        for key, vals in sorted(report.config_diff.items()):
            val_strs = " | ".join(f"`{v}`" for v in vals)
            lines.append(f"| `{key}` | {val_strs} |")
        lines.append("")
    else:
        lines.append("## Configuration differences")
        lines.append("")
        lines.append("_All compared runs share identical configurations._")
        lines.append("")

    # Metrics table
    if report.metrics_table:
        lines.append("## Final metrics")
        lines.append("")
        run_labels = [f"Run {i}" for i in range(len(report.runs))]
        header = "| Metric | " + " | ".join(run_labels) + " |"
        sep = "|--------|" + "|".join("---" for _ in run_labels) + "|"
        lines.append(header)
        lines.append(sep)
        for metric, vals in sorted(report.metrics_table.items()):
            val_strs = " | ".join(
                f"`{v:.6g}`" if v is not None else "—" for v in vals
            )
            lines.append(f"| `{metric}` | {val_strs} |")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "render_sweep_markdown",
    "write_sweep_report",
    "render_compare_markdown",
]
