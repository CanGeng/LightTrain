"""Multi-run comparison.

``compare()`` accepts a list of run directories, diffs their resolved
configs, aligns metric histories, and queries their lineage for fork
ancestry.  The result is a :class:`CompareReport` that can be rendered as
ASCII (terminal) or Markdown (via :mod:`~.auto_report`).

Usage::

    from lighttrain.lab.compare import compare, render_ascii

    report = compare([Path("runs/exp/run_001"), Path("runs/exp/run_002")])
    print(render_ascii(report))

PNG export requires ``matplotlib`` (``pip install -e '.[dev]'``)::

    from lighttrain.lab.compare import render_png
    render_png(report, Path("compare.png"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class CompareReport:
    runs: list[Path]
    config_diff: dict[str, list[Any]]       # key → [val_run0, val_run1, …]
    metrics_table: dict[str, list[float | None]]  # metric → [run0, run1, …]
    fork_ancestry: dict[str, str | None]    # str(run_dir) → parent str or None


# ---------------------------------------------------------------------------
# Config loading & diff
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    for name in ("config.resolved.yaml", "config.snapshot.yaml", "config.yaml"):
        p = run_dir / name
        if p.exists():
            try:
                import yaml

                with open(p, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:  # noqa: BLE001
                pass
    return {}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to dot-separated keys."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, full_key))
            else:
                out[full_key] = v
    return out


def _diff_configs(configs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Return only the keys where at least two runs differ."""
    if not configs:
        return {}
    flat = [_flatten(c) for c in configs]
    all_keys: set[str] = set()
    for f in flat:
        all_keys.update(f.keys())

    diff: dict[str, list[Any]] = {}
    for key in sorted(all_keys):
        vals = [f.get(key, None) for f in flat]
        if len(set(repr(v) for v in vals)) > 1:
            diff[key] = vals
    return diff


# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------


def _read_last_metrics(run_dir: Path) -> dict[str, float]:
    """Return the last value of every numeric metric in metrics.jsonl."""
    for candidate in (
        run_dir / "logs" / "metrics.jsonl",
        run_dir / "metrics.jsonl",
    ):
        if not candidate.exists():
            continue
        last: dict[str, float] = {}
        try:
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        for k, v in entry.items():
                            if isinstance(v, (int, float)) and k != "step":
                                last[k] = float(v)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            continue
        return last
    return {}


def _build_metrics_table(
    run_metrics: list[dict[str, float]],
) -> dict[str, list[float | None]]:
    all_keys: set[str] = set()
    for m in run_metrics:
        all_keys.update(m.keys())
    table: dict[str, list[float | None]] = {}
    for key in sorted(all_keys):
        table[key] = [m.get(key) for m in run_metrics]
    return table


# ---------------------------------------------------------------------------
# Lineage / fork ancestry
# ---------------------------------------------------------------------------


def _query_fork_ancestry(run_dir: Path) -> str | None:
    """Return parent run dir string if this run was forked, else None."""
    fork_meta = run_dir / "fork_meta.json"
    if fork_meta.exists():
        try:
            with open(fork_meta, encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("fork_of_run_dir")
        except Exception:  # noqa: BLE001
            pass
    # Also try lineage store
    sqlite_path = run_dir / "lineage.sqlite"
    if not sqlite_path.exists():
        return None
    try:
        from ..lineage.store import LineageStore

        with LineageStore(sqlite_path) as store:
            for edge in store.iter_edges(kind="fork_of"):
                payload_raw = edge.get("payload")
                if payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                        parent = payload.get("parent_run_dir") or payload.get("fork_of_run_dir")
                        if parent:
                            return str(parent)
                    except (json.JSONDecodeError, TypeError):
                        pass
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare(run_paths: list[Path]) -> CompareReport:
    """Produce a :class:`CompareReport` for the given run directories.

    Args:
        run_paths: List of run directories to compare.  Each must contain
            at least one of ``config.resolved.yaml`` / ``config.snapshot.yaml``
            or ``logs/metrics.jsonl``.

    Returns:
        :class:`CompareReport` with config diff, metrics table, and fork ancestry.
    """
    resolved = [Path(p).resolve() for p in run_paths]

    configs = [_load_run_config(r) for r in resolved]
    config_diff = _diff_configs(configs)

    run_metrics = [_read_last_metrics(r) for r in resolved]
    metrics_table = _build_metrics_table(run_metrics)

    fork_ancestry = {str(r): _query_fork_ancestry(r) for r in resolved}

    return CompareReport(
        runs=resolved,
        config_diff=config_diff,
        metrics_table=metrics_table,
        fork_ancestry=fork_ancestry,
    )


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------


def _col_width(values: list[str], header: str) -> int:
    return max(len(header), *(len(v) for v in values))


def render_ascii(report: CompareReport) -> str:
    """Render a :class:`CompareReport` as an ASCII table for terminal output."""
    lines: list[str] = []
    n = len(report.runs)
    run_labels = [f"Run {i}" for i in range(n)]

    lines.append("=== Run summary ===")
    for i, r in enumerate(report.runs):
        parent = report.fork_ancestry.get(str(r))
        suffix = f"  ← fork of {parent}" if parent else ""
        lines.append(f"  Run {i}: {r}{suffix}")
    lines.append("")

    if report.config_diff:
        lines.append("=== Config diff (changed fields only) ===")
        key_w = max(len("Key"), *(len(k) for k in report.config_diff))
        val_w = 18
        header = f"  {'Key':<{key_w}}  " + "  ".join(f"{lbl:^{val_w}}" for lbl in run_labels)
        sep = "  " + "-" * (key_w + 2 + (val_w + 2) * n)
        lines.append(header)
        lines.append(sep)
        for key, vals in sorted(report.config_diff.items()):
            val_strs = [str(v)[:val_w] for v in vals]
            row = f"  {key:<{key_w}}  " + "  ".join(f"{v:^{val_w}}" for v in val_strs)
            lines.append(row)
        lines.append("")
    else:
        lines.append("=== Config diff: no differences ===")
        lines.append("")

    if report.metrics_table:
        lines.append("=== Final metrics ===")
        key_w = max(len("Metric"), *(len(k) for k in report.metrics_table))
        val_w = 12
        header = f"  {'Metric':<{key_w}}  " + "  ".join(
            f"{lbl:^{val_w}}" for lbl in run_labels
        )
        sep = "  " + "-" * (key_w + 2 + (val_w + 2) * n)
        lines.append(header)
        lines.append(sep)
        for metric, vals in sorted(report.metrics_table.items()):
            val_strs = [
                f"{v:.5g}" if v is not None else "—" for v in vals
            ]
            row = f"  {metric:<{key_w}}  " + "  ".join(
                f"{v:^{val_w}}" for v in val_strs
            )
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG renderer (matplotlib, soft dep)
# ---------------------------------------------------------------------------


def render_png(report: CompareReport, out_path: Path) -> None:
    """Write a metrics bar-chart PNG (requires ``matplotlib``)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "render_png requires matplotlib. "
            "Install with: pip install -e '.[dev]'"
        ) from exc

    metrics = [
        k
        for k, vals in report.metrics_table.items()
        if any(v is not None for v in vals)
    ]
    if not metrics:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)
    run_labels = [f"Run {i}" for i in range(len(report.runs))]

    for ax, metric in zip(axes[0], metrics):
        vals = report.metrics_table[metric]
        bar_vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(run_labels, bar_vals)
        ax.set_title(metric)
        ax.set_xlabel("Run")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


__all__ = ["compare", "render_ascii", "render_png", "CompareReport"]
