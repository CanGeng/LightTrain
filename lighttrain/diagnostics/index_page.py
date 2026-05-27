"""Diagnostics index page.

Scans every artifact under ``<run_dir>/`` produced by the diagnostics
callbacks and exception path, then renders a single
``diagnostics/index.md`` summary — the first file a user opens after a
crashed run.

Always safe to call multiple times; idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_index_page(run_dir: str | Path, *, bus: Any | None = None) -> Path:
    """Render ``<run_dir>/diagnostics/index.md``."""
    run_dir = Path(run_dir)
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    frozen = sorted((run_dir / "frozen_steps").glob("*.zip")) if (
        run_dir / "frozen_steps"
    ).exists() else []
    crash = sorted(diag.glob("crash_*")) if diag.exists() else []
    nan_repros = sorted(diag.glob("repro_nan_*")) if diag.exists() else []
    oom = sorted(diag.glob("oom_*")) if diag.exists() else []
    loss_attr = sorted(diag.glob("loss_attribution_*.json")) if diag.exists() else []
    nan_dumps = sorted((diag / "nan_dumps").rglob("*.pt")) if (
        diag / "nan_dumps"
    ).exists() else []
    sample_prev = sorted((diag / "sample_preview").glob("*.txt")) if (
        diag / "sample_preview"
    ).exists() else []
    grad_flow = sorted(diag.glob("grad_flow_*.json"))
    dead_neurons = sorted(diag.glob("dead_neurons_*.json"))
    cb_failures = diag / "callback_failures.jsonl"
    cb_failures_n = (
        sum(1 for _ in cb_failures.read_text(encoding="utf-8").splitlines() if _.strip())
        if cb_failures.exists()
        else 0
    )

    quarantined: list[str] = []
    if bus is not None and hasattr(bus, "quarantined"):
        try:
            quarantined = list(bus.quarantined)
        except Exception:  # noqa: BLE001
            quarantined = []

    # also regenerate callback_report.md if there were failures.
    if cb_failures_n > 0:
        try:
            from .callback_isolation import write_callback_report

            write_callback_report(run_dir, bus=bus)
        except Exception:  # noqa: BLE001
            pass

    last_frozen = frozen[-1].name if frozen else "—"
    lines = [
        f"# Run diagnostics — `{run_dir.name}`",
        "",
        f"- Frozen step bundles: **{len(frozen)}**  (last: `{last_frozen}`)",
        f"- Crash bundles: **{len(crash)}**" + (
            f" → `{crash[-1].relative_to(run_dir)}`" if crash else ""
        ),
        f"- NaN repros: **{len(nan_repros)}**" + (
            f" → `{nan_repros[-1].relative_to(run_dir)}`" if nan_repros else ""
        ),
        f"- OOM reports: **{len(oom)}**" + (
            f" → `{oom[-1].relative_to(run_dir)}`" if oom else ""
        ),
        f"- Loss attribution dumps: **{len(loss_attr)}**",
        f"- NaN dumps (module I/O): **{len(nan_dumps)}**",
        f"- Sample previews: **{len(sample_prev)}**",
        f"- Grad-flow snapshots: **{len(grad_flow)}**",
        f"- Dead-neuron snapshots: **{len(dead_neurons)}**",
        f"- Callback failures (isolated): **{cb_failures_n}**",
        f"- Callbacks currently quarantined: {', '.join(quarantined) or '_none_'}",
        "",
    ]
    if (run_dir / "lineage.sqlite").exists():
        lines.append("- Lineage DB: `lineage.sqlite` present")
        lines.append("")

    if crash:
        lines += ["## Latest crash", "", _crash_section(crash[-1])]
    if nan_repros:
        lines += ["## Latest NaN repro", ""]
        readme = nan_repros[-1] / "README.md"
        if readme.exists():
            lines.append(readme.read_text(encoding="utf-8"))
    if oom:
        lines += ["## Latest OOM report", ""]
        rpt = oom[-1] / "report.md"
        if rpt.exists():
            lines.append(rpt.read_text(encoding="utf-8"))

    out = diag / "index.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _crash_section(crash_dir: Path) -> str:
    tb_path = crash_dir / "traceback.txt"
    if tb_path.exists():
        tb = tb_path.read_text(encoding="utf-8")[:3000]
        return f"`{crash_dir.name}`\n\n```\n{tb.strip()}\n```\n"
    return f"`{crash_dir.name}`\n"


__all__ = ["write_index_page"]
