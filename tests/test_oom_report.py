"""OOM report generation (F2 — DESIGN §18.4). CPU-only path."""

from __future__ import annotations

from lighttrain.observability.diagnostics.oom_report import (
    is_oom_exception,
    write_oom_report,
)


def test_is_oom_exception_text_match():
    assert is_oom_exception(RuntimeError("CUDA out of memory.")) is True
    assert is_oom_exception(RuntimeError("normal error")) is False


def test_write_oom_report_cpu_path(tmp_path):
    out = write_oom_report(
        tmp_path,
        exception=RuntimeError("CUDA out of memory."),
        config_path="recipe.yaml",
    )
    assert out.exists()
    # All three artifacts present.
    assert (out / "report.md").exists()
    assert (out / "patch.yaml").exists()
    assert (out / "apply.sh").exists()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "OOM report" in report
    assert "patch.yaml" in report or "apply.sh" in report
    sh = (out / "apply.sh").read_text(encoding="utf-8")
    assert "lighttrain train" in sh and "--apply-degrade" in sh
