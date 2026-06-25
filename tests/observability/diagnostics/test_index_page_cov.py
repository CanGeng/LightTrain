"""Edge-case and branch-coverage tests for
``lighttrain.observability.diagnostics.index_page``.

Lines targeted (previously uncovered):
  52-53  bus.quarantined raises → warning logged, quarantined = []
  57     quarantined reset to [] after exception
  61-66  cb_failures_n > 0 triggers write_callback_report (import + call path)
  65-66  write_callback_report import/call raises → warning logged, continues
  101-104 nan_repros present → "Latest NaN repro" section; README read when exists
  106-109 oom present → "Latest OOM report" section; report.md read when exists
  121   _crash_section when traceback.txt absent → short form returned
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from lighttrain.observability.diagnostics.index_page import (
    _crash_section,
    write_index_page,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _RaisingIterable:
    """An iterable that raises when iterated (list() on it raises)."""

    def __iter__(self):
        raise RuntimeError("quarantine exploded")


class _BusWithRaisingQuarantine:
    """Bus whose .quarantined attribute is present (hasattr passes) but raises when iterated.

    Python's hasattr catches only AttributeError; to reach the try/except
    inside write_index_page we need hasattr to return True, then list() to fail.
    """

    quarantined = _RaisingIterable()


class _BusWithQuarantine:
    """Bus with a well-behaved .quarantined list."""

    def __init__(self, names):
        self.quarantined = list(names)


# ---------------------------------------------------------------------------
# _crash_section — line 121 (no traceback.txt → short form)
# ---------------------------------------------------------------------------

def test_invariant_crash_section_without_traceback(tmp_path):
    """_crash_section returns the short `` `name`\\n `` form when traceback.txt is absent."""
    crash_dir = tmp_path / "crash_42"
    crash_dir.mkdir()
    result = _crash_section(crash_dir)
    assert result == f"`{crash_dir.name}`\n"
    assert "```" not in result


def test_invariant_crash_section_with_traceback(tmp_path):
    """_crash_section wraps traceback.txt contents in a code-fence block."""
    crash_dir = tmp_path / "crash_77"
    crash_dir.mkdir()
    (crash_dir / "traceback.txt").write_text(
        "RuntimeError: boom\n  File foo.py, line 1\n", encoding="utf-8"
    )
    result = _crash_section(crash_dir)
    assert "```" in result
    assert "RuntimeError: boom" in result


# ---------------------------------------------------------------------------
# bus.quarantined raises (lines 52-53, 57)
# ---------------------------------------------------------------------------

def test_invariant_bus_quarantine_exception_logs_warning(tmp_path, caplog):
    """When bus.quarantined raises, a WARNING is emitted and quarantined stays []."""
    bus = _BusWithRaisingQuarantine()
    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.index_page"):
        out = write_index_page(tmp_path, bus=bus)
    body = out.read_text(encoding="utf-8")
    # The fallback is '_none_' because quarantined became []
    assert "_none_" in body
    assert "failed to read bus.quarantined" in caplog.text


def test_invariant_bus_quarantine_exception_does_not_raise(tmp_path):
    """write_index_page must not propagate the bus.quarantined exception."""
    bus = _BusWithRaisingQuarantine()
    # Must not raise — returns the output path normally
    result = write_index_page(tmp_path, bus=bus)
    assert result.exists()


def test_invariant_bus_with_quarantined_list(tmp_path):
    """Quarantined callback names appear in the index when bus.quarantined succeeds."""
    bus = _BusWithQuarantine(["MyCallback", "OtherCallback"])
    out = write_index_page(tmp_path, bus=bus)
    body = out.read_text(encoding="utf-8")
    assert "MyCallback" in body
    assert "OtherCallback" in body


def test_invariant_bus_none_skips_quarantine_block(tmp_path):
    """bus=None → quarantine block simply shows _none_."""
    out = write_index_page(tmp_path, bus=None)
    body = out.read_text(encoding="utf-8")
    assert "_none_" in body


# ---------------------------------------------------------------------------
# cb_failures_n > 0 triggers write_callback_report (lines 61-66)
# ---------------------------------------------------------------------------

def _make_cb_failures(run_dir: Path, n: int = 1) -> None:
    """Write n synthetic callback failure JSONL lines."""
    import json

    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / "callback_failures.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "ts": 1.0,
                        "step": i,
                        "callback": "FakeCb",
                        "event": "on_step_end",
                        "exc_type": "ValueError",
                        "traceback": "ValueError: fake\n",
                    }
                )
                + "\n"
            )


def test_invariant_cb_failures_triggers_callback_report(tmp_path):
    """When callback_failures.jsonl is non-empty, callback_report.md is regenerated."""
    _make_cb_failures(tmp_path, n=2)
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Callback failures (isolated): **2**" in body
    # callback_report.md should also be present after regeneration
    assert (tmp_path / "diagnostics" / "callback_report.md").exists()


def test_invariant_cb_failures_zero_no_report_regenerated(tmp_path):
    """When there are no callback failures the report regeneration branch is skipped."""
    # No callback_failures.jsonl at all
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Callback failures (isolated): **0**" in body
    # report should NOT be written (branch not taken)
    assert not (tmp_path / "diagnostics" / "callback_report.md").exists()


def test_pin_current_behavior_cb_report_regen_failure_is_swallowed(tmp_path, caplog):
    """Pin: when write_callback_report raises inside the cb_failures_n > 0 branch,
    the exception is swallowed and a WARNING is emitted, but write_index_page
    still completes and returns the index path.

    This pins the current broad-except swallow on lines 65-69 of index_page.py.
    """
    _make_cb_failures(tmp_path, n=1)
    target = "lighttrain.observability.diagnostics.index_page"
    with caplog.at_level(logging.WARNING, logger=target):
        with patch(
            f"{target}.write_index_page.__module__",  # dummy — we patch the import below
        ):
            pass  # just ensure context is set up cleanly

    # Patch the imported symbol inside the function's closure
    with caplog.at_level(logging.WARNING, logger=target):
        import lighttrain.observability.diagnostics.callback_isolation as _ci_mod

        original = _ci_mod.write_callback_report

        def _raising(*_a, **_kw):
            raise RuntimeError("fake report failure")

        _ci_mod.write_callback_report = _raising
        try:
            result = write_index_page(tmp_path)
        finally:
            _ci_mod.write_callback_report = original

    assert result.exists()
    assert "callback report regeneration failed" in caplog.text


# ---------------------------------------------------------------------------
# nan_repros section (lines 101-104)
# ---------------------------------------------------------------------------

def test_invariant_nan_repro_section_present_without_readme(tmp_path):
    """When nan_repros exist but no README.md the section header still appears."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    repro = diag / "repro_nan_step100"
    repro.mkdir()
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Latest NaN repro" in body
    # Count should be 1
    assert "NaN repros: **1**" in body


def test_invariant_nan_repro_readme_content_included(tmp_path):
    """When README.md exists inside a nan_repro dir its content appears in the index."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    repro = diag / "repro_nan_step200"
    repro.mkdir()
    readme_content = "# NaN repro\n\nSeed: 42\n"
    (repro / "README.md").write_text(readme_content, encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "NaN repro" in body
    assert "Seed: 42" in body


def test_invariant_nan_repros_multiple_uses_last(tmp_path):
    """With multiple nan_repro dirs the *last* (sorted) one is displayed."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    for name in ("repro_nan_step010", "repro_nan_step020", "repro_nan_step030"):
        d = diag / name
        d.mkdir()
        (d / "README.md").write_text(f"# {name}", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    # latest alphabetically = repro_nan_step030
    assert "repro_nan_step030" in body
    assert "NaN repros: **3**" in body


# ---------------------------------------------------------------------------
# oom section (lines 106-109)
# ---------------------------------------------------------------------------

def test_invariant_oom_section_present_without_report_md(tmp_path):
    """When oom dirs exist but no report.md the section header still appears."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    oom = diag / "oom_step55"
    oom.mkdir()
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Latest OOM report" in body
    assert "OOM reports: **1**" in body


def test_invariant_oom_report_md_content_included(tmp_path):
    """When report.md exists inside an oom dir its content appears in the index."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    oom = diag / "oom_step77"
    oom.mkdir()
    report_content = "# OOM\n\nAllocation: 16 GB\n"
    (oom / "report.md").write_text(report_content, encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "OOM" in body
    assert "Allocation: 16 GB" in body


def test_invariant_oom_multiple_uses_last(tmp_path):
    """With multiple oom dirs the *last* (sorted) one is displayed."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    for name in ("oom_step001", "oom_step002", "oom_step003"):
        d = diag / name
        d.mkdir()
        (d / "report.md").write_text(f"# {name}", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "oom_step003" in body
    assert "OOM reports: **3**" in body


# ---------------------------------------------------------------------------
# General / combined edge cases
# ---------------------------------------------------------------------------

def test_invariant_idempotent(tmp_path):
    """Calling write_index_page twice overwrites rather than appending."""
    out1 = write_index_page(tmp_path)
    content1 = out1.read_text(encoding="utf-8")
    out2 = write_index_page(tmp_path)
    content2 = out2.read_text(encoding="utf-8")
    assert out1 == out2
    assert content1 == content2


def test_invariant_lineage_sqlite_section(tmp_path):
    """When lineage.sqlite exists its line appears in the index."""
    (tmp_path / "lineage.sqlite").write_text("dummy", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "lineage.sqlite" in body


def test_invariant_no_lineage_sqlite_no_line(tmp_path):
    """When lineage.sqlite is absent that line does not appear."""
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "lineage.sqlite" not in body


def test_invariant_nan_dumps_counted(tmp_path):
    """NaN dump .pt files under diagnostics/nan_dumps/ are counted."""
    nd = tmp_path / "diagnostics" / "nan_dumps"
    nd.mkdir(parents=True)
    (nd / "layer0.pt").write_text("x", encoding="utf-8")
    (nd / "layer1.pt").write_text("x", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "NaN dumps (module I/O): **2**" in body


def test_invariant_sample_preview_counted(tmp_path):
    """Sample preview .txt files are counted in the index."""
    sp = tmp_path / "diagnostics" / "sample_preview"
    sp.mkdir(parents=True)
    (sp / "step10.txt").write_text("sample", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Sample previews: **1**" in body


def test_invariant_grad_flow_counted(tmp_path):
    """grad_flow_*.json files are counted in the index."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    (diag / "grad_flow_step5.json").write_text("{}", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Grad-flow snapshots: **1**" in body


def test_invariant_dead_neurons_counted(tmp_path):
    """dead_neurons_*.json files are counted in the index."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    (diag / "dead_neurons_step3.json").write_text("{}", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Dead-neuron snapshots: **1**" in body


def test_invariant_crash_section_in_index(tmp_path):
    """Latest crash section appears in the index with its name."""
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    crash = diag / "crash_abc"
    crash.mkdir()
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Latest crash" in body
    assert "crash_abc" in body


def test_invariant_run_dir_name_in_header(tmp_path):
    """The run directory name appears in the index header."""
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert tmp_path.name in body


def test_invariant_last_frozen_displayed(tmp_path):
    """When frozen_steps exist the last bundle name appears in the summary."""
    fs = tmp_path / "frozen_steps"
    fs.mkdir()
    (fs / "step_001.zip").write_text("x", encoding="utf-8")
    (fs / "step_002.zip").write_text("x", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "step_002.zip" in body


def test_invariant_cb_failures_blank_lines_skipped(tmp_path):
    """Blank lines in callback_failures.jsonl are skipped in the count."""
    import json

    diag = tmp_path / "diagnostics"
    diag.mkdir()
    entry = json.dumps(
        {
            "ts": 1.0,
            "step": 0,
            "callback": "X",
            "event": "e",
            "exc_type": "E",
            "traceback": "",
        }
    )
    # Surround with blank lines
    (diag / "callback_failures.jsonl").write_text(
        "\n" + entry + "\n\n", encoding="utf-8"
    )
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Callback failures (isolated): **1**" in body


def test_invariant_all_artifacts_combined(tmp_path):
    """Smoke test: all artifact types together produce a coherent index."""
    # frozen
    fs = tmp_path / "frozen_steps"
    fs.mkdir()
    (fs / "step_010.zip").write_text("x", encoding="utf-8")
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    # crash
    crash = diag / "crash_x"
    crash.mkdir()
    (crash / "traceback.txt").write_text("RuntimeError\n", encoding="utf-8")
    # nan repro
    repro = diag / "repro_nan_step5"
    repro.mkdir()
    # oom
    oom_d = diag / "oom_step3"
    oom_d.mkdir()
    # loss attribution
    (diag / "loss_attribution_1.json").write_text("{}", encoding="utf-8")
    # nan dumps
    nd = diag / "nan_dumps"
    nd.mkdir()
    (nd / "mod.pt").write_text("", encoding="utf-8")
    # sample preview
    sp = diag / "sample_preview"
    sp.mkdir()
    (sp / "step1.txt").write_text("tok", encoding="utf-8")
    # grad flow
    (diag / "grad_flow_1.json").write_text("{}", encoding="utf-8")
    # dead neuron
    (diag / "dead_neurons_1.json").write_text("{}", encoding="utf-8")
    # callback failures
    import json

    (diag / "callback_failures.jsonl").write_text(
        json.dumps(
            {
                "ts": 1.0,
                "step": 1,
                "callback": "Cb",
                "event": "on_step_end",
                "exc_type": "E",
                "traceback": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # lineage
    (tmp_path / "lineage.sqlite").write_text("db", encoding="utf-8")

    bus = _BusWithQuarantine(["BadCallback"])
    out = write_index_page(tmp_path, bus=bus)
    body = out.read_text(encoding="utf-8")

    assert "Frozen step bundles: **1**" in body
    assert "Crash bundles: **1**" in body
    assert "NaN repros: **1**" in body
    assert "OOM reports: **1**" in body
    assert "Loss attribution dumps: **1**" in body
    assert "NaN dumps (module I/O): **1**" in body
    assert "Sample previews: **1**" in body
    assert "Grad-flow snapshots: **1**" in body
    assert "Dead-neuron snapshots: **1**" in body
    assert "Callback failures (isolated): **1**" in body
    assert "BadCallback" in body
    assert "lineage.sqlite" in body
    assert "Latest crash" in body
    assert "Latest NaN repro" in body
    assert "Latest OOM report" in body
