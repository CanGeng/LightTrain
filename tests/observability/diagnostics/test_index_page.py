"""Diagnostics index page (DESIGN §18.8)."""

from __future__ import annotations

from lighttrain.observability.diagnostics.index_page import write_index_page
from tests._diagnostics import expect_exists


def test_empty_run_still_writes(tmp_path):
    out = write_index_page(tmp_path)
    expect_exists(out, tmp_path, what="index page")
    body = out.read_text(encoding="utf-8")
    assert "Run diagnostics" in body
    assert "Frozen step bundles: **0**" in body


def test_lists_present_artifacts(tmp_path):
    (tmp_path / "frozen_steps").mkdir()
    (tmp_path / "frozen_steps" / "step_5_scheduled.zip").write_text("x", encoding="utf-8")
    (tmp_path / "diagnostics").mkdir()
    crash = tmp_path / "diagnostics" / "crash_123"
    crash.mkdir()
    (crash / "traceback.txt").write_text("Traceback ...\nRuntimeError: synthetic\n", encoding="utf-8")
    (tmp_path / "diagnostics" / "loss_attribution_10.json").write_text("{}", encoding="utf-8")
    out = write_index_page(tmp_path)
    body = out.read_text(encoding="utf-8")
    assert "Frozen step bundles: **1**" in body
    assert "Crash bundles: **1**" in body
    assert "Loss attribution dumps: **1**" in body
    assert "Latest crash" in body
