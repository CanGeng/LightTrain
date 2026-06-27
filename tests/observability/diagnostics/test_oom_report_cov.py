"""Extended coverage for ``lighttrain.observability.diagnostics.oom_report``.

Pins and exercises every branch not yet reached by test_oom_report.py:

* ``_classify_peak`` — all five return values (unknown, kv_cache,
  optimizer_state, fragmentation, activation) [lines 41, 48, 50, 52, 54].
* ``write_oom_report`` — CUDA-stats-read failure path (lines 121-122),
  CPU-only path (line 157), YAML-dump failure / JSON fallback (lines
  188-189, 193).
* ``_fmt_bytes`` — sub-1024 branch (line 211) and all unit labels.
* ``is_oom_exception`` — CUDA isinstance match returning True (lines
  224-225) and isinstance-check exception → warning fallback (lines
  226-227).
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest
import torch

from lighttrain.observability.diagnostics.oom_report import (
    _classify_peak,
    _fmt_bytes,
    is_oom_exception,
    write_oom_report,
)
from tests._diagnostics import expect_exists

# ---------------------------------------------------------------------------
# _classify_peak — all five classification arms
# ---------------------------------------------------------------------------

def test_invariant_classify_peak_empty_stats_returns_unknown():
    """Empty stats dict must return 'unknown' without exceptions (line 41)."""
    assert _classify_peak({}, "") == "unknown"


def test_invariant_classify_peak_kv_keyword_in_summary():
    """'kv' in summary triggers 'kv_cache' regardless of numeric stats (line 48)."""
    stats = {
        "allocated_bytes.all.peak": 1000,
        "reserved_bytes.all.peak": 0,
        "active_bytes.all.peak": 0,
    }
    assert _classify_peak(stats, "has kv store") == "kv_cache"


def test_invariant_classify_peak_decoder_keyword_in_summary():
    """'decoder' in summary also triggers 'kv_cache' (line 48)."""
    stats = {
        "allocated_bytes.all.peak": 1000,
        "reserved_bytes.all.peak": 0,
        "active_bytes.all.peak": 0,
    }
    assert _classify_peak(stats, "Decoder layers dominate") == "kv_cache"


def test_invariant_classify_peak_optimizer_state_heuristic():
    """When active > 0 and active*2 < alloc the heuristic picks
    'optimizer_state' (line 50).

    Setup: alloc=100, active=10 → 10*2=20 < 100, no kv/decoder in summary,
    reserved=0 so fragmentation guard also false.
    """
    stats = {
        "allocated_bytes.all.peak": 100,
        "reserved_bytes.all.peak": 0,
        "active_bytes.all.peak": 10,
    }
    assert _classify_peak(stats, "no special keyword") == "optimizer_state"


def test_invariant_classify_peak_fragmentation_heuristic():
    """reserved > alloc*1.5 triggers 'fragmentation' (line 52).

    Setup: alloc=100, reserved=200, active=0 (so optimizer guard doesn't fire).
    200 > 150 → fragmentation.
    """
    stats = {
        "allocated_bytes.all.peak": 100,
        "reserved_bytes.all.peak": 200,
        "active_bytes.all.peak": 0,
    }
    assert _classify_peak(stats, "") == "fragmentation"


def test_invariant_classify_peak_activation_large_alloc():
    """alloc > 10 GiB triggers 'activation' (line 54)."""
    big = 11 * (1 << 30)  # 11 GiB
    stats = {
        "allocated_bytes.all.peak": big,
        "reserved_bytes.all.peak": 0,
        "active_bytes.all.peak": 0,
    }
    assert _classify_peak(stats, "") == "activation"


def test_invariant_classify_peak_activation_default_fallthrough():
    """When no other heuristic fires the default fallthrough also yields
    'activation' (final return on line 55, the else-less tail).

    Setup: small alloc, reserved slightly above 0 but not > alloc*1.5,
    active == 0.
    """
    stats = {
        "allocated_bytes.all.peak": 50,
        "reserved_bytes.all.peak": 60,   # 60 < 50*1.5=75 → fragmentation guard false
        "active_bytes.all.peak": 0,      # active==0 → optimizer guard false
    }
    result = _classify_peak(stats, "")
    # Both optimizer and fragmentation guards are false; alloc=50 < 10 GiB
    # so the function falls through to the final 'return "activation"'
    assert result == "activation"


# ---------------------------------------------------------------------------
# _fmt_bytes — full unit ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [
    (0,               "0 B"),
    (1,               "1 B"),
    (1023,            "1023 B"),
    (1024,            "1.0 KiB"),
    (2048,            "2.0 KiB"),
    (2 * 1024 ** 2,   "2.0 MiB"),
    (2 * 1024 ** 3,   "2.0 GiB"),
    (2 * 1024 ** 4,   "2.0 TiB"),
])
def test_invariant_fmt_bytes_all_units(n, expected):
    """``_fmt_bytes`` returns the correct human-readable size for each unit tier.

    Line 211 is covered by the sub-1024 cases; TiB covers the ``unit == 'TiB'``
    short-circuit on line 214.
    """
    assert _fmt_bytes(n) == expected


def test_invariant_fmt_bytes_accepts_float_coercion():
    """``_fmt_bytes`` coerces its input via ``int()`` so float inputs work (line 209)."""
    assert _fmt_bytes(1024.9) == "1.0 KiB"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_oom_report — CUDA stats read failure (lines 121-122)
# ---------------------------------------------------------------------------

def test_invariant_write_oom_report_cuda_stats_read_failure(tmp_path, caplog):
    """When CUDA is 'available' but ``torch.cuda.memory_stats()`` raises,
    a WARNING is logged and the report still succeeds (lines 121-122).

    The report picks 'unknown' classification (stats empty after the
    exception).
    """
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_stats", side_effect=RuntimeError("stats broken")):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.oom_report"):
            out = write_oom_report(tmp_path, exception=RuntimeError("CUDA out of memory"))

    expect_exists(out / "report.md", out, what="report.md")
    assert any("reading CUDA memory stats failed" in r.message for r in caplog.records)
    report = (out / "report.md").read_text(encoding="utf-8")
    # Component should fall back to 'unknown' because stats dict is empty
    assert "unknown" in report


# ---------------------------------------------------------------------------
# write_oom_report — CPU-only path (line 157)
# ---------------------------------------------------------------------------

def test_invariant_write_oom_report_cpu_only_path(tmp_path):
    """When ``torch.cuda.is_available()`` returns False, the report contains
    the CPU-environment note (line 157).
    """
    with patch("torch.cuda.is_available", return_value=False):
        out = write_oom_report(tmp_path, exception=RuntimeError("out of memory"))

    expect_exists(out / "report.md", out, what="report.md")
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "CPU-only environment" in report
    assert "no `torch.cuda.memory_stats` available" in report


# ---------------------------------------------------------------------------
# write_oom_report — YAML fallback (lines 188-189, 193)
# ---------------------------------------------------------------------------

def test_invariant_write_oom_report_yaml_dump_failure_falls_back_to_json(tmp_path, caplog):
    """When ``yaml.safe_dump`` raises inside the try block, a WARNING is logged
    and ``patch.yaml`` is written as JSON instead (lines 188-189, 193).
    """
    import yaml

    with patch.object(yaml, "safe_dump", side_effect=RuntimeError("yaml broke")):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.oom_report"):
            out = write_oom_report(tmp_path, exception=RuntimeError("out of memory"))

    expect_exists(out / "patch.yaml", out, what="patch.yaml")
    assert any("YAML dump of patch failed" in r.message for r in caplog.records)
    # Content must be valid JSON
    content = (out / "patch.yaml").read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# is_oom_exception — CUDA isinstance branch
# ---------------------------------------------------------------------------

def test_invariant_is_oom_exception_cuda_isinstance_true():
    """When CUDA is available and exc IS a ``torch.cuda.OutOfMemoryError``
    the function returns True via the isinstance path (lines 224-225).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    oom = torch.cuda.OutOfMemoryError("ran out of memory")
    assert is_oom_exception(oom) is True


def test_invariant_is_oom_exception_cuda_isinstance_not_oom_falls_through():
    """When CUDA is available but exc is NOT an OOM exception, the function
    falls through to the string-match path and returns False for a generic
    error (lines 224-225 branch not taken → line 231-232).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    assert is_oom_exception(RuntimeError("normal runtime error")) is False


def test_invariant_is_oom_exception_isinstance_check_raises_falls_back_to_string(caplog):
    """If the isinstance check itself throws (e.g. OutOfMemoryError patched to
    a non-type), a WARNING is logged and the function falls back to
    string-matching (lines 226-227).
    """
    with patch.object(torch.cuda, "OutOfMemoryError", new=42), \
         patch("torch.cuda.is_available", return_value=True):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.oom_report"):
            # "out of memory" in msg → True via string path
            result_true = is_oom_exception(RuntimeError("out of memory"))
            # no OOM keyword → False via string path
            result_false = is_oom_exception(RuntimeError("some other error"))

    assert result_true is True
    assert result_false is False
    assert any("OutOfMemoryError isinstance check failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# is_oom_exception — string-match variants (CPU / no-CUDA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg, expected", [
    ("out of memory here",         True),
    ("CUDA out of memory.",        True),
    ("CUDA Out Of Memory exactly", True),
    ("normal runtime error",       False),
    ("",                           False),
])
def test_invariant_is_oom_exception_string_matching(msg, expected):
    """String-matching logic is case-insensitive and catches both OOM phrases."""
    with patch("torch.cuda.is_available", return_value=False):
        assert is_oom_exception(RuntimeError(msg)) is expected


# ---------------------------------------------------------------------------
# write_oom_report — all five components produce complete artifact sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component_setup", [
    # (summary, stats overrides) tuples that steer _classify_peak
    ("kv decoder attention", {}),                              # kv_cache
    ("",          {"allocated_bytes.all.peak": 100,
                   "active_bytes.all.peak": 10}),             # optimizer_state
    ("",          {"allocated_bytes.all.peak": 100,
                   "reserved_bytes.all.peak": 200}),          # fragmentation
    ("",          {"allocated_bytes.all.peak": 11*(1<<30)}),  # activation
])
def test_invariant_write_oom_report_produces_three_artifacts(tmp_path, component_setup):
    """Each component classification produces report.md, patch.yaml, apply.sh."""
    summary_text, extra_stats = component_setup
    base_stats: dict = {
        "allocated_bytes.all.peak": 0,
        "reserved_bytes.all.peak": 0,
        "active_bytes.all.peak": 0,
    }
    base_stats.update(extra_stats)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_stats", return_value=base_stats), \
         patch("torch.cuda.memory_summary", return_value=summary_text):
        out = write_oom_report(tmp_path / component_setup[0][:8].strip().replace(" ", "_"),
                               exception=RuntimeError("oom"))

    for name in ("report.md", "patch.yaml", "apply.sh"):
        expect_exists(out / name, out, what=name)


# ---------------------------------------------------------------------------
# write_oom_report — config_path plumbing
# ---------------------------------------------------------------------------

def test_invariant_write_oom_report_config_path_in_apply_sh(tmp_path):
    """The supplied config_path appears in apply.sh (line 200)."""
    with patch("torch.cuda.is_available", return_value=False):
        out = write_oom_report(tmp_path, config_path="/recipes/my.yaml")
    sh = (out / "apply.sh").read_text(encoding="utf-8")
    assert "/recipes/my.yaml" in sh


def test_invariant_write_oom_report_no_config_path_placeholder(tmp_path):
    """When config_path is None the placeholder '<your-recipe.yaml>' appears."""
    with patch("torch.cuda.is_available", return_value=False):
        out = write_oom_report(tmp_path)
    sh = (out / "apply.sh").read_text(encoding="utf-8")
    assert "<your-recipe.yaml>" in sh


# ---------------------------------------------------------------------------
# write_oom_report — return value is a Path that exists
# ---------------------------------------------------------------------------

def test_invariant_write_oom_report_returns_existing_path(tmp_path):
    """``write_oom_report`` always returns a Path that exists."""
    with patch("torch.cuda.is_available", return_value=False):
        out = write_oom_report(tmp_path)
    assert isinstance(out, __import__("pathlib").Path)
    assert out.exists()
