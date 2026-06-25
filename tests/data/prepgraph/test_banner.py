"""Tests for ``lighttrain.data.prepgraph._banner``.

Coverage targets (uncovered lines at time of authoring):

* 31–35  — ``format_plan``: list conversion, n_total / n_cached / n_run counts,
            and construction of the header lines.
* 39–43  — ``format_plan`` for-loop: tag / eta / reason per entry.
* 46–47  — ``format_plan`` closing separator + return.
* 53–54  — ``print_plan``: console=None fallback (calls ``print(format_plan(...))``,
            returns early without Rich).

General edge cases covered:
* Empty plan (zero nodes).
* All-cached plan (n_run == 0).
* All-to-run plan (n_cached == 0).
* Mixed hit/miss plan.
* ETA present vs. absent.
* Various ``reason`` values (cache_hit, config_changed, first_run, …).
* ``print_plan`` with real Rich Console (covers the table-construction path).
* ``print_plan`` with ``console=None`` falls back to ``print``.
"""

from __future__ import annotations

import io

import pytest

from lighttrain.data.prepgraph._banner import PlanEntry, format_plan, print_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    *,
    name: str = "node",
    kind: str = "tokenise",
    fp: str = "abcd1234efgh5678",
    hit: bool = False,
    reason: str = "first_run",
    eta_s: float | None = None,
    rows: int | None = None,
) -> PlanEntry:
    return PlanEntry(
        name=name,
        kind=kind,
        fingerprint=fp[:16],
        full_fp=fp,
        hit=hit,
        reason=reason,
        eta_s=eta_s,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# PlanEntry dataclass
# ---------------------------------------------------------------------------

def test_invariant_plan_entry_fields_stored_verbatim():
    """All PlanEntry fields are stored exactly as supplied."""
    e = _entry(
        name="tok",
        kind="tokenise",
        fp="0102030405060708",
        hit=True,
        reason="cache_hit",
        eta_s=3.7,
        rows=1024,
    )
    assert e.name == "tok"
    assert e.kind == "tokenise"
    assert e.fingerprint == "0102030405060708"
    assert e.hit is True
    assert e.reason == "cache_hit"
    assert e.eta_s == pytest.approx(3.7)
    assert e.rows == 1024


def test_invariant_plan_entry_optional_defaults_to_none():
    """eta_s and rows default to None when not supplied."""
    e = _entry()
    assert e.eta_s is None
    assert e.rows is None


# ---------------------------------------------------------------------------
# format_plan — header (lines 31-37)
# ---------------------------------------------------------------------------

def test_invariant_format_plan_empty_plan_header():
    """Empty plan produces a header with 0 nodes, 0 cached, 0 to run."""
    out = format_plan([])
    lines = out.splitlines()
    assert lines[0] == "PrepGraph: 0 nodes, 0 cached, 0 to run"


def test_invariant_format_plan_separators_present():
    """format_plan output begins and ends with a separator line of dashes."""
    out = format_plan([])
    lines = out.splitlines()
    sep = "─" * 65
    assert lines[1] == sep
    assert lines[-1] == sep


def test_invariant_format_plan_counts_all_cached():
    """All-cache-hit plan: n_run == 0 is reflected in the header."""
    plan = [_entry(name=f"n{i}", hit=True, reason="cache_hit") for i in range(3)]
    out = format_plan(plan)
    assert out.splitlines()[0] == "PrepGraph: 3 nodes, 3 cached, 0 to run"


def test_invariant_format_plan_counts_all_run():
    """All-miss plan: n_cached == 0 is reflected in the header."""
    plan = [_entry(name=f"n{i}", hit=False, reason="first_run") for i in range(2)]
    out = format_plan(plan)
    assert out.splitlines()[0] == "PrepGraph: 2 nodes, 0 cached, 2 to run"


def test_invariant_format_plan_mixed_counts():
    """Mixed plan (2 hit, 1 miss) gives correct header counts."""
    plan = [
        _entry(name="a", hit=True, reason="cache_hit"),
        _entry(name="b", hit=True, reason="cache_hit"),
        _entry(name="c", hit=False, reason="config_changed"),
    ]
    out = format_plan(plan)
    assert out.splitlines()[0] == "PrepGraph: 3 nodes, 2 cached, 1 to run"


# ---------------------------------------------------------------------------
# format_plan — per-row content (lines 39-44)
# ---------------------------------------------------------------------------

def test_invariant_format_plan_cache_hit_row_tag():
    """A hit entry is tagged [CACHE] and annotated with '(hit)'."""
    plan = [_entry(name="tok", hit=True, reason="cache_hit")]
    out = format_plan(plan)
    body = out.splitlines()[2]           # header, sep, body-row
    assert "[CACHE]" in body
    assert "(hit)" in body
    assert "[ RUN ]" not in body


def test_invariant_format_plan_cache_miss_row_tag():
    """A miss entry is tagged [ RUN ] and includes the reason."""
    plan = [_entry(name="tok", hit=False, reason="config_changed")]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "[ RUN ]" in body
    assert "reason: config_changed" in body
    assert "[CACHE]" not in body


def test_invariant_format_plan_eta_present():
    """When eta_s is set, 'ETA: Xs' (integer-rounded) appears in the row."""
    plan = [_entry(name="tok", hit=False, reason="first_run", eta_s=42.9)]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "ETA: 43s" in body


def test_invariant_format_plan_eta_absent():
    """When eta_s is None, 'ETA: ?' appears in the row."""
    plan = [_entry(name="tok", hit=False, reason="first_run", eta_s=None)]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "ETA: ?" in body


def test_invariant_format_plan_eta_zero():
    """eta_s=0.0 is formatted as 'ETA: 0s'."""
    plan = [_entry(name="tok", hit=False, reason="first_run", eta_s=0.0)]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "ETA: 0s" in body


def test_invariant_format_plan_fingerprint_in_row():
    """The fingerprint value appears verbatim in the output row."""
    plan = [_entry(name="tok", fp="deadbeef00112233", hit=False, reason="first_run")]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "deadbeef00112233" in body


def test_invariant_format_plan_name_and_kind_in_row():
    """The name and kind of a node appear in its formatted row."""
    plan = [_entry(name="my_tokeniser", kind="tokenise", hit=False, reason="first_run")]
    out = format_plan(plan)
    body = out.splitlines()[2]
    assert "my_tokeniser" in body
    assert "tokenise" in body


@pytest.mark.parametrize("reason", [
    "cache_hit",
    "config_changed",
    "code_version_changed",
    "upstream_changed",
    "schema_version_bumped",
    "first_run",
])
def test_invariant_format_plan_reason_values(reason: str):
    """All documented reason codes appear verbatim in a miss row."""
    plan = [_entry(name="n", hit=False, reason=reason)]
    out = format_plan(plan)
    body = out.splitlines()[2]
    # hit entries show '(hit)', miss entries show the reason
    assert reason in body


# ---------------------------------------------------------------------------
# format_plan — accepts generator (line 31: list(plan))
# ---------------------------------------------------------------------------

def test_invariant_format_plan_accepts_generator():
    """format_plan consumes a lazy generator, not just a list."""
    def _gen():
        yield _entry(name="a", hit=True, reason="cache_hit")
        yield _entry(name="b", hit=False, reason="first_run")

    out = format_plan(_gen())
    assert "PrepGraph: 2 nodes, 1 cached, 1 to run" in out


# ---------------------------------------------------------------------------
# format_plan — multi-node output structure
# ---------------------------------------------------------------------------

def test_invariant_format_plan_line_count_matches_entries():
    """Total lines == 2 separators + header + 1 line per entry."""
    n = 5
    plan = [_entry(name=f"n{i}", hit=(i % 2 == 0)) for i in range(n)]
    out = format_plan(plan)
    lines = out.splitlines()
    # header + sep + n body rows + sep = n + 3
    assert len(lines) == n + 3


# ---------------------------------------------------------------------------
# print_plan — console=None falls back to print (lines 52-54)
# ---------------------------------------------------------------------------

def test_invariant_print_plan_none_console_calls_print(capsys):
    """print_plan(None, plan) prints the format_plan text to stdout."""
    plan = [_entry(name="tok", hit=False, reason="first_run")]
    print_plan(None, plan)
    captured = capsys.readouterr()
    assert "PrepGraph: 1 nodes" in captured.out
    assert "[ RUN ]" in captured.out


def test_invariant_print_plan_none_console_empty_plan(capsys):
    """print_plan(None, []) prints the empty-plan format_plan header."""
    print_plan(None, [])
    captured = capsys.readouterr()
    assert "PrepGraph: 0 nodes, 0 cached, 0 to run" in captured.out


def test_invariant_print_plan_none_console_returns_none(capsys):
    """print_plan(None, ...) returns None (early return, line 54)."""
    result = print_plan(None, [])
    assert result is None


# ---------------------------------------------------------------------------
# print_plan — with rich Console (lines 55-73)
# ---------------------------------------------------------------------------

def test_invariant_print_plan_with_rich_console_renders_table():
    """print_plan with a real Rich Console renders a table without error."""
    try:
        from rich.console import Console
    except ImportError:
        pytest.skip("rich not installed")

    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False)
    plan = [
        _entry(name="tok", kind="tokenise", hit=True, reason="cache_hit", eta_s=1.0),
        _entry(name="pack", kind="pack", hit=False, reason="config_changed", eta_s=None),
    ]
    print_plan(console, plan)
    rendered = buf.getvalue()
    # Table header / title should appear
    assert "PrepGraph" in rendered
    assert "2 nodes" in rendered


def test_invariant_print_plan_rich_table_hit_and_miss():
    """Rich table includes both cached and run entries."""
    try:
        from rich.console import Console
    except ImportError:
        pytest.skip("rich not installed")

    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False)
    plan = [
        _entry(name="a", hit=True, reason="cache_hit", eta_s=5.0),
        _entry(name="b", hit=False, reason="upstream_changed"),
    ]
    print_plan(console, plan)
    rendered = buf.getvalue()
    assert "a" in rendered
    assert "b" in rendered


def test_invariant_print_plan_rich_eta_question_mark():
    """Rich table shows '?' when eta_s is None."""
    try:
        from rich.console import Console
    except ImportError:
        pytest.skip("rich not installed")

    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False)
    plan = [_entry(name="tok", hit=False, reason="first_run", eta_s=None)]
    print_plan(console, plan)
    rendered = buf.getvalue()
    assert "?" in rendered


def test_invariant_print_plan_rich_eta_numeric():
    """Rich table formats eta_s as integer seconds when provided."""
    try:
        from rich.console import Console
    except ImportError:
        pytest.skip("rich not installed")

    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False)
    plan = [_entry(name="tok", hit=False, reason="first_run", eta_s=7.6)]
    print_plan(console, plan)
    rendered = buf.getvalue()
    assert "8s" in rendered


def test_invariant_print_plan_rich_empty_plan():
    """print_plan with Rich Console handles an empty plan without error."""
    try:
        from rich.console import Console
    except ImportError:
        pytest.skip("rich not installed")

    buf = io.StringIO()
    console = Console(file=buf, width=120, highlight=False)
    print_plan(console, [])
    rendered = buf.getvalue()
    assert "0 nodes" in rendered
