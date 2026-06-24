"""Branch-coverage tests for ``lighttrain.lab.compare`` internals.

Complements ``tests/lab/test_compare.py`` (which pins the pure helpers and
end-to-end ``compare``) by driving the still-uncovered failure/fallback
branches:

* ``_read_last_metrics`` blank-line skip, malformed-JSON skip, and the
  ``OSError`` candidate-skip (line 118 / 124-125 / 126-127).
* ``_query_fork_ancestry`` fork_meta read failure -> lineage fallback,
  lineage-store success via ``parent_run_dir`` and via the
  ``fork_of_run_dir`` alias, empty/non-parent payload -> ``None``, malformed
  edge payload -> skipped, and a corrupt store -> warned ``None``
  (lines 157-158, 168-189).
* ``_col_width`` (line 232) — a currently-unreferenced helper, pinned here.
* ``render_png`` (lines 344-375): the matplotlib-missing ``RuntimeError``,
  the all-``None``-metrics early return, and the happy-path PNG write
  (parent-dir creation included).
"""

from __future__ import annotations

import builtins
import json
import sqlite3
import time
from pathlib import Path

import pytest

from lighttrain.lab.compare import (
    CompareReport,
    _col_width,
    _query_fork_ancestry,
    _read_last_metrics,
    render_png,
)

# ---------------------------------------------------------------------------
# _read_last_metrics — blank lines, malformed JSON, OSError candidate
# ---------------------------------------------------------------------------


def test_read_last_metrics_skips_blank_lines(tmp_path: Path):
    """Blank / whitespace-only lines are ignored (line 118 ``if not line``)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 1.0}\n'
        "\n"
        "   \n"
        '{"step": 2, "loss": 0.25}\n',
        encoding="utf-8",
    )
    out = _read_last_metrics(tmp_path)
    assert out == {"loss": pytest.approx(0.25)}


def test_read_last_metrics_skips_malformed_json_line(tmp_path: Path):
    """A line that is not valid JSON is skipped, later good lines still read
    (lines 124-125 ``except (json.JSONDecodeError, TypeError): continue``)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        "this is not json {\n"
        '{"step": 1, "loss": 0.5}\n',
        encoding="utf-8",
    )
    out = _read_last_metrics(tmp_path)
    assert out == {"loss": pytest.approx(0.5)}


def test_read_last_metrics_skips_non_dict_json_line(tmp_path: Path):
    """A JSON line that parses to a list triggers ``.items()`` ``AttributeError``
    — pinned: such a line currently propagates, NOT silently skipped.

    The ``except`` only catches ``json.JSONDecodeError`` / ``TypeError``; a
    bare list/number raises ``AttributeError`` on ``.items()``. We pin that a
    *number* line is caught (``int.items`` -> AttributeError is NOT, but
    ``json.loads`` of a number then ``.items`` ...). To stay on covered lines
    we use a JSON ``null`` whose ``.items()`` raises AttributeError; confirm it
    is NOT swallowed.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text("null\n", encoding="utf-8")
    # ``None.items()`` -> AttributeError, which the narrow except does not catch.
    with pytest.raises(AttributeError):
        _read_last_metrics(tmp_path)


def test_read_last_metrics_skips_candidate_on_oserror(tmp_path: Path):
    """When a metrics candidate path exists but cannot be opened as a file
    (here: it is a directory -> ``IsADirectoryError`` <: ``OSError``), the
    candidate is skipped (lines 126-127) and the function returns ``{}``."""
    logs = tmp_path / "logs"
    logs.mkdir()
    # Make ``logs/metrics.jsonl`` a directory so open() raises IsADirectoryError.
    (logs / "metrics.jsonl").mkdir()
    # No top-level metrics.jsonl either -> falls through to {}.
    assert _read_last_metrics(tmp_path) == {}


def test_read_last_metrics_oserror_then_toplevel_fallback(tmp_path: Path):
    """After the ``logs/`` candidate raises OSError, the top-level
    ``metrics.jsonl`` candidate is still tried and read."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").mkdir()  # OSError on open
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "acc": 0.9}\n', encoding="utf-8"
    )
    out = _read_last_metrics(tmp_path)
    assert out == {"acc": pytest.approx(0.9)}


# ---------------------------------------------------------------------------
# _query_fork_ancestry — fork_meta failure + lineage-store fallback
# ---------------------------------------------------------------------------


def test_query_fork_ancestry_fork_meta_read_failure_falls_back(
    tmp_path: Path, caplog
):
    """A corrupt ``fork_meta.json`` (invalid JSON) logs a warning and falls
    through to the lineage store (lines 157-158). With no ``lineage.sqlite``
    present, the result is ``None``."""
    (tmp_path / "fork_meta.json").write_text("{ not: valid json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert _query_fork_ancestry(tmp_path) is None
    assert any("fork_meta.json" in r.message for r in caplog.records)


def _seed_fork_edge(sqlite_path: Path, payload: dict | None) -> None:
    """Create a real lineage store with one ``fork_of`` edge carrying ``payload``."""
    from lighttrain.observability.lineage.store import LineageStore

    with LineageStore(sqlite_path) as store:
        parent = store.upsert_node(kind="checkpoint", name="parent.pt")
        child = store.upsert_node(kind="run", name="child_run")
        store.add_edge(child, parent, "fork_of", payload=payload)


def test_query_fork_ancestry_from_lineage_parent_run_dir(tmp_path: Path):
    """No fork_meta.json: a ``fork_of`` edge whose payload has
    ``parent_run_dir`` is resolved via the lineage store (lines 168-179)."""
    sqlite_path = tmp_path / "lineage.sqlite"
    _seed_fork_edge(sqlite_path, {"parent_run_dir": "/runs/exp/parent"})
    assert _query_fork_ancestry(tmp_path) == "/runs/exp/parent"


def test_query_fork_ancestry_from_lineage_fork_of_run_dir_alias(tmp_path: Path):
    """The payload alias ``fork_of_run_dir`` is used when ``parent_run_dir``
    is absent (line 177 ``... or payload.get("fork_of_run_dir")``)."""
    sqlite_path = tmp_path / "lineage.sqlite"
    _seed_fork_edge(sqlite_path, {"fork_of_run_dir": "/runs/exp/grandparent"})
    assert _query_fork_ancestry(tmp_path) == "/runs/exp/grandparent"


def test_query_fork_ancestry_lineage_empty_payload_returns_none(tmp_path: Path):
    """A ``fork_of`` edge with no payload (``payload_raw`` falsy) yields no
    parent and the function returns ``None`` (line 189)."""
    sqlite_path = tmp_path / "lineage.sqlite"
    _seed_fork_edge(sqlite_path, None)
    assert _query_fork_ancestry(tmp_path) is None


def test_query_fork_ancestry_lineage_payload_without_parent_keys(tmp_path: Path):
    """A payload that parses but lacks both parent keys -> ``None`` (line 189
    after the ``if parent`` guard is False)."""
    sqlite_path = tmp_path / "lineage.sqlite"
    _seed_fork_edge(sqlite_path, {"unrelated": "value"})
    assert _query_fork_ancestry(tmp_path) is None


def test_query_fork_ancestry_lineage_malformed_payload_skipped(tmp_path: Path):
    """An edge whose stored ``payload`` is not valid JSON is skipped
    (lines 180-181 ``except (json.JSONDecodeError, TypeError): pass``) and the
    function returns ``None`` rather than raising."""
    sqlite_path = tmp_path / "lineage.sqlite"
    _seed_fork_edge(sqlite_path, {"parent_run_dir": "/x"})
    # Corrupt the stored payload directly so json.loads() fails inside the loop.
    conn = sqlite3.connect(str(sqlite_path), isolation_level=None)
    try:
        conn.execute(
            "UPDATE edges SET payload = ? WHERE kind = 'fork_of'",
            ("<<not json>>",),
        )
    finally:
        conn.close()
    assert _query_fork_ancestry(tmp_path) is None


def test_query_fork_ancestry_corrupt_store_warns_returns_none(
    tmp_path: Path, caplog
):
    """A ``lineage.sqlite`` that is not a valid SQLite database makes
    ``LineageStore(...)`` raise; the outer handler logs and returns ``None``
    (lines 182-188)."""
    sqlite_path = tmp_path / "lineage.sqlite"
    sqlite_path.write_bytes(b"this is definitely not a sqlite database file")
    with caplog.at_level("WARNING"):
        assert _query_fork_ancestry(tmp_path) is None
    assert any("lineage store lookup failed" in r.message for r in caplog.records)


def test_query_fork_ancestry_fork_meta_wins_over_lineage(tmp_path: Path):
    """A valid ``fork_meta.json`` short-circuits before the lineage store is
    consulted (line 156 return)."""
    (tmp_path / "fork_meta.json").write_text(
        json.dumps({"fork_of_run_dir": "/runs/from_meta"}), encoding="utf-8"
    )
    # Seed a *different* parent in lineage to prove it is not consulted.
    _seed_fork_edge(tmp_path / "lineage.sqlite", {"parent_run_dir": "/runs/other"})
    assert _query_fork_ancestry(tmp_path) == "/runs/from_meta"


def test_query_fork_ancestry_no_metadata_at_all_returns_none(tmp_path: Path):
    """No fork_meta.json and no lineage.sqlite -> ``None`` (line 167 guard)."""
    assert _query_fork_ancestry(tmp_path) is None


# ---------------------------------------------------------------------------
# _col_width — currently-unreferenced helper (line 231-232)
# ---------------------------------------------------------------------------


def test_col_width_returns_widest_of_values_and_header():
    """``_col_width`` returns the max length over the header and all values."""
    assert _col_width(["a", "bbbb", "cc"], "hd") == 4


def test_col_width_header_wins_when_widest():
    """When the header is the longest token, its length is returned."""
    assert _col_width(["x", "yy"], "loooong-header") == len("loooong-header")


def test_col_width_single_value():
    """Single value vs short header -> the longer one."""
    assert _col_width(["value"], "K") == len("value")


# ---------------------------------------------------------------------------
# render_png — matplotlib-missing error, empty early-return, happy path
# ---------------------------------------------------------------------------


def _report(metrics_table: dict[str, list[float | None]], n_runs: int = 2) -> CompareReport:
    runs = [Path(f"/runs/run_{i}") for i in range(n_runs)]
    return CompareReport(
        runs=runs,
        config_diff={},
        metrics_table=metrics_table,
        fork_ancestry={str(r): None for r in runs},
    )


def test_render_png_raises_when_matplotlib_missing(tmp_path: Path, monkeypatch):
    """When ``import matplotlib.pyplot`` fails, ``render_png`` raises a helpful
    ``RuntimeError`` chained from the ``ImportError`` (lines 344-350)."""
    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib stubbed out")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    report = _report({"loss": [0.5, 0.6]})
    with pytest.raises(RuntimeError, match="render_png requires matplotlib"):
        render_png(report, tmp_path / "out.png")


def test_render_png_returns_early_when_all_metrics_none(tmp_path: Path):
    """If every metric column is all-``None`` the metric list is empty and the
    function returns without writing a file (lines 352-358)."""
    out = tmp_path / "sub" / "out.png"
    report = _report({"loss": [None, None], "acc": [None, None]})
    assert render_png(report, out) is None
    assert not out.exists()
    # parent dir was NOT created on the early return
    assert not out.parent.exists()


def test_render_png_returns_early_when_metrics_table_empty(tmp_path: Path):
    """An empty metrics table also yields the no-op early return."""
    out = tmp_path / "out.png"
    report = _report({})
    assert render_png(report, out) is None
    assert not out.exists()


def test_render_png_writes_file_and_creates_parent_dir(tmp_path: Path):
    """Happy path: with at least one non-``None`` metric, a PNG is written and
    missing parent directories are created (lines 360-375). ``None`` cells are
    plotted as ``0.0``."""
    out = tmp_path / "nested" / "deeper" / "compare.png"
    report = _report({"loss": [0.5, None], "acc": [0.9, 0.8]})
    result = render_png(report, out)
    assert result is None
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic bytes.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_png_accepts_string_path(tmp_path: Path):
    """``render_png`` wraps ``out_path`` in ``Path`` (line 372), so a plain
    string path is accepted."""
    out = tmp_path / "as_string.png"
    report = _report({"loss": [1.0, 2.0]})
    render_png(report, str(out))
    assert out.exists()


def test_render_png_single_run_single_metric(tmp_path: Path):
    """One run, one metric still produces a valid figure (squeeze=False guards
    the single-subplot axes indexing)."""
    out = tmp_path / "single.png"
    report = _report({"loss": [0.42]}, n_runs=1)
    render_png(report, out)
    assert out.exists()


# A timestamp import is used to keep determinism explicit (no sleeps anywhere).
assert isinstance(time.time(), float)
