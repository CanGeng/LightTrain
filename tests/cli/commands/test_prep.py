"""Tests for ``lighttrain.cli.commands.prep`` — prep / prep-graph / prep-clean / prep-status.

Coverage targets (currently uncovered lines):
  38-40   prep_cmd error branch  (ConfigError / FileNotFoundError / RuntimeError)
  57-59   prep_graph_cmd error branch
  71-72   prep_graph_cmd --out file-write branch
  85-87   prep_clean_cmd error branch
  90-93   prep_clean_cmd missing-orphans-flag branch (exit 2)
  98-100  prep_clean_cmd removal loop (dry-run prefix and actual-remove prefix)
  112-114 prep_status_cmd error branch
  124-131 prep_status_cmd --extras branches (no-data warning + metrics table)

All tests use ``CliRunner`` exactly like the existing ``tests/cli/test_app.py`` harness.
Heavy work is bypassed by using minimal JSONL sources and real (fast) PrepGraph runs
with a single ``load`` node — no GPU, no network, no distributed launch required.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test."""
    return CliRunner()


def _write_data(tmp_path: Path, name: str = "data.jsonl") -> Path:
    """Write a tiny JSONL file with one record."""
    p = tmp_path / name
    p.write_text('{"text": "hello world"}\n', encoding="utf-8")
    return p


def _write_prep_recipe(tmp_path: Path, data: Path, *, exp: str = "exp1") -> Path:
    """Write a minimal recipe that has a single ``load`` prep_graph node."""
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
mode: lab
seed: 7
run_root: {tmp_path}/runs
exp: {exp}
prep_graph:
  nodes:
    - name: raw
      kind: load
      source: jsonl:{data}
      limit: 1
"""
        ),
        encoding="utf-8",
    )
    return cfg


def _write_no_prepgraph_recipe(tmp_path: Path) -> Path:
    """Write a valid recipe that has NO ``prep_graph:`` section."""
    cfg = tmp_path / "no_pg_recipe.yaml"
    cfg.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    return cfg


# ===========================================================================
# prep_cmd — lines 34-47
# ===========================================================================


def test_invariant_prep_missing_config_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``prep -c <missing>`` must print an error and exit 1.

    Covers lines 38-40 (error branch in prep_cmd).
    """
    missing = tmp_path / "nope.yaml"
    res = runner.invoke(app, ["prep", "-c", str(missing)])
    assert res.exit_code == 1
    assert "prep error:" in res.stdout


def test_invariant_prep_no_prepgraph_section_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``prep -c <recipe-without-prep_graph>`` must exit 1 with a RuntimeError.

    build_prep_runner raises RuntimeError when prep_graph: is absent.
    Covers lines 38-40 (RuntimeError arm).
    """
    cfg = _write_no_prepgraph_recipe(tmp_path)
    res = runner.invoke(app, ["prep", "-c", str(cfg)])
    assert res.exit_code == 1
    assert "prep error:" in res.stdout
    assert "prep_graph" in res.stdout


def test_invariant_prep_dry_run_exits_zero_no_run(runner: CliRunner, tmp_path: Path) -> None:
    """``prep --dry-run`` resolves fingerprints and prints a banner, but exits
    before calling ``runner.run()`` (no cache dirs created).

    Covers the dry-run early-return at line 44-45.
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)
    store_root = tmp_path / "runs" / "exp1" / "prep"

    res = runner.invoke(app, ["prep", "-c", str(cfg), "--dry-run"])

    assert res.exit_code == 0
    # Banner must appear; "prep complete" must NOT appear (no actual run).
    assert "PrepGraph" in res.stdout
    assert "prep complete" not in res.stdout
    # No cache should have been written.
    assert not store_root.exists()


def test_invariant_prep_success_prints_complete(runner: CliRunner, tmp_path: Path) -> None:
    """``prep -c <valid>`` runs the graph and prints "prep complete".

    Covers the full success path (lines 41-47).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep", "-c", str(cfg)])

    assert res.exit_code == 0
    assert "prep complete" in res.stdout


# ===========================================================================
# prep_graph_cmd — lines 50-74
# ===========================================================================


def test_invariant_prep_graph_missing_config_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-graph -c <missing>`` must exit 1 with a named error.

    Covers lines 57-59 (error branch in prep_graph_cmd).
    """
    missing = tmp_path / "ghost.yaml"
    res = runner.invoke(app, ["prep-graph", "-c", str(missing)])
    assert res.exit_code == 1
    assert "prep-graph error:" in res.stdout


def test_invariant_prep_graph_stdout_contains_digraph(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-graph`` without ``--out`` prints the DOT graph to stdout.

    Covers the ``else`` branch at lines 73-74.
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep-graph", "-c", str(cfg)])

    assert res.exit_code == 0
    assert "digraph prepgraph {" in res.stdout
    assert "raw" in res.stdout


def test_invariant_prep_graph_out_writes_dot_file(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-graph --out <file>`` writes the DOT content to disk and prints a
    "wrote" confirmation to stdout.

    Covers lines 71-72 (out-file branch).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)
    out_file = tmp_path / "graph.dot"

    res = runner.invoke(app, ["prep-graph", "-c", str(cfg), "--out", str(out_file)])

    assert res.exit_code == 0
    assert "wrote" in res.stdout.lower()
    assert out_file.exists()
    dot = out_file.read_text(encoding="utf-8")
    assert "digraph prepgraph {" in dot
    assert "raw" in dot


# ===========================================================================
# prep_clean_cmd — lines 77-100
# ===========================================================================


def test_invariant_prep_clean_missing_config_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-clean -c <missing>`` must exit 1 with a named error.

    Covers lines 85-87 (error branch in prep_clean_cmd).
    """
    missing = tmp_path / "ghost.yaml"
    res = runner.invoke(app, ["prep-clean", "-c", str(missing)])
    assert res.exit_code == 1
    assert "prep-clean error:" in res.stdout


def test_invariant_prep_clean_without_orphans_flag_exits_two(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``prep-clean`` without ``--orphans`` must print a hint and exit 2.

    Covers lines 90-93 (missing-orphans-flag branch).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep-clean", "-c", str(cfg)])

    assert res.exit_code == 2
    assert "orphans" in res.stdout.lower()


def test_invariant_prep_clean_orphans_nothing_to_clean(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-clean --orphans`` when no orphans exist prints "nothing to clean".

    Covers the ``if not removed: return`` branch at lines 95-97.
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep-clean", "-c", str(cfg), "--orphans"])

    assert res.exit_code == 0
    assert "nothing to clean" in res.stdout


def test_invariant_prep_clean_orphans_dry_run_shows_would_remove(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``prep-clean --orphans --dry-run`` lists orphaned dirs with "would remove" prefix.

    Covers lines 98-100 (removal loop, dry-run prefix branch).
    Steps:
      1. Run ``prep`` to materialise a cache entry.
      2. Create a second recipe pointing at different data (so original cache
         becomes an orphan under the new fingerprint).
      3. Run ``prep-clean --orphans --dry-run`` against the second recipe.
    """
    data1 = _write_data(tmp_path, "data1.jsonl")
    cfg1 = _write_prep_recipe(tmp_path, data1, exp="exp1")
    # Materialise cache for data1.
    runner.invoke(app, ["prep", "-c", str(cfg1)])

    data2 = _write_data(tmp_path, "data2.jsonl")
    data2.write_text('{"text": "different"}\n', encoding="utf-8")
    cfg2 = _write_prep_recipe(tmp_path, data2, exp="exp1")  # same store root

    res = runner.invoke(app, ["prep-clean", "-c", str(cfg2), "--orphans", "--dry-run"])

    assert res.exit_code == 0
    assert "would remove" in res.stdout.lower()


def test_invariant_prep_clean_orphans_actual_removal(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-clean --orphans`` actually removes orphaned dirs and prints "removed" prefix.

    Covers lines 98-100 (removal loop, actual-remove prefix branch).
    """
    data1 = _write_data(tmp_path, "a.jsonl")
    cfg1 = _write_prep_recipe(tmp_path, data1, exp="exp1")
    runner.invoke(app, ["prep", "-c", str(cfg1)])

    data2 = _write_data(tmp_path, "b.jsonl")
    data2.write_text('{"text": "other"}\n', encoding="utf-8")
    cfg2 = _write_prep_recipe(tmp_path, data2, exp="exp1")

    res = runner.invoke(app, ["prep-clean", "-c", str(cfg2), "--orphans"])

    assert res.exit_code == 0
    assert "removed" in res.stdout.lower()


# ===========================================================================
# prep_status_cmd — lines 103-131
# ===========================================================================


def test_invariant_prep_status_missing_config_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-status -c <missing>`` must exit 1 with a named error.

    Covers lines 112-114 (error branch in prep_status_cmd).
    """
    missing = tmp_path / "ghost.yaml"
    res = runner.invoke(app, ["prep-status", "-c", str(missing)])
    assert res.exit_code == 1
    assert "prep-status error:" in res.stdout


def test_invariant_prep_status_prints_banner(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-status`` without ``--extras`` prints the cache-status banner.

    Covers lines 115-116 (banner call in prep_status_cmd).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep-status", "-c", str(cfg)])

    assert res.exit_code == 0
    assert "PrepGraph" in res.stdout


def test_invariant_prep_status_extras_no_data_warns(runner: CliRunner, tmp_path: Path) -> None:
    """``prep-status --extras`` before any run prints the "no extras on disk" notice.

    Covers lines 124-127 (empty-extras path in prep_status_cmd).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    res = runner.invoke(app, ["prep-status", "-c", str(cfg), "--extras"])

    assert res.exit_code == 0
    assert "no extras on disk" in res.stdout.lower()


def test_invariant_prep_status_extras_after_run_shows_metrics(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``prep-status --extras`` after a full ``prep`` run shows per-node metrics.

    Covers lines 124-131 (extras table render in prep_status_cmd), including
    the ``rendered`` string assembly (line 128-130) and ``console.print`` (131).
    """
    data = _write_data(tmp_path)
    cfg = _write_prep_recipe(tmp_path, data)

    # Materialise the cache so extras exist on disk.
    run_res = runner.invoke(app, ["prep", "-c", str(cfg)])
    assert run_res.exit_code == 0

    res = runner.invoke(app, ["prep-status", "-c", str(cfg), "--extras"])

    assert res.exit_code == 0
    # The LoadNode emits ``row_count`` as an extra metric.
    assert "raw" in res.stdout
    assert "row_count" in res.stdout
