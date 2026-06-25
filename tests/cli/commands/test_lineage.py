"""Tests for ``lighttrain.cli.commands.lineage`` — all reachable branches.

Covers every subcommand exposed under ``lighttrain lineage``:
  tag / untag / invalidate / pin / gc / prune-orphans / graph

Strategy
--------
* ``CliRunner().invoke(app, ["lineage", ...])`` drives every command.
* A real ``LineageStore`` (SQLite, in-memory via tmp_path) replaces monkeypatching
  the store itself, because the store is cheap and tests stay realistic.
* Heavy external operations (``gc_artifacts``, ``prune_orphans``, DAG renderers)
  are tested through the real implementations — they operate entirely on the
  in-process SQLite db, no GPU/network needed.
* Paths that require a missing db file are tested by not creating the db first,
  which exercises the ``_open_lineage`` early-exit (exit code 1).
* ``_resolve_node`` 'no match' paths are exercised by passing a ref that does
  not exist in an otherwise valid (but empty) db.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.observability.lineage.store import LineageStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test."""
    return CliRunner()


def _make_db(tmp_path: Path, name: str = "lineage.sqlite") -> Path:
    """Create a real LineageStore db file and close it; returns the db path."""
    db = tmp_path / name
    s = LineageStore(db)
    s.close()
    return db


def _make_db_with_node(tmp_path: Path) -> tuple[Path, int]:
    """Create a db with a single artifact node; return (db_path, node_id)."""
    db = tmp_path / "lineage.sqlite"
    s = LineageStore(db)
    nid = s.upsert_node(kind="artifact", name="myart", version="v1")
    s.close()
    return db, nid


# ---------------------------------------------------------------------------
# _open_lineage — db not found (line 20-22)
# ---------------------------------------------------------------------------


def test_invariant_open_lineage_missing_db_exits_1_tag(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage tag`` with a non-existent ``--db`` path exits 1 and prints error.

    Covers lines 20-22 (``_open_lineage`` missing-db guard).
    """
    missing = tmp_path / "no.sqlite"
    res = runner.invoke(
        app,
        ["lineage", "tag", "artifact:x:v1", "--tag", "good", "--db", str(missing)],
    )
    assert res.exit_code == 1
    assert "not found" in res.stdout.lower() or "lineage db" in res.stdout.lower()


def test_invariant_open_lineage_missing_db_exits_1_gc(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage gc`` with a non-existent ``--db`` path exits 1.

    Covers lines 20-22 via gc subcommand path.
    """
    missing = tmp_path / "no.sqlite"
    res = runner.invoke(app, ["lineage", "gc", "--db", str(missing)])
    assert res.exit_code == 1


# ---------------------------------------------------------------------------
# _resolve_node — ref not found (lines 28-30)
# ---------------------------------------------------------------------------


def test_invariant_resolve_node_no_match_tag(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage tag`` with a ref that resolves to None exits 1.

    Covers ``_resolve_node`` missing-ref branch (lines 28-30).
    """
    db = _make_db(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "tag", "artifact:ghost:v99", "--tag", "x", "--db", str(db)],
    )
    assert res.exit_code == 1
    assert "no lineage node" in res.stdout.lower() or "matches ref" in res.stdout.lower()


def test_invariant_resolve_node_no_match_untag(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage untag`` with unknown ref exits 1 (covers lines 28-30 via untag)."""
    db = _make_db(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "untag", "artifact:ghost:v99", "--tag", "x", "--db", str(db)],
    )
    assert res.exit_code == 1


def test_invariant_resolve_node_no_match_invalidate(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage invalidate`` with unknown name:version ref exits 1.

    Note: ``#<id>`` refs bypass the existence check in ``resolve_ref`` —
    that path is covered by ``test_pin_current_behavior_hash_ref_no_existence_check``.
    """
    db = _make_db(tmp_path)
    res = runner.invoke(
        app, ["lineage", "invalidate", "artifact:ghost:v99", "--db", str(db)]
    )
    assert res.exit_code == 1


def test_invariant_resolve_node_no_match_pin(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage pin`` with unknown ref exits 1."""
    db = _make_db(tmp_path)
    res = runner.invoke(
        app, ["lineage", "pin", "artifact:ghost:v99", "--db", str(db)]
    )
    assert res.exit_code == 1


def test_invariant_resolve_node_no_match_graph(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph`` with unknown ref exits 1."""
    db = _make_db(tmp_path)
    res = runner.invoke(
        app, ["lineage", "graph", "artifact:ghost:v99", "--db", str(db)]
    )
    assert res.exit_code == 1


# ---------------------------------------------------------------------------
# lineage_tag_cmd success path (lines 39-45)
# ---------------------------------------------------------------------------


def test_invariant_tag_success(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage tag`` on an existing node exits 0 and prints 'tagged'.

    Covers lines 39-43 (store.tag + console.print).
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        [
            "lineage", "tag",
            f"#{nid}",
            "--tag", "best",
            "--db", str(db),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "tagged" in res.stdout.lower()
    assert "best" in res.stdout

    # Verify the tag was actually written to the db.
    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert "best" in (node or {}).get("tags", [])


# ---------------------------------------------------------------------------
# lineage_untag_cmd success path (lines 53-59)
# ---------------------------------------------------------------------------


def test_invariant_untag_success(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage untag`` removes a tag and exits 0.

    Covers lines 53-57.
    """
    db = tmp_path / "lineage.sqlite"
    s = LineageStore(db)
    nid = s.upsert_node(kind="artifact", name="myart", version="v1")
    s.tag(nid, "old-tag")
    s.close()

    res = runner.invoke(
        app,
        [
            "lineage", "untag",
            f"#{nid}",
            "--tag", "old-tag",
            "--db", str(db),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "untagged" in res.stdout.lower()

    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert "old-tag" not in (node or {}).get("tags", [])


# ---------------------------------------------------------------------------
# lineage_invalidate_cmd success path (lines 66-72)
# ---------------------------------------------------------------------------


def test_invariant_invalidate_success(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage invalidate`` marks the node deprecated and exits 0.

    Covers lines 66-70.
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "invalidate", f"#{nid}", "--db", str(db)],
    )
    assert res.exit_code == 0, res.stdout
    assert "invalidated" in res.stdout.lower()

    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert (node or {}).get("deprecated") == 1


# ---------------------------------------------------------------------------
# lineage_pin_cmd success path (lines 79-85)
# ---------------------------------------------------------------------------


def test_invariant_pin_success(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage pin`` sets pinned=1 and exits 0.

    Covers lines 79-83.
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "pin", f"#{nid}", "--db", str(db)],
    )
    assert res.exit_code == 0, res.stdout
    assert "pinned" in res.stdout.lower()

    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert (node or {}).get("pinned") == 1


# ---------------------------------------------------------------------------
# lineage_gc_cmd (lines 94-110)
# ---------------------------------------------------------------------------


def test_invariant_gc_empty_db_exits_0(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage gc`` on an empty db exits 0 and prints 'gc'.

    Covers lines 94-105 (gc_artifacts import + call + console.print).
    """
    db = _make_db(tmp_path)
    res = runner.invoke(app, ["lineage", "gc", "--db", str(db)])
    assert res.exit_code == 0, res.stdout
    assert "gc" in res.stdout.lower()


def test_invariant_gc_dry_run_flag(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage gc --dry-run`` exits 0.

    Covers the ``dry_run=True`` branch.
    """
    db = _make_db(tmp_path)
    res = runner.invoke(app, ["lineage", "gc", "--db", str(db), "--dry-run"])
    assert res.exit_code == 0, res.stdout
    assert "gc" in res.stdout.lower()


def test_invariant_gc_keep_last_and_kind(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage gc --keep-last 1 --kind checkpoint`` exits 0 with a real node.

    Covers the ``keep_last`` + ``kind`` path (lines 98-104).
    """
    db = tmp_path / "lineage.sqlite"
    s = LineageStore(db)
    s.upsert_node(kind="checkpoint", name="ckpt", version="v1")
    s.upsert_node(kind="checkpoint", name="ckpt", version="v2")
    s.close()

    res = runner.invoke(
        app,
        ["lineage", "gc", "--db", str(db), "--keep-last", "1", "--kind", "checkpoint"],
    )
    assert res.exit_code == 0, res.stdout
    assert "gc" in res.stdout.lower()


def test_invariant_gc_reports_counts_in_stdout(runner: CliRunner, tmp_path: Path) -> None:
    """The gc report line includes deprecated/deleted/paths_deleted counts.

    Covers line 105-108 (console.print format string).
    """
    db = _make_db(tmp_path)
    res = runner.invoke(app, ["lineage", "gc", "--db", str(db)])
    assert res.exit_code == 0, res.stdout
    # Expect the format string from line 106-108: deprecated=N deleted=N paths_deleted=N
    assert "deprecated" in res.stdout
    assert "deleted" in res.stdout


# ---------------------------------------------------------------------------
# lineage_prune_cmd (lines 117-124)
# ---------------------------------------------------------------------------


def test_invariant_prune_empty_db_exits_0(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage prune-orphans`` on an empty db exits 0 and prints 'pruned'.

    Covers lines 117-122 (prune_orphans import + call + console.print).
    """
    db = _make_db(tmp_path)
    res = runner.invoke(app, ["lineage", "prune-orphans", "--db", str(db)])
    assert res.exit_code == 0, res.stdout
    assert "pruned" in res.stdout.lower()


def test_invariant_prune_dry_run(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage prune-orphans --dry-run`` exits 0 without deleting anything."""
    db = tmp_path / "lineage.sqlite"
    # Create a node whose payload_path points to a real directory, then remove it.
    payload_dir = tmp_path / "artifact_files"
    payload_dir.mkdir()
    s = LineageStore(db)
    nid = s.upsert_node(
        kind="artifact",
        name="todel",
        version="v1",
        payload_path=str(payload_dir),
    )
    s.close()
    # Remove the directory so the node becomes an orphan.
    payload_dir.rmdir()

    res = runner.invoke(
        app, ["lineage", "prune-orphans", "--db", str(db), "--dry-run"]
    )
    assert res.exit_code == 0, res.stdout
    assert "pruned" in res.stdout.lower()

    # In dry-run mode the node must still exist in the db.
    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert node is not None, "dry-run must not delete the node"


def test_invariant_prune_removes_orphan(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage prune-orphans`` (not dry-run) removes orphan nodes.

    Covers the actual deletion path inside prune_orphans (line 122).
    """
    db = tmp_path / "lineage.sqlite"
    payload_dir = tmp_path / "gone"
    payload_dir.mkdir()
    s = LineageStore(db)
    nid = s.upsert_node(
        kind="artifact", name="orphan", version="v1",
        payload_path=str(payload_dir),
    )
    s.close()
    payload_dir.rmdir()  # make it an orphan

    res = runner.invoke(app, ["lineage", "prune-orphans", "--db", str(db)])
    assert res.exit_code == 0, res.stdout
    assert "1" in res.stdout  # "pruned 1 orphan node(s)"

    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert node is None, "real prune must delete the orphan node"


# ---------------------------------------------------------------------------
# lineage_graph_cmd — mermaid (default) (lines 134-149)
# ---------------------------------------------------------------------------


def test_invariant_graph_mermaid_stdout(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph`` defaults to mermaid and prints to stdout when --out absent.

    Covers lines 134-142 (to_mermaid branch) + line 147 (console.print).
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "graph", f"#{nid}", "--db", str(db)],
    )
    assert res.exit_code == 0, res.stdout
    # to_mermaid starts with "graph TD"
    assert "graph" in res.stdout.lower()


def test_invariant_graph_dot_stdout(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph --fmt dot`` calls to_dot and prints to stdout.

    Covers lines 139-140 (``fmt == "dot"`` branch).
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "graph", f"#{nid}", "--db", str(db), "--fmt", "dot"],
    )
    assert res.exit_code == 0, res.stdout
    # to_dot starts with "digraph lineage {"
    assert "digraph" in res.stdout.lower()


def test_invariant_graph_out_file_written(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph --out <file>`` writes the graph text to the file.

    Covers lines 143-145 (``out is not None`` branch: write + print "wrote …").
    """
    db, nid = _make_db_with_node(tmp_path)
    out_file = tmp_path / "g.mermaid"
    res = runner.invoke(
        app,
        [
            "lineage", "graph", f"#{nid}",
            "--db", str(db),
            "--out", str(out_file),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "wrote" in res.stdout.lower()
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "graph" in content.lower()  # mermaid header


def test_invariant_graph_out_dot_file_written(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph --fmt dot --out <file>`` writes DOT to the output file."""
    db, nid = _make_db_with_node(tmp_path)
    out_file = tmp_path / "g.dot"
    res = runner.invoke(
        app,
        [
            "lineage", "graph", f"#{nid}",
            "--db", str(db),
            "--fmt", "dot",
            "--out", str(out_file),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "digraph" in content.lower()


def test_invariant_graph_depth_option(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage graph --depth 2`` is accepted without error.

    Covers the depth kwarg forwarding path.
    """
    db, nid = _make_db_with_node(tmp_path)
    res = runner.invoke(
        app,
        ["lineage", "graph", f"#{nid}", "--db", str(db), "--depth", "2"],
    )
    assert res.exit_code == 0, res.stdout


# ---------------------------------------------------------------------------
# resolve_ref via name:version notation (covers store.resolve_ref code path
# and makes sure the CLI ref-parsing round-trips correctly)
# ---------------------------------------------------------------------------


def test_invariant_tag_via_name_version_ref(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage tag artifact:myart:v1`` resolves via name:version ref, exits 0."""
    db = tmp_path / "lineage.sqlite"
    s = LineageStore(db)
    s.upsert_node(kind="artifact", name="myart", version="v1")
    s.close()

    res = runner.invoke(
        app,
        [
            "lineage", "tag",
            "artifact:myart:v1",
            "--tag", "released",
            "--db", str(db),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "tagged" in res.stdout.lower()


def test_invariant_tag_via_name_latest_ref(runner: CliRunner, tmp_path: Path) -> None:
    """``lineage tag artifact:myart`` (no version = latest) resolves and exits 0."""
    db = tmp_path / "lineage.sqlite"
    s = LineageStore(db)
    nid = s.upsert_node(kind="artifact", name="myart", version="v1")
    s.close()

    res = runner.invoke(
        app,
        [
            "lineage", "tag",
            "artifact:myart",
            "--tag", "latest-tag",
            "--db", str(db),
        ],
    )
    assert res.exit_code == 0, res.stdout

    s = LineageStore(db)
    node = s.get_node(nid)
    s.close()
    assert "latest-tag" in (node or {}).get("tags", [])


# ---------------------------------------------------------------------------
# Verify that ``store.close()`` is always called (finally clause, lines 44,
# 58, 71, 84, 109, 123, 148) by asserting commands do not leave the SQLite
# connection in a state that blocks a subsequent open in the same test.
# ---------------------------------------------------------------------------


def test_invariant_store_closed_after_tag(runner: CliRunner, tmp_path: Path) -> None:
    """After ``lineage tag``, the SQLite db can be immediately re-opened.

    This pins the ``finally: store.close()`` contract (line 44).
    """
    db, nid = _make_db_with_node(tmp_path)
    runner.invoke(
        app,
        ["lineage", "tag", f"#{nid}", "--tag", "x", "--db", str(db)],
    )
    # If close() was not called this would block (WAL lock) on some platforms.
    s = LineageStore(db)
    s.close()


def test_invariant_store_closed_after_error_path(runner: CliRunner, tmp_path: Path) -> None:
    """After an error exit (unknown ref), the SQLite db is still reopenable.

    This pins the ``finally: store.close()`` on the error path (line 44 via
    the exception from ``raise typer.Exit(code=1)``).
    """
    db = _make_db(tmp_path)
    runner.invoke(
        app,
        ["lineage", "tag", "artifact:ghost:v99", "--tag", "x", "--db", str(db)],
    )
    s = LineageStore(db)
    s.close()


# ---------------------------------------------------------------------------
# Pin current behavior — suspected bug: #<id> ref does not check existence
# ---------------------------------------------------------------------------


def test_pin_current_behavior_hash_ref_no_existence_check(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Pin: ``#<id>`` refs return the integer id without checking that the node
    exists in the db (``resolve_ref`` line 379). This means commands like
    ``invalidate #9999`` exit 0 even when node 9999 doesn't exist, silently
    no-op'ing. The current behavior is pinned here rather than treated as an
    assertion error; the source-fix policy prohibits editing source.

    Suspected bug: the missing existence check means GC/pin/tag on phantom
    ids silently succeed instead of returning exit code 1.
    """
    db = _make_db(tmp_path)
    res = runner.invoke(
        app, ["lineage", "invalidate", "#9999", "--db", str(db)]
    )
    # Current behavior: exits 0 (no existence check for #<id> refs).
    assert res.exit_code == 0, (
        "pin: #<id> refs bypass existence check — if this now fails with "
        "exit 1, the store.resolve_ref behavior has been fixed upstream"
    )
