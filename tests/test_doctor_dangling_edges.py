"""Doctor must surface dangling edges in both directions, and FK cascade
must actually fire on ``delete_node``.

Pre-fix:
* ``PRAGMA foreign_keys=ON`` was missing → ``ON DELETE CASCADE`` never fired,
  so deleting a node left orphan rows in ``edges``.
* Doctor iterated ``edges_from(existing_nodes)`` only → edges whose ``src``
  is gone were invisible.
"""

from __future__ import annotations

from typer.testing import CliRunner

from lighttrain.checkpoint.manager import CheckpointManager
from lighttrain.cli._app import app
from lighttrain.lineage.store import LineageStore


def test_fk_cascade_actually_fires_on_delete_node(tmp_path):
    """Deleting a node now removes any edges that reference it."""
    ls = LineageStore(tmp_path / "lineage.sqlite")
    a = ls.upsert_node(kind="run", name="ra", version="ra")
    b = ls.upsert_node(kind="artifact", name="art", version="v1")
    ls.add_edge(a, b, "produced_by")
    assert len(list(ls.iter_edges())) == 1

    ls.delete_node(a)
    # CASCADE removes the edge automatically.
    assert list(ls.iter_edges()) == []


def test_iter_edges_returns_every_row(tmp_path):
    ls = LineageStore(tmp_path / "lineage.sqlite")
    a = ls.upsert_node(kind="run", name="ra", version="ra")
    b = ls.upsert_node(kind="artifact", name="art1", version="v1")
    c = ls.upsert_node(kind="artifact", name="art2", version="v1")
    ls.add_edge(a, b, "produced_by")
    ls.add_edge(a, c, "produced_by")
    assert len(list(ls.iter_edges())) == 2
    assert len(list(ls.iter_edges(kind="produced_by"))) == 2


def test_doctor_flags_orphan_edges_in_both_directions(tmp_path):
    """Doctor must report an edge as dangling even when its ``src`` node is
    the deleted one. We bypass FK cascade by writing the edge with FK off,
    simulating data from before the post-review fix landed.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    db = run / "lineage.sqlite"
    # Construct nodes via the official API, then manually insert an orphan
    # edge whose src no longer exists.
    ls = LineageStore(db)
    art = ls.upsert_node(kind="artifact", name="art", version="v1",
                        schema_kind="artifact_header", schema_version="0.4")
    # Add an edge from a non-existent node (id=99999) to a real one.
    ls.conn.execute("PRAGMA foreign_keys=OFF;")
    ls.conn.execute(
        "INSERT INTO edges (src, dst, kind, ts) VALUES (?, ?, ?, 0)",
        (99999, art, "produced_by"),
    )
    ls.conn.execute("PRAGMA foreign_keys=ON;")
    ls.close()

    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    # Doctor must detect the orphan and exit non-zero.
    assert res.exit_code == 2, res.stdout
    assert "dangling" in res.stdout.lower() or "orphan" in res.stdout.lower()


def test_doctor_clean_db_still_passes(tmp_path):
    """Regression guard — doctor must not falsely flag a clean DB."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    CheckpointManager(run).save(step=1, state={"trainer": {"step": 1}}, kind="step")
    ls = LineageStore(run / "lineage.sqlite")
    a = ls.upsert_node(kind="run", name="r1", version="r1",
                       schema_kind="run_meta", schema_version="0.4")
    b = ls.upsert_node(kind="artifact", name="a1", version="v1",
                       schema_kind="artifact_header", schema_version="0.4")
    ls.add_edge(a, b, "produced_by")
    ls.close()

    res = CliRunner().invoke(app, ["doctor", "--run", str(run)])
    assert res.exit_code == 0, res.stdout
    assert "no orphans" in res.stdout
