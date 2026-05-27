"""LineageStore — SQLite persistence for lineage nodes / edges.

A single ``.sqlite`` file per run lives at ``<run_dir>/lineage.sqlite``. Four
node kinds (``artifact`` / ``checkpoint`` / ``config`` / ``run`` /
``frozen_step``) and five edge kinds (``produced_by`` / ``derived_from`` /
``migrated_from`` / ``fork_of`` / ``evaluated_by``) form the lineage graph.
Schema is intentionally narrow — adding a column requires bumping
``SCHEMA_VERSION["run_meta"]`` in ``prepgraph/_fp.py``.

Lineage is a **soft dependency**: deleting the SQLite file does not lose
training data, only the ability to traverse history quickly.

Notes:
* Global aggregated DB at ``~/.lighttrain/lineage.global.sqlite`` is not yet
  implemented. The class accepts an optional ``global_path`` so ``compare``
  / ``fork`` can enable it without restructuring.
* ``journal_mode=WAL`` is enabled; cross-process locking for sweep scenarios
  is untested.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


_NODE_KINDS = ("artifact", "checkpoint", "config", "run", "frozen_step")
_EDGE_KINDS = (
    "produced_by",
    "derived_from",
    "migrated_from",
    "fork_of",
    "evaluated_by",
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    name            TEXT,
    version         TEXT,
    hash            TEXT,
    ts              REAL NOT NULL,
    run_id          TEXT,
    step            INTEGER,
    schema_kind     TEXT,
    schema_version  TEXT,
    payload_path    TEXT,
    payload         TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    pinned          INTEGER NOT NULL DEFAULT 0,
    deprecated      INTEGER NOT NULL DEFAULT 0,
    deprecated_ts   REAL,
    UNIQUE(kind, name, version)
);
CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id, step);
CREATE INDEX IF NOT EXISTS idx_nodes_kind_name ON nodes(kind, name);

CREATE TABLE IF NOT EXISTS edges (
    src     INTEGER NOT NULL,
    dst     INTEGER NOT NULL,
    kind    TEXT NOT NULL,
    payload TEXT,
    ts      REAL NOT NULL,
    PRIMARY KEY (src, dst, kind),
    FOREIGN KEY (src) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (dst) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, kind);
"""


class LineageStore:
    """SQLite-backed lineage store.

    Instances are not thread-safe — callers wrap mutations in :meth:`transaction`
    if they need atomicity across many ``upsert_node`` / ``add_edge`` calls.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        # WAL for concurrent readers; foreign_keys ON so the ``ON DELETE CASCADE``
        # declared on ``edges`` actually fires when ``delete_node()`` is called.
        # Without this PRAGMA SQLite silently leaves orphan edges and doctor
        # cannot see them (post-review fix).
        self.conn.executescript("PRAGMA journal_mode=WAL;")
        self.conn.executescript("PRAGMA foreign_keys=ON;")
        self.conn.executescript(_SCHEMA)

    # ---------------------------------------------------------------- nodes

    def upsert_node(
        self,
        *,
        kind: str,
        name: str | None = None,
        version: str | None = None,
        hash_: str | None = None,
        run_id: str | None = None,
        step: int | None = None,
        schema_kind: str | None = None,
        schema_version: str | None = None,
        payload_path: str | None = None,
        payload: Mapping[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        if kind not in _NODE_KINDS:
            raise ValueError(
                f"unknown lineage node kind {kind!r}; expected one of {_NODE_KINDS}"
            )
        existing = self.find(kind, name, version)
        if existing is not None:
            updates = {
                "hash": hash_,
                "run_id": run_id,
                "step": step,
                "schema_kind": schema_kind,
                "schema_version": schema_version,
                "payload_path": payload_path,
                "payload": json.dumps(dict(payload)) if payload is not None else None,
            }
            sets = ", ".join(f"{k} = COALESCE(?, {k})" for k in updates)
            self.conn.execute(
                f"UPDATE nodes SET {sets} WHERE id = ?",
                (*updates.values(), existing),
            )
            return existing
        cur = self.conn.execute(
            """
            INSERT INTO nodes (kind, name, version, hash, ts, run_id, step,
                               schema_kind, schema_version, payload_path, payload,
                               tags, pinned, deprecated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 0, 0)
            """,
            (
                kind,
                name,
                version,
                hash_,
                float(ts) if ts is not None else time.time(),
                run_id,
                step,
                schema_kind,
                schema_version,
                payload_path,
                json.dumps(dict(payload)) if payload is not None else None,
            ),
        )
        return int(cur.lastrowid or 0)

    def find(self, kind: str, name: str | None, version: str | None) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE kind = ? AND name IS ? AND version IS ?",
            (kind, name, version),
        ).fetchone()
        return int(row["id"]) if row else None

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["tags"] = json.loads(d["tags"] or "[]")
        if d.get("payload"):
            try:
                d["payload"] = json.loads(d["payload"])
            except json.JSONDecodeError:
                pass
        return d

    def update_node_payload(
        self,
        node_id: int,
        payload: Mapping[str, Any],
        *,
        merge: bool = True,
    ) -> None:
        """Merge (or replace) ``payload`` on the existing node identified by id.

        ``merge=True`` (default) reads the current JSON payload and shallow-merges
        the new dict on top so callers can incrementally append fields without
        clobbering earlier writes (e.g. ``on_train_end`` adding ``ended_ts``
        without losing ``started_ts``). Used by ``LineageRecorderCallback`` to
        keep a single Run node per training run.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        new: dict[str, Any] = dict(payload)
        if merge:
            row = self.conn.execute(
                "SELECT payload FROM nodes WHERE id = ?", (int(node_id),)
            ).fetchone()
            if row is None:
                return
            cur = row["payload"]
            if cur:
                try:
                    merged = json.loads(cur)
                    if isinstance(merged, dict):
                        merged.update(new)
                        new = merged
                except json.JSONDecodeError:
                    pass
        self.conn.execute(
            "UPDATE nodes SET payload = ? WHERE id = ?",
            (json.dumps(new), int(node_id)),
        )

    def iter_nodes(self, *, kind: str | None = None) -> Iterator[dict[str, Any]]:
        if kind:
            rows = self.conn.execute("SELECT * FROM nodes WHERE kind = ?", (kind,))
        else:
            rows = self.conn.execute("SELECT * FROM nodes")
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"] or "[]")
            yield d

    # ---------------------------------------------------------------- edges

    def add_edge(
        self,
        src: int,
        dst: int,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if kind not in _EDGE_KINDS:
            raise ValueError(
                f"unknown lineage edge kind {kind!r}; expected one of {_EDGE_KINDS}"
            )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges (src, dst, kind, payload, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(src),
                int(dst),
                kind,
                json.dumps(dict(payload)) if payload else None,
                time.time(),
            ),
        )

    def iter_edges(self, *, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Iterate every edge in the graph (filter by kind if given).

        Used by ``lighttrain doctor`` to detect orphan edges in both
        directions — ``edges_from(existing_nodes)`` alone misses edges whose
        ``src`` node was deleted before this connection enabled FK cascade.
        """
        if kind:
            rows = self.conn.execute("SELECT * FROM edges WHERE kind = ?", (kind,))
        else:
            rows = self.conn.execute("SELECT * FROM edges")
        for r in rows:
            yield dict(r)

    def edges_from(self, src: int, *, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE src = ? AND kind = ?", (src, kind)
            )
        else:
            rows = self.conn.execute("SELECT * FROM edges WHERE src = ?", (src,))
        return [dict(r) for r in rows]

    def edges_to(self, dst: int, *, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE dst = ? AND kind = ?", (dst, kind)
            )
        else:
            rows = self.conn.execute("SELECT * FROM edges WHERE dst = ?", (dst,))
        return [dict(r) for r in rows]

    # ----- DAG queries -----------------------------------------------------

    def parents(self, node_id: int, *, edge_kind: str | None = None) -> list[int]:
        """For edge X (src → dst): X's *parents* are sources of edges into X
        plus destinations of edges X→Y depending on edge semantics.
        ``produced_by(run → artifact)`` treats the run as the parent. Both
        directions are exposed; the canonical 'derived_from' parents are
        ``edges_to`` srcs.
        """
        edges = self.edges_to(node_id, kind=edge_kind)
        return [int(e["src"]) for e in edges]

    def children(self, node_id: int, *, edge_kind: str | None = None) -> list[int]:
        edges = self.edges_from(node_id, kind=edge_kind)
        return [int(e["dst"]) for e in edges]

    def ancestors_until(self, node_id: int, *, kind: str) -> list[int]:
        """Walk ``derived_from`` + ``produced_by`` backwards, collect ids whose
        kind matches ``kind``. Bounded BFS, prevents cycles."""
        visited: set[int] = set()
        frontier: list[int] = [node_id]
        out: list[int] = []
        while frontier:
            nxt: list[int] = []
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                node = self.get_node(n)
                if node and node["kind"] == kind:
                    out.append(n)
                for src in self.parents(n, edge_kind="derived_from"):
                    nxt.append(src)
                for src in self.parents(n, edge_kind="produced_by"):
                    nxt.append(src)
            frontier = nxt
        return out

    # ----- tag / pin / invalidate -----------------------------------------

    def _tags_of(self, node_id: int) -> list[str]:
        row = self.conn.execute("SELECT tags FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return json.loads(row["tags"]) if row else []

    def tag(self, node_id: int, tag: str) -> None:
        tags = self._tags_of(node_id)
        if tag not in tags:
            tags.append(tag)
        self.conn.execute(
            "UPDATE nodes SET tags = ? WHERE id = ?",
            (json.dumps(tags), node_id),
        )

    def untag(self, node_id: int, tag: str) -> None:
        tags = [t for t in self._tags_of(node_id) if t != tag]
        self.conn.execute(
            "UPDATE nodes SET tags = ? WHERE id = ?",
            (json.dumps(tags), node_id),
        )

    def pin(self, node_id: int) -> None:
        self.conn.execute("UPDATE nodes SET pinned = 1 WHERE id = ?", (node_id,))

    def unpin(self, node_id: int) -> None:
        self.conn.execute("UPDATE nodes SET pinned = 0 WHERE id = ?", (node_id,))

    def invalidate(self, node_id: int) -> None:
        """Mark ``deprecated = 1`` and stamp ``deprecated_ts``. GC physically
        removes only after ``deprecated_ts + 24h``."""
        self.conn.execute(
            "UPDATE nodes SET deprecated = 1, deprecated_ts = ? WHERE id = ?",
            (time.time(), node_id),
        )

    def by_tag(self, tag: str, *, kind: str | None = None) -> list[int]:
        nodes = self.iter_nodes(kind=kind) if kind else self.iter_nodes()
        return [n["id"] for n in nodes if tag in n["tags"]]

    # ----- maintenance ----------------------------------------------------

    def delete_node(self, node_id: int) -> None:
        self.conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    def resolve_ref(self, ref: str) -> int | None:
        """Resolve ``"<kind>:<name>:<version>"`` (version defaults to ``latest``)
        or ``"#<id>"`` to a node id. Returns ``None`` when not found."""
        if ref.startswith("#"):
            try:
                return int(ref[1:])
            except ValueError:
                return None
        parts = ref.split(":")
        if len(parts) == 2:
            kind, name = parts
            version = None
        elif len(parts) == 3:
            kind, name, version = parts
        else:
            return None
        if version in (None, "latest"):
            row = self.conn.execute(
                "SELECT id FROM nodes WHERE kind = ? AND name = ? "
                "ORDER BY ts DESC, id DESC LIMIT 1",
                (kind, name),
            ).fetchone()
            return int(row["id"]) if row else None
        return self.find(kind, name, version)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "LineageStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["LineageStore"]
