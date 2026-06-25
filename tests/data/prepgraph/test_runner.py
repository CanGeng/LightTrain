"""Edge-case tests for ``lighttrain.data.prepgraph.runner``.

Coverage targets (uncovered lines at time of authoring):

* 134  — ``node_extras``: rfp-is-None defensive ``continue``
* 176–179, 181, 190 — ``run()``: cache-hit pre-load paths (rehydrate + plain NodeResult)
* 285, 287–288, 290–291, 293, 295–300, 303 — ``_rehydrate_cached``:
  shards.json branch and header.json (MemmapDataset) branch
* 353, 356, 360 — ``_explain_miss``: code_version_changed + upstream_changed
* 421, 424 — ``cleanup_orphans``: non-dir file skip inside name_dir / fp_dir
* 453–479 — ``_run_node_in_subprocess``: module-level subprocess entry-point
* 488, 492–494 — ``_rebind_store``: has-dir-attr but constructor fails fallback
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Registry side-effects must happen before PrepGraph.from_config is called.
from lighttrain.builtin_plugins.data import (
    processors as _processors,  # noqa: F401 — registry side-effect
)
from lighttrain.builtin_plugins.data.prepgraph import (
    nodes as _nodes,  # noqa: F401 — registry side-effect
)
from lighttrain.data.prepgraph import _io
from lighttrain.data.prepgraph.dag import PrepGraph
from lighttrain.data.prepgraph.node import NodeResult, PrepNode, RunContext
from lighttrain.data.prepgraph.runner import (
    PrepRunner,
    _rebind_store,
    _ResolvedFingerprint,
    _run_node_in_subprocess,
)

# ---------------------------------------------------------------------------
# Minimal node stubs
# ---------------------------------------------------------------------------


class _EchoNode(PrepNode):
    """Trivial node that echoes its config and returns in-memory rows."""

    kind = "dummy"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:
        return NodeResult(
            fingerprint="",
            rows=[{"val": self.config.get("val", 0)}],
        )


class _SinkNode(PrepNode):
    """Trivial sink: reads upstream rows; requires upstream to have rows."""

    kind = "dummy"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:
        up = next(iter(ctx.upstream.values()))
        rows = list(up.rows or [])
        return NodeResult(fingerprint="", rows=rows)


_ECHO_TARGET = f"{__name__}._EchoNode"
_SINK_TARGET = f"{__name__}._SinkNode"


def _echo_node_entry(name: str, val: int = 0, inputs: list[str] | None = None) -> dict:
    return {
        "name": name,
        "kind": "dummy",
        "_target_": _ECHO_TARGET,
        "inputs": list(inputs or []),
        "val": val,
    }


def _sink_node_entry(name: str, inputs: list[str]) -> dict:
    return {
        "name": name,
        "kind": "dummy",
        "_target_": _SINK_TARGET,
        "inputs": list(inputs),
    }


def _simple_spec(val: int = 0) -> dict:
    return {
        "nodes": [_echo_node_entry("src", val=val)],
        "terminals": ["src"],
    }


def _chain_spec(val: int = 0) -> dict:
    return {
        "nodes": [
            _echo_node_entry("src", val=val),
            _sink_node_entry("dst", ["src"]),
        ],
        "terminals": ["dst"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_corpus(tmp_path: Path) -> Path:
    """Minimal JSONL chat corpus with 4 rows."""
    p = tmp_path / "rows.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(4)
        ),
        encoding="utf-8",
    )
    return p


def _three_node_spec(jsonl: Path, p99_max: int = 4096) -> dict:
    return {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl}",
                "raw_data_version": "v0",
            },
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {"name": "byte"},
                },
            },
            {
                "name": "validated",
                "kind": "validate",
                "inputs": ["tok"],
                "p99_length_max": p99_max,
            },
        ],
        "terminals": ["validated"],
    }


def _mat_chain_spec(jsonl: Path, p99_max: int = 4096) -> dict:
    """raw → tok → mat (materialize) → validated chain."""
    return {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl}",
                "raw_data_version": "v0",
            },
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {"name": "byte"},
                },
            },
            {
                "name": "mat",
                "kind": "materialize",
                "inputs": ["tok"],
            },
            {
                "name": "validated",
                "kind": "validate",
                "inputs": ["mat"],
                "p99_length_max": p99_max,
            },
        ],
        "terminals": ["validated"],
    }


# ---------------------------------------------------------------------------
# node_extras: rfp-is-None defensive continue (line 134)
# ---------------------------------------------------------------------------


def test_pin_current_behavior_node_extras_skips_none_rfp(tmp_path: Path) -> None:
    """Pin: ``node_extras`` silently skips names whose rfp is None in ``_fp_cache``.

    The guard at line 134 (``if rfp is None: continue``) exists as a defensive
    belt-and-suspenders for a scenario that cannot happen through normal API
    usage (plan() always populates every name). We force it by post-hoc
    inserting ``None`` into the internal cache after plan() runs.

    This is a pin of defensive/unreachable behavior; flagged as such.
    """
    store = tmp_path / "store"
    runner = PrepRunner(PrepGraph.from_config(_simple_spec(val=1)), store_root=store)
    runner.run()

    # Now manipulate the cache to inject None for a synthetic key.
    # topo_order() will find "src" — whose real rfp is present — plus we also
    # inject a "ghost" node key so that when node_extras iterates topo_order()
    # it hits a name whose _fp_cache value is None.
    # The graph only has "src", so we'll patch _fp_cache directly.
    runner2 = PrepRunner(PrepGraph.from_config(_simple_spec(val=1)), store_root=store)
    runner2.plan()
    # Overwrite "src" entry with None to force the defensive branch.
    runner2._fp_cache["src"] = None  # type: ignore[assignment]
    result = runner2.node_extras()
    # "src" was skipped because its rfp is None → empty dict returned.
    assert result == {}


# ---------------------------------------------------------------------------
# run(): cache-hit pre-load path — plain NodeResult branch (line 181)
# ---------------------------------------------------------------------------


def test_invariant_run_cache_hit_plain_noderesult_for_hit_nodes(
    tmp_path: Path,
) -> None:
    """On second run all nodes are hits; the hit-node pre-load populates results
    with a plain NodeResult (line 181 branch).

    The test verifies that the second run's results contain fingerprints matching
    those from the first run (proving pre-load from cache happened, not re-execution).
    """
    store = tmp_path / "store"
    runner1 = PrepRunner(PrepGraph.from_config(_simple_spec(val=42)), store_root=store)
    res1 = runner1.run()
    fp1 = res1["src"].fingerprint

    runner2 = PrepRunner(PrepGraph.from_config(_simple_spec(val=42)), store_root=store)
    plan2 = runner2.plan()
    assert all(e.hit for e in plan2), "second run must be all cache hits"

    res2 = runner2.run()
    assert res2["src"].fingerprint == fp1


# ---------------------------------------------------------------------------
# run(): patch plan so must_execute is non-empty + line 190 (continue branch)
# ---------------------------------------------------------------------------


def test_invariant_run_skips_empty_layer(tmp_path: Path) -> None:
    """When a layer has no nodes in must_execute, the runner skips it (line 190).

    Setup: run once (everything misses). Run again (all hits). On second run
    every layer is empty in todo → the `continue` branch at line 190 fires.
    Verified by counting _run_one invocations on the second run.
    """
    store = tmp_path / "store"
    spec = _simple_spec(val=7)
    PrepRunner(PrepGraph.from_config(spec), store_root=store).run()

    runner2 = PrepRunner(PrepGraph.from_config(spec), store_root=store)
    called: list[str] = []
    real = runner2._run_one

    def _spy(name, results):
        called.append(name)
        return real(name, results)

    with patch.object(runner2, "_run_one", side_effect=_spy):
        runner2.run()

    assert called == [], f"No node should re-execute on full cache hit; called={called}"


# ---------------------------------------------------------------------------
# run(): rehydrate path (lines 178-179) — materialize cache hit with downstream
# ---------------------------------------------------------------------------


def test_invariant_run_rehydrate_path_for_materialize_hit(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """When a materialize node is a cache hit but its downstream misses,
    the runner rehydrates it (lines 178-179) rather than re-executing.

    Verified by patching ``_rehydrate_cached`` to record calls.
    """
    store = tmp_path / "store"
    PrepRunner(
        PrepGraph.from_config(_mat_chain_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    ).run()

    runner2 = PrepRunner(
        PrepGraph.from_config(_mat_chain_spec(jsonl_corpus, p99_max=2048)),
        store_root=store,
    )
    rehydrated: list[str] = []
    real_rehydrate = runner2._rehydrate_cached

    def _spy(node, rfp):
        rehydrated.append(node.name)
        return real_rehydrate(node, rfp)

    with patch.object(runner2, "_rehydrate_cached", side_effect=_spy):
        runner2.run()

    assert "mat" in rehydrated, (
        f"'mat' should have been rehydrated; got rehydrated={rehydrated}"
    )


# ---------------------------------------------------------------------------
# _rehydrate_cached: shards.json branch (lines 290-296)
# ---------------------------------------------------------------------------


def test_invariant_rehydrate_cached_shards_branch(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """``_rehydrate_cached`` with a materialize node that has shards.json loads
    a ``_RowsDataset`` and row list (lines 285, 290-296).

    After a full mat-chain run, call ``_rehydrate_cached`` directly.
    Expect rows to be a non-empty list and store to be non-None.
    """
    store = tmp_path / "store"
    runner = PrepRunner(
        PrepGraph.from_config(_mat_chain_spec(jsonl_corpus)), store_root=store
    )
    runner.run()

    mat_node = runner.graph.nodes["mat"]
    rfp = runner._fp_cache["mat"]

    result = runner._rehydrate_cached(mat_node, rfp)

    assert result.rows is not None and len(result.rows) > 0
    assert result.store is not None
    assert result.fingerprint == rfp.fp
    assert result.final_dir == rfp.final_dir


# ---------------------------------------------------------------------------
# _rehydrate_cached: no shards.json, no header.json → no rows/store
# (line 303 only — non-matching materialize dir)
# ---------------------------------------------------------------------------


def test_pin_current_behavior_rehydrate_cached_no_artifact_returns_empty(
    tmp_path: Path,
) -> None:
    """Pin: when a materialize node's final_dir has neither shards.json nor
    header.json, ``_rehydrate_cached`` returns a NodeResult with rows=None
    and store=None (defensive fallthrough, line 303).

    This can happen if a node wrote its MANIFEST_COMPLETE but no data shards
    (e.g. zero-row corpus). Pin current behavior.
    """
    final_dir = tmp_path / "mat" / "node_a" / "abc123"
    final_dir.mkdir(parents=True)
    _io.write_manifest(final_dir, {"kind": "materialize", "schema_version": "0.1"})

    class _FakeMatNode(PrepNode):
        kind = "materialize"
        schema_kind = "rows"

        def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover
            return NodeResult(fingerprint="")

    node = _FakeMatNode(name="node_a")
    rfp = _ResolvedFingerprint(
        fp="abc123",
        upstream_fps=[],
        final_dir=final_dir,
        hit=True,
        reason="cache_hit",
        schema_kind="rows",
        schema_version_known="0.1",
        schema_version_recorded="0.1",
    )
    store_root = tmp_path
    runner = PrepRunner.__new__(PrepRunner)
    runner.store_root = store_root
    runner.workers = 1
    runner.console = None
    runner.pool_kind = "thread"
    runner._fp_cache = {}
    # We need a minimal graph; inject a fake one.
    from lighttrain.data.prepgraph.dag import PrepGraph as _PG
    runner.graph = _PG(nodes={"node_a": node}, terminals=["node_a"], layers=[["node_a"]])

    result = runner._rehydrate_cached(node, rfp)
    assert result.rows is None
    assert result.store is None


# ---------------------------------------------------------------------------
# _rehydrate_cached: header.json branch (lines 297-300)
# ---------------------------------------------------------------------------


def test_invariant_rehydrate_cached_memmap_branch(tmp_path: Path) -> None:
    """``_rehydrate_cached`` with a materialize node whose final_dir has
    header.json (memmap layout) loads a ``MemmapDataset`` store
    (lines 297-300).
    """
    from lighttrain.data.cache._memmap import write_memmap

    final_dir = tmp_path / "memmap_node" / "fp_abc"
    final_dir.mkdir(parents=True)

    # Write a minimal memmap (1 row, seq_len=4).
    write_memmap(
        final_dir,
        [{"input_ids": [1, 2, 3, 4], "position_ids": [0, 1, 2, 3], "document_ids": [0, 0, 0, 0]}],
        seq_len=4,
    )
    # Write MANIFEST_COMPLETE so _io.is_complete() returns True.
    _io.write_manifest(final_dir, {"kind": "materialize", "schema_version": "0.1"})

    class _FakeMatNode(PrepNode):
        kind = "materialize"
        schema_kind = "rows"

        def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover
            return NodeResult(fingerprint="")

    node = _FakeMatNode(name="mm_node")
    rfp = _ResolvedFingerprint(
        fp="fp_abc",
        upstream_fps=[],
        final_dir=final_dir,
        hit=True,
        reason="cache_hit",
        schema_kind="rows",
        schema_version_known="0.1",
        schema_version_recorded="0.1",
    )

    runner = PrepRunner.__new__(PrepRunner)
    runner.store_root = tmp_path
    runner.workers = 1
    runner.console = None
    runner.pool_kind = "thread"
    runner._fp_cache = {}
    from lighttrain.data.prepgraph.dag import PrepGraph as _PG
    runner.graph = _PG(nodes={"mm_node": node}, terminals=["mm_node"], layers=[["mm_node"]])

    result = runner._rehydrate_cached(node, rfp)
    # header.json path: store should be a MemmapDataset, rows=None.
    from lighttrain.data.cache._memmap import MemmapDataset
    assert isinstance(result.store, MemmapDataset)
    assert result.rows is None


# ---------------------------------------------------------------------------
# _explain_miss: code_version_changed (line 356)
# ---------------------------------------------------------------------------


def test_invariant_explain_miss_code_version_changed(tmp_path: Path) -> None:
    """``_explain_miss`` returns ``code_version_changed`` when a sibling
    manifest carries a different ``code_version`` than the live node.

    We write a sibling manifest manually with a stale code_version and
    then invoke ``_explain_miss`` with a node whose ``code_version()`` returns
    a different hash.
    """
    node = _EchoNode(name="n1", config={"val": 0})
    node.code_version()

    # Create a sibling dir under store_root/dummy/n1/old_fp/
    sibling_dir = tmp_path / "dummy" / "n1" / "old_fp"
    sibling_dir.mkdir(parents=True)
    _io.write_manifest(
        sibling_dir,
        {
            "kind": "dummy",
            "name": "n1",
            "code_version": "stale_hash_000000000000",
            "config": {"val": 0},
            "schema_version": "0.1",
        },
    )

    runner = PrepRunner.__new__(PrepRunner)
    runner.store_root = tmp_path
    runner.workers = 1
    runner.console = None
    runner.pool_kind = "thread"
    runner._fp_cache = {}
    from lighttrain.data.prepgraph.dag import PrepGraph as _PG
    runner.graph = _PG(nodes={"n1": node}, terminals=["n1"], layers=[["n1"]])

    reason = runner._explain_miss(node, "n1", tmp_path / "dummy" / "n1" / "new_fp")
    assert reason == "code_version_changed", (
        f"Expected 'code_version_changed' because sibling has stale cv; got {reason!r}"
    )


# ---------------------------------------------------------------------------
# _explain_miss: upstream_changed (line 360)
# ---------------------------------------------------------------------------


def test_invariant_explain_miss_upstream_changed(tmp_path: Path) -> None:
    """``_explain_miss`` returns ``upstream_changed`` when sibling manifests
    exist but neither code_version nor config differ — implying an upstream
    fingerprint changed (line 360).
    """
    node = _EchoNode(name="n2", config={"val": 5})
    live_cv = node.code_version()

    sibling_dir = tmp_path / "dummy" / "n2" / "old_fp"
    sibling_dir.mkdir(parents=True)
    _io.write_manifest(
        sibling_dir,
        {
            "kind": "dummy",
            "name": "n2",
            "code_version": live_cv,  # same → no code_version_changed
            "config": {"val": 5},     # same → no config_changed
            "schema_version": "0.1",
        },
    )

    runner = PrepRunner.__new__(PrepRunner)
    runner.store_root = tmp_path
    runner.workers = 1
    runner.console = None
    runner.pool_kind = "thread"
    runner._fp_cache = {}
    from lighttrain.data.prepgraph.dag import PrepGraph as _PG
    runner.graph = _PG(nodes={"n2": node}, terminals=["n2"], layers=[["n2"]])

    reason = runner._explain_miss(node, "n2", tmp_path / "dummy" / "n2" / "new_fp")
    assert reason == "upstream_changed", (
        f"Expected 'upstream_changed'; got {reason!r}"
    )


# ---------------------------------------------------------------------------
# _explain_miss: sibling dir exists but has no manifest → falls through
# ---------------------------------------------------------------------------


def test_invariant_explain_miss_sibling_no_manifest_upstream_changed(
    tmp_path: Path,
) -> None:
    """When all sibling dirs exist but none have a valid manifest, ``_explain_miss``
    still returns ``upstream_changed`` (the only remaining fallthrough after
    skipping the ``if not sib_manifest`` guard at line 353).
    """
    node = _EchoNode(name="n3", config={"val": 1})

    # Sibling dir with NO manifest (just empty dir)
    sibling_dir = tmp_path / "dummy" / "n3" / "fp_empty"
    sibling_dir.mkdir(parents=True)

    runner = PrepRunner.__new__(PrepRunner)
    runner.store_root = tmp_path
    runner.workers = 1
    runner.console = None
    runner.pool_kind = "thread"
    runner._fp_cache = {}
    from lighttrain.data.prepgraph.dag import PrepGraph as _PG
    runner.graph = _PG(nodes={"n3": node}, terminals=["n3"], layers=[["n3"]])

    reason = runner._explain_miss(node, "n3", tmp_path / "dummy" / "n3" / "new_fp")
    assert reason == "upstream_changed"


# ---------------------------------------------------------------------------
# cleanup_orphans: non-dir file skip inside name_dir (line 421)
# ---------------------------------------------------------------------------


def test_invariant_cleanup_orphans_skips_non_dir_in_name_dir(
    tmp_path: Path,
) -> None:
    """A regular file inside a <kind>/<name>/ dir is not treated as an fp_dir
    (line 421 ``if not name_dir.is_dir(): continue``).

    We plant a stray file inside the name_dir and ensure cleanup_orphans
    does not choke on it.
    """
    store = tmp_path / "store"
    spec = _simple_spec(val=3)
    PrepRunner(PrepGraph.from_config(spec), store_root=store).run()

    # Locate the kind/name dir and plant a plain file alongside the fp_dir.
    kind_dir = store / "dummy" / "src"
    stray_file = kind_dir / "stray.txt"
    stray_file.write_text("not a dir", encoding="utf-8")

    runner = PrepRunner(PrepGraph.from_config(spec), store_root=store)
    removed = runner.cleanup_orphans(dry_run=True)
    # The current fingerprint is live → nothing removed.
    assert removed == []
    # Stray file must still be there (not deleted).
    assert stray_file.exists()


# ---------------------------------------------------------------------------
# cleanup_orphans: non-dir file skip inside fp_dir (line 424)
# ---------------------------------------------------------------------------


def test_invariant_cleanup_orphans_skips_non_dir_in_fp_parent(
    tmp_path: Path,
) -> None:
    """A regular file inside a <kind>/<name>/ dir at the fp_dir level is
    skipped by ``if not fp_dir.is_dir(): continue`` (line 424).

    Plant such a file and confirm cleanup_orphans ignores it.
    """
    store = tmp_path / "store"
    spec = _simple_spec(val=4)
    PrepRunner(PrepGraph.from_config(spec), store_root=store).run()

    # Plant a plain file in the name_dir (not a real fp directory).
    kind_name_dir = store / "dummy" / "src"
    plain_file = kind_name_dir / "loose_file.json"
    plain_file.write_text("{}", encoding="utf-8")

    runner = PrepRunner(PrepGraph.from_config(spec), store_root=store)
    removed = runner.cleanup_orphans(dry_run=True)
    assert removed == []
    assert plain_file.exists()


# ---------------------------------------------------------------------------
# cleanup_orphans: actual removal of orphan fp_dir
# ---------------------------------------------------------------------------


def test_invariant_cleanup_orphans_removes_orphan_fp_dir(tmp_path: Path) -> None:
    """A fingerprint directory not referenced by any live node is removed
    and returned in the list when dry_run=False.

    We plant a fake fp_dir with a manifest under the right kind/name path to
    simulate a stale prior run, then sweep with the new runner.
    """
    store = tmp_path / "store"
    spec = _simple_spec(val=5)
    PrepRunner(PrepGraph.from_config(spec), store_root=store).run()

    # Plant an orphaned fp_dir.
    orphan_dir = store / "dummy" / "src" / "deadbeef00000000deadbeef"
    orphan_dir.mkdir(parents=True)
    _io.write_manifest(orphan_dir, {"kind": "dummy", "schema_version": "0.1"})

    runner = PrepRunner(PrepGraph.from_config(spec), store_root=store)
    removed = runner.cleanup_orphans(dry_run=False)
    assert any(p.name == "deadbeef00000000deadbeef" for p in removed)
    assert not orphan_dir.exists()


# ---------------------------------------------------------------------------
# _run_node_in_subprocess (lines 453-479)
# ---------------------------------------------------------------------------


def test_invariant_run_node_in_subprocess_produces_committed_manifest(
    tmp_path: Path,
) -> None:
    """``_run_node_in_subprocess`` mirrors ``_run_one``: it writes to staging,
    commits to final_dir, and returns a NodeResult whose fingerprint and
    final_dir are set.

    Called directly (not via ProcessPool) to keep the test synchronous and
    deterministic.
    """
    node = _EchoNode(name="direct", config={"val": 99})
    fp = node.fingerprint([])
    store_root = tmp_path / "store"
    final_dir = _io.final_dir(store_root, node.kind, node.name, fp)

    result = _run_node_in_subprocess(
        node=node,
        fp=fp,
        upstream_fps=[],
        final_dir=final_dir,
        upstream={},
        store_root=store_root,
        workers=1,
    )

    assert result.fingerprint == fp
    assert result.final_dir == final_dir
    assert _io.is_complete(final_dir)


def test_invariant_run_node_in_subprocess_overwrites_existing_staging(
    tmp_path: Path,
) -> None:
    """When the staging dir already exists (stale), ``_run_node_in_subprocess``
    removes it first (line 454-455) and re-creates it cleanly.
    """
    node = _EchoNode(name="stale", config={"val": 1})
    fp = node.fingerprint([])
    store_root = tmp_path / "store"
    final_dir = _io.final_dir(store_root, node.kind, node.name, fp)

    # Pre-create the staging dir with junk content.
    staging = _io.staging_dir(store_root, fp)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "leftover.txt").write_text("stale", encoding="utf-8")

    result = _run_node_in_subprocess(
        node=node,
        fp=fp,
        upstream_fps=[],
        final_dir=final_dir,
        upstream={},
        store_root=store_root,
        workers=1,
    )

    assert result.fingerprint == fp
    assert _io.is_complete(final_dir)
    # Leftover must be gone (staging was wiped).
    assert not (final_dir / "leftover.txt").exists()


def test_invariant_run_node_in_subprocess_with_upstream(tmp_path: Path) -> None:
    """``_run_node_in_subprocess`` correctly passes upstream NodeResults into
    ``RunContext.upstream`` (line 458-463).

    We give _SinkNode an upstream result and confirm it sees the rows.
    """
    src_node = _EchoNode(name="src", config={"val": 42})
    src_fp = src_node.fingerprint([])
    src_store = tmp_path / "src_store"
    src_final = _io.final_dir(src_store, src_node.kind, src_node.name, src_fp)
    # Run src node via subprocess helper first.
    src_result = _run_node_in_subprocess(
        node=src_node,
        fp=src_fp,
        upstream_fps=[],
        final_dir=src_final,
        upstream={},
        store_root=src_store,
        workers=1,
    )

    sink_node = _SinkNode(name="sink", inputs=["src"])
    sink_fp = sink_node.fingerprint([src_fp])
    sink_store = tmp_path / "sink_store"
    sink_final = _io.final_dir(sink_store, sink_node.kind, sink_node.name, sink_fp)

    sink_result = _run_node_in_subprocess(
        node=sink_node,
        fp=sink_fp,
        upstream_fps=[src_fp],
        final_dir=sink_final,
        upstream={"src": src_result},
        store_root=sink_store,
        workers=1,
    )

    assert sink_result.rows is not None
    assert len(sink_result.rows) == 1
    assert sink_result.rows[0]["val"] == 42


# ---------------------------------------------------------------------------
# _rebind_store: store with recognized dir attribute is rebound (line 491)
# ---------------------------------------------------------------------------


def test_invariant_rebind_store_none_returns_none() -> None:
    """``_rebind_store(None, ...)`` returns ``None`` immediately (line 484)."""
    result = _rebind_store(None, Path("/tmp"))
    assert result is None


def test_invariant_rebind_store_no_dir_attr_returns_original(tmp_path: Path) -> None:
    """When the store has neither ``out_dir`` nor ``root``, ``_rebind_store``
    returns the original store unchanged (line 488).
    """

    class _NoAttrStore:
        pass

    store = _NoAttrStore()
    result = _rebind_store(store, tmp_path / "final")
    assert result is store


def test_invariant_rebind_store_out_dir_attr_rebinds(tmp_path: Path) -> None:
    """When the store has an ``out_dir`` attribute, ``_rebind_store`` reconstructs
    the store pointing at ``final_dir`` (line 491).
    """
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    class _StoreWithOutDir:
        def __init__(self, d: Path) -> None:
            self.out_dir = d

    store = _StoreWithOutDir(tmp_path / "staging")
    result = _rebind_store(store, final_dir)
    assert isinstance(result, _StoreWithOutDir)
    assert result.out_dir == final_dir


def test_invariant_rebind_store_root_attr_rebinds(tmp_path: Path) -> None:
    """When the store has a ``root`` attribute (but no ``out_dir``), ``_rebind_store``
    reconstructs the store pointing at ``final_dir`` (line 486 alternate path).
    """
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    class _StoreWithRoot:
        def __init__(self, d: Path) -> None:
            self.root = d

    store = _StoreWithRoot(tmp_path / "staging")
    result = _rebind_store(store, final_dir)
    assert isinstance(result, _StoreWithRoot)
    assert result.root == final_dir


def test_pin_current_behavior_rebind_store_constructor_raises_returns_original(
    tmp_path: Path,
) -> None:
    """Pin: when the store has an ``out_dir`` attr but its constructor raises,
    ``_rebind_store`` logs a warning and returns the original store (lines 492-494).

    This is documented as a fallback for exotic store types; pin current behavior.
    """
    _SENTINEL = object()

    class _BrokenStore:
        """Raises on construction except when called with the sentinel."""

        def __init__(self, d: Any) -> None:
            if d is not _SENTINEL:
                raise RuntimeError("cannot rebind to arbitrary path")
            self.out_dir = tmp_path / "staging"

    # Build the original store using the sentinel bypass.
    store = _BrokenStore(_SENTINEL)
    final_dir = tmp_path / "final"
    final_dir.mkdir()

    result = _rebind_store(store, final_dir)
    # Constructor raised → falls back to original store.
    assert result is store


# ---------------------------------------------------------------------------
# _rebind_store: out_dir=None but root=None → no attr → return original
# ---------------------------------------------------------------------------


def test_invariant_rebind_store_out_dir_none_root_attr_used(tmp_path: Path) -> None:
    """When ``out_dir`` is ``None`` (falsy), ``_rebind_store`` falls through to
    ``root`` (``or getattr(store, 'root', None)``). If root is also None,
    returns the original.
    """

    class _NullOutDir:
        out_dir = None

    store = _NullOutDir()
    result = _rebind_store(store, tmp_path)
    # out_dir is None (falsy), root attr missing → out_dir_attr = None → return store
    assert result is store


# ---------------------------------------------------------------------------
# Integration: node_extras returns domain metrics, framework keys stripped
# ---------------------------------------------------------------------------


def test_invariant_node_extras_strips_framework_keys(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """``node_extras`` returns extras without any framework manifest key
    and includes manifest-level domain keys from nodes that write them.
    """
    store = tmp_path / "store"
    spec = {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl_corpus}",
                "raw_data_version": "v0",
            },
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {"name": "byte"},
                },
            },
            {
                "name": "validated",
                "kind": "validate",
                "inputs": ["tok"],
                "p99_length_max": 4096,
            },
        ],
        "terminals": ["validated"],
    }
    runner = PrepRunner(PrepGraph.from_config(spec), store_root=store)
    runner.run()
    extras = runner.node_extras()

    _FRAMEWORK_KEYS = frozenset({
        "kind", "name", "schema_kind", "schema_version", "fingerprint",
        "code_version", "config", "lineage_pending", "derived_from", "elapsed_s",
    })
    for node_name, node_extras in extras.items():
        overlap = set(node_extras.keys()) & _FRAMEWORK_KEYS
        assert not overlap, (
            f"Node {node_name!r} extras contain framework keys: {overlap}"
        )


# ---------------------------------------------------------------------------
# Integration: _explain_miss config_changed path (already covered by existing
# test_dag_and_runner but we ensure the reason field is correct here too)
# ---------------------------------------------------------------------------


def test_invariant_explain_miss_config_changed_reason(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """After changing a node's config, ``plan()`` returns reason='config_changed'
    for that node and its fingerprint-dependent descendants.
    """
    store = tmp_path / "store"
    PrepRunner(
        PrepGraph.from_config(_three_node_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    ).run()

    plan = PrepRunner(
        PrepGraph.from_config(_three_node_spec(jsonl_corpus, p99_max=2048)),
        store_root=store,
    ).plan()
    by_name = {e.name: e for e in plan}
    assert not by_name["validated"].hit
    assert by_name["validated"].reason == "config_changed"


# ---------------------------------------------------------------------------
# Integration: run() banner patch — rerun_for_downstream entry.reason update
# ---------------------------------------------------------------------------


def test_invariant_run_patches_plan_rerun_for_downstream(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """When a non-materialize cache hit is demoted to must_execute,
    ``run()`` patches the PlanEntry.reason to 'rerun_for_downstream' in-place
    (lines 165-167).

    We capture the plan list post-run and confirm the demoted entry changed.
    """
    store = tmp_path / "store"
    PrepRunner(
        PrepGraph.from_config(_three_node_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    ).run()

    runner2 = PrepRunner(
        PrepGraph.from_config(_three_node_spec(jsonl_corpus, p99_max=2048)),
        store_root=store,
    )
    # Capture the plan object before run mutates it.
    plan = runner2.plan()
    # Re-attach the same plan object to inspect post-mutation.
    plan_ref = plan

    # Patch run() to inject the same plan we captured.

    def _return_same_plan():
        return plan_ref

    with patch.object(runner2, "plan", side_effect=_return_same_plan):
        runner2.run()

    by_name = {e.name: e for e in plan_ref}
    # tok was a hit but must rerun (non-materialize upstream of miss).
    # After run(), its reason should be 'rerun_for_downstream'.
    assert by_name["tok"].reason == "rerun_for_downstream"
    assert not by_name["tok"].hit
