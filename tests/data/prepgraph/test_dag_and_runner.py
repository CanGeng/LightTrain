"""Adversarial tests for ``lighttrain.data.prepgraph.dag`` + ``runner``.

Targets that the legacy tests miss:
  * Exact layer assignment from Kahn's algorithm (not just "ran without error")
  * Cycle / dangling-input / duplicate-name rejection with ``ValueError``
  * The historical ``PREP_PARTIAL_01`` fix (docs/changelog/v0.1.3 + the
    docstring of ``tests/test_prepgraph_partial_cache.py``): non-materialize
    cache hits must be **re-executed** when any descendant misses. The test
    here pins re-execution via a runner-internal call counter, going beyond
    the row-count check the legacy reproducer uses.
  * Cache-hit reason codes (``cache_hit`` / ``schema_version_bumped`` /
    miss reasons) — the legacy tests check ``hit`` but not the reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure prep_node kinds are registered so PrepGraph.from_config can resolve
# the real node classes used by the partial-cache reproducer.
from lighttrain.builtin_plugins.data import (
    processors as _processors,  # noqa: F401 — registry side-effect
)
from lighttrain.builtin_plugins.data.prepgraph import (
    nodes as _nodes,  # noqa: F401 — registry side-effect
)
from lighttrain.data.prepgraph._fp import SCHEMA_VERSION
from lighttrain.data.prepgraph.dag import PrepGraph
from lighttrain.data.prepgraph.node import NodeResult, PrepNode, RunContext
from lighttrain.data.prepgraph.runner import PrepRunner

# --------------------------------------------------------------------------- #
# Dummy PrepNodes used by the topology tests                                  #
# (Referenced via ``_target_`` so we do not mutate the global registry.)      #
# --------------------------------------------------------------------------- #


class _DummyNode(PrepNode):
    """Trivial node: does nothing — used only for DAG topology tests."""

    kind = "dummy"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover — not invoked
        return NodeResult(fingerprint="x", rows=[])


_DUMMY_TARGET = f"{__name__}._DummyNode"


def _node_entry(name: str, inputs: list[str] | None = None) -> dict:
    return {
        "name": name,
        "kind": "dummy",
        "_target_": _DUMMY_TARGET,
        "inputs": list(inputs or []),
    }


# --------------------------------------------------------------------------- #
# Topology + Kahn-layering correctness                                        #
# --------------------------------------------------------------------------- #


def test_topology_layered_kahn_correctness() -> None:
    """Layers from ``_topo_layers`` exactly match Kahn's algorithm result.

    Input DAG (edges A→B, A→C, B→D, C→D):
        A → B → D
         ↘ C ↗
    Analytical (Kahn's layered form):
        layer 0: [A]            (in-degree 0)
        layer 1: [B, C]         (in-degree drops to 0 after A removed)
        layer 2: [D]            (in-degree drops to 0 after B,C removed)
    """
    spec = {
        "nodes": [
            _node_entry("A"),
            _node_entry("B", ["A"]),
            _node_entry("C", ["A"]),
            _node_entry("D", ["B", "C"]),
        ],
        "terminals": ["D"],
    }
    graph = PrepGraph.from_config(spec)
    # Within-layer order is sorted (per Kahn loop in dag.py:103).
    assert graph.layers == [["A"], ["B", "C"], ["D"]]


def test_topology_detects_cycle() -> None:
    """A→B and B→A is a 2-cycle; from_config must raise ``ValueError``.

    Contract: cycles are caught at construction, not at runtime.
    """
    spec = {
        "nodes": [
            _node_entry("A", ["B"]),
            _node_entry("B", ["A"]),
        ],
        "terminals": ["A"],
    }
    with pytest.raises(ValueError, match="cycle"):
        PrepGraph.from_config(spec)


def test_topology_detects_dangling_input() -> None:
    """A node referencing an unknown upstream raises ``ValueError``.

    Contract: schema validation is eager, no silent skip.
    """
    spec = {
        "nodes": [
            _node_entry("A", ["MISSING"]),
        ],
        "terminals": ["A"],
    }
    with pytest.raises(ValueError, match="unknown input"):
        PrepGraph.from_config(spec)


def test_topology_duplicate_node_names_rejected() -> None:
    """Two nodes with the same ``name`` raise ``ValueError`` at construction.

    Contract: names are unique within a graph.
    """
    spec = {
        "nodes": [
            _node_entry("A"),
            _node_entry("A"),
        ],
        "terminals": ["A"],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        PrepGraph.from_config(spec)


# --------------------------------------------------------------------------- #
# Partial-cache regression (PREP_PARTIAL_01)                                  #
# --------------------------------------------------------------------------- #


def _partial_cache_spec(jsonl: Path, p99_max: int) -> dict:
    """The exact reproducer spec from tests/test_prepgraph_partial_cache.py."""
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
                    "tokenizer": {
                        "name": "byte"
                    },
                },
            },
            {
                "name": "validated",
                "kind": "validate",
                "inputs": ["tok"],
                "p99_length_max": int(p99_max),
            },
        ],
        "terminals": ["validated"],
    }


@pytest.fixture
def jsonl_corpus(tmp_path: Path) -> Path:
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
            for i in range(6)
        ),
        encoding="utf-8",
    )
    return p


def test_regression_PREP_PARTIAL_01_non_materialize_demoted(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """Pre-fix bug: partial-cache reuse left a non-materialize upstream as an
    inert hit; the downstream consumer saw ``upstream.rows = None`` and
    crashed with ``ValidateNode 'validated': only 0 rows, need >= 1`` (see
    tests/test_prepgraph_partial_cache.py:1-8 — the existing reproducer
    documents the pre-fix behavior).

    Input: pipeline ``raw → tok → validated``. Pass 1 runs everything.
    Pass 2 changes only validate's threshold so validate misses; tok stays
    a fingerprint hit and is non-materialize.

    Analytical: post-fix, ``_compute_runtime_sets`` walks the dependency
    graph back from any miss and demotes non-materialize hits to
    ``must_execute``. So in pass 2, ``raw`` and ``tok`` BOTH re-execute,
    not just ``validated``.

    The pin goes beyond the legacy test's "rows == 6" check: it captures
    ``_run_one`` invocations by node name and asserts ``tok`` was actually
    called again (not merely hit). Without the demotion the runner would
    skip ``tok`` and ``validated`` would see ``upstream.rows = None``.
    """
    store_root = tmp_path / "store"

    # Pass 1: full run, everything misses, all manifests written.
    runner1 = PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=4096)),
        store_root=store_root,
    )
    assert all(not e.hit for e in runner1.plan())
    runner1.run()

    # Pass 2: tighten threshold — only ``validated``'s fingerprint changes.
    runner2 = PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=2048)),
        store_root=store_root,
    )
    plan2 = runner2.plan()
    by_name = {e.name: e for e in plan2}
    assert by_name["raw"].hit
    assert by_name["tok"].hit
    assert not by_name["validated"].hit  # fingerprint really changed

    # Capture which nodes _run_one actually executes during run().
    executed: list[str] = []
    real_run_one = runner2._run_one

    def _spy(name: str, results: dict) -> NodeResult:
        executed.append(name)
        return real_run_one(name, results)

    with patch.object(runner2, "_run_one", side_effect=_spy):
        results = runner2.run()

    # The pin: ``tok`` (non-materialize cache hit) MUST be re-executed
    # because ``validated`` (downstream) is going to execute and would
    # otherwise read ``upstream.rows = None``.
    assert "tok" in executed, (
        "Pre-fix: non-materialize cache hit 'tok' stays inert; "
        f"executed only {executed}"
    )
    assert "validated" in executed
    # And the legacy row-count safety check still holds.
    report = results["validated"].extras.get("report", {})
    assert report.get("rows", 0) == 6


def test_invariant_partial_cache_only_demotes_non_materialize(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """A materialize-kind upstream cache hit is NOT demoted; instead it is
    rehydrated.

    Input: pipeline ``raw → tok → mat → validated``. Pass 1 runs everything,
    persisting ``mat`` to disk. Pass 2 changes only ``validated``'s threshold.

    Analytical:
        - ``mat`` is materialize → its rows ARE on disk; it must NOT
          re-execute.
        - It DOES need to be rehydrated so downstream consumes its rows.

    The invariant: under the partial-cache reuse policy, materialize hits
    are rehydrated, not re-executed; only non-materialize hits are demoted.
    """
    store_root = tmp_path / "store"

    def _spec(p99_max: int) -> dict:
        return {
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
                        "tokenizer": {
                            "name": "byte"
                        },
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
                    "p99_length_max": int(p99_max),
                },
            ],
            "terminals": ["validated"],
        }

    runner1 = PrepRunner(PrepGraph.from_config(_spec(4096)), store_root=store_root)
    runner1.run()

    runner2 = PrepRunner(PrepGraph.from_config(_spec(2048)), store_root=store_root)
    plan2 = runner2.plan()
    must_execute, must_rehydrate = runner2._compute_runtime_sets(plan2)

    # validated misses → must_execute. mat is materialize hit → NOT in must_execute
    # but must_rehydrate (because its direct downstream `validated` executes).
    assert "validated" in must_execute
    assert "mat" not in must_execute
    assert "mat" in must_rehydrate


# --------------------------------------------------------------------------- #
# Cache-hit reason codes                                                      #
# --------------------------------------------------------------------------- #


def test_cache_hit_reason_codes(tmp_path: Path, jsonl_corpus: Path, monkeypatch) -> None:
    """The plan's ``reason`` field matches well-defined codes.

    Inputs:
      * unrun graph → ``reason == "first_run"``
      * fully cached → ``reason == "cache_hit"``
      * bump SCHEMA_VERSION → ``reason == "schema_version_bumped"``

    Analytical: the codes are produced by ``_resolve`` and ``_explain_miss``
    in runner.py and are part of the public banner contract.
    """
    store_root = tmp_path / "store"
    spec = _partial_cache_spec(jsonl_corpus, p99_max=4096)
    runner = PrepRunner(PrepGraph.from_config(spec), store_root=store_root)

    plan = runner.plan()
    by_name_first = {e.name: e for e in plan}
    assert by_name_first["raw"].reason == "first_run"

    runner.run()
    runner2 = PrepRunner(PrepGraph.from_config(spec), store_root=store_root)
    plan2 = runner2.plan()
    {e.name: e for e in plan2}
    assert all(e.hit for e in plan2)
    assert all(e.reason == "cache_hit" for e in plan2)

    # Bump schema version for "rows" so tokenize/validate manifests look stale.
    bumped_schema = dict(SCHEMA_VERSION)
    bumped_schema["rows"] = SCHEMA_VERSION["rows"] + "-x"
    monkeypatch.setattr(
        "lighttrain.data.prepgraph.runner.SCHEMA_VERSION", bumped_schema, raising=True
    )
    runner3 = PrepRunner(PrepGraph.from_config(spec), store_root=store_root)
    plan3 = runner3.plan()
    {e.name: e for e in plan3}
    # At least one rows-schema node must now report the bump.
    bumped = [e for e in plan3 if e.reason == "schema_version_bumped"]
    assert bumped, f"no node reported schema_version_bumped: {[(e.name, e.reason) for e in plan3]}"


# --------------------------------------------------------------------------- #
# Runner: dry-run, config-change reason codes, orphan cleanup                  #
# --------------------------------------------------------------------------- #


def test_dry_run_writes_nothing_to_store(tmp_path: Path, jsonl_corpus: Path) -> None:
    """``dry_run`` plans without materializing any artifact to the store dir.

    Contract: a dry-run is side-effect-free on disk; the store dir is either
    absent or empty afterwards.
    """
    store = tmp_path / "store"
    runner = PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    )
    runner.dry_run()
    assert not store.exists() or not any(store.iterdir())


def test_config_change_reports_config_changed_and_propagates_upstream(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """Changing one node's config flips its reason to ``config_changed`` and
    its descendants to an upstream-driven miss reason.

    Setup: run once, then tighten ``validated``'s ``p99_length_max``. Only
    ``validated``'s own fingerprint changes.
    Analytical: ``validated`` misses with ``reason == "config_changed"``;
    ``raw``/``tok`` remain hits (their fingerprints are unchanged).
    """
    store = tmp_path / "store"
    PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    ).run()

    plan = PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=2048)),
        store_root=store,
    ).plan()
    by_name = {e.name: e for e in plan}
    assert by_name["raw"].hit
    assert by_name["tok"].hit
    assert not by_name["validated"].hit
    assert by_name["validated"].reason == "config_changed"


def test_cleanup_orphans_dry_run_keeps_live_fingerprints(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """``cleanup_orphans(dry_run=True)`` on an unchanged graph flags nothing.

    Regression: cleanup on a graph whose nodes have inputs used to raise
    ``KeyError`` because ``_fp_cache`` was not populated before ``_resolve``.
    All current fingerprints are live → the removal list is empty and no
    exception is raised.
    """
    store = tmp_path / "store"
    spec = _partial_cache_spec(jsonl_corpus, p99_max=4096)
    PrepRunner(PrepGraph.from_config(spec), store_root=store).run()
    removed = PrepRunner(
        PrepGraph.from_config(spec), store_root=store
    ).cleanup_orphans(dry_run=True)
    assert removed == []


def test_cleanup_orphans_removes_stale_fingerprint_dirs(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """After a config change re-run, the prior fingerprint dirs are orphaned
    and ``cleanup_orphans`` removes them.

    Setup: run with one threshold, then run with a tightened threshold (which
    re-materializes ``validated`` under a new fingerprint), then sweep.
    The original ``validated`` fp dir is no longer live → it is removed.
    """
    store = tmp_path / "store"
    PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=4096)),
        store_root=store,
    ).run()
    runner2 = PrepRunner(
        PrepGraph.from_config(_partial_cache_spec(jsonl_corpus, p99_max=2048)),
        store_root=store,
    )
    runner2.run()
    removed = runner2.cleanup_orphans()
    assert removed


# --------------------------------------------------------------------------- #
# Node behaviors: validate report, chunk splitting, mix round-robin            #
# --------------------------------------------------------------------------- #


def test_validate_node_persists_report_json(tmp_path: Path, jsonl_corpus: Path) -> None:
    """A ``validate`` node writes a ``report.json`` with a positive row count
    and the histogram/keep-ratio diagnostic keys.

    Pin: the validate node materializes its report to disk so ``prep-status``
    can surface it.
    """
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
                "name": "checked",
                "kind": "validate",
                "inputs": ["tok"],
                "vocab_size": 260,
                "min_rows": 1,
            },
        ],
        "terminals": ["checked"],
    }
    runner = PrepRunner(PrepGraph.from_config(spec), store_root=tmp_path / "store")
    runner.run()
    final_dir = next((tmp_path / "store" / "validate" / "checked").iterdir())
    report = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
    assert report["rows"] >= 1
    assert "length_histogram" in report
    assert "label_keep_ratio" in report


def test_chunk_node_splits_long_rows_within_max_len(tmp_path: Path) -> None:
    """A ``chunk`` node splits an over-long tokenized row into multiple rows,
    each no longer than ``max_len``.

    Setup: one 500-char doc, ``max_len=64``, ``overlap=8``.
    Expected: more than one output row, every row length <= 64.
    """
    p = tmp_path / "long.jsonl"
    p.write_text(json.dumps({"text": "x" * 500}), encoding="utf-8")
    spec = {
        "nodes": [
            {"name": "raw", "kind": "load", "source": f"jsonl:{p}"},
            {
                "name": "tok",
                "kind": "tokenize",
                "inputs": ["raw"],
                "tokenizer": {"name": "byte"},
                "text_field": "text",
            },
            {
                "name": "chunked",
                "kind": "chunk",
                "inputs": ["tok"],
                "max_len": 64,
                "overlap": 8,
            },
        ],
        "terminals": ["chunked"],
    }
    results = PrepRunner(PrepGraph.from_config(spec), store_root=tmp_path / "store").run()
    rows = results["chunked"].rows
    assert len(rows) > 1  # type: ignore[arg-type]
    assert all(len(r["input_ids"]) <= 64 for r in rows)  # type: ignore[union-attr]


def test_mix_node_round_robin_interleaves_sources(tmp_path: Path) -> None:
    """A ``mix`` node with ``strategy='round_robin'`` interleaves its two
    upstream sources a0 b0 a1 b1 ...

    Setup: two 3-row sources. Expected: 6 rows total, row 0 from src_a, row 1
    from src_b.
    """
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("\n".join(json.dumps({"text": f"a{i}"}) for i in range(3)), encoding="utf-8")
    p2.write_text("\n".join(json.dumps({"text": f"b{i}"}) for i in range(3)), encoding="utf-8")
    spec = {
        "nodes": [
            {"name": "src_a", "kind": "load", "source": f"jsonl:{p1}"},
            {"name": "src_b", "kind": "load", "source": f"jsonl:{p2}"},
            {
                "name": "mixed",
                "kind": "mix",
                "inputs": ["src_a", "src_b"],
                "strategy": "round_robin",
            },
        ],
        "terminals": ["mixed"],
    }
    results = PrepRunner(PrepGraph.from_config(spec), store_root=tmp_path / "store").run()
    rows = results["mixed"].rows
    assert len(rows) == 6  # type: ignore[arg-type]
    assert rows[0]["text"].startswith("a")  # type: ignore[index]
    assert rows[1]["text"].startswith("b")  # type: ignore[index]
