"""PrepRunner pool_kind='process' actually runs CPU-bound nodes in
subprocess workers (REVIEW #12 / DESIGN §7.7.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data import processors as _processors  # noqa: F401
from lighttrain.builtin_plugins.prepgraph import nodes  # noqa: F401
from lighttrain.data.prepgraph.dag import PrepGraph
from lighttrain.data.prepgraph.runner import PrepRunner


@pytest.fixture
def small_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "rows.jsonl"
    p.write_text(
        "\n".join(json.dumps({"messages": [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]}) for i in range(4)),
        encoding="utf-8",
    )
    return p


def _spec(jsonl_path: Path) -> dict:
    return {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl_path}",
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
                "name": "out",
                "kind": "materialize",
                "inputs": ["tok"],
                "layout": "rows",
            },
        ],
        "terminals": ["out"],
    }


def test_pool_kind_process_completes(tmp_path, small_jsonl):
    """Smoke test: pool_kind='process' produces complete fingerprinted manifests."""
    graph = PrepGraph.from_config(_spec(small_jsonl))
    runner = PrepRunner(
        graph, store_root=tmp_path / "store", workers=2, pool_kind="process"
    )
    plan_before = runner.plan()
    assert all(not e.hit for e in plan_before)
    runner.run()
    # Second run should fully cache-hit.
    plan_after = runner.plan()
    assert all(e.hit for e in plan_after)


def test_pool_kind_validates_value(tmp_path, small_jsonl):
    graph = PrepGraph.from_config(_spec(small_jsonl))
    with pytest.raises(ValueError, match="pool_kind"):
        PrepRunner(
            graph,
            store_root=tmp_path / "store",
            pool_kind="garbage",  # type: ignore[arg-type]
        )
