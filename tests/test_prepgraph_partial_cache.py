"""Partial PrepGraph cache reuse: upstream hit + downstream miss must NOT
leave the downstream consumer with ``upstream.rows = None``.

The bug:
    Run sft_chat.yaml → tokenize cached. Bump validate's threshold → only
    validate misses, but it reads ``upstream.rows`` and sees None because
    tokenize is non-materialize (it streams rows in-memory, nothing on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data import (
    processors as _processors,  # noqa: F401 — register
)
from lighttrain.builtin_plugins.prepgraph import (
    nodes as _nodes,  # noqa: F401 — register
)
from lighttrain.prepgraph.dag import PrepGraph
from lighttrain.prepgraph.runner import PrepRunner


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


def _spec(jsonl: Path, p99_max: int) -> dict:
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
                "p99_length_max": int(p99_max),  # the only knob we'll change
            },
        ],
        "terminals": ["validated"],
    }


def test_downstream_miss_with_upstream_cache_hit_still_executes(tmp_path, jsonl_corpus):
    """The bug reproducer: re-run with only the validate threshold changed,
    expect raw/tok to *re-execute* (not just sit as inert hits) so
    validate gets rows. Pre-fix this raised
    `ValidateNode 'validated': only 0 rows, need >= 1.`
    """
    store_root = tmp_path / "store"

    # Pass 1: full run, everything misses.
    runner = PrepRunner(
        PrepGraph.from_config(_spec(jsonl_corpus, p99_max=4096)),
        store_root=store_root,
    )
    plan = runner.plan()
    assert all(not e.hit for e in plan)
    runner.run()  # writes manifests

    # Pass 2: change ONLY the validate threshold → only `validated` misses.
    runner2 = PrepRunner(
        PrepGraph.from_config(_spec(jsonl_corpus, p99_max=2048)),
        store_root=store_root,
    )
    plan2 = runner2.plan()
    by_name = {e.name: e for e in plan2}
    # After dependency analysis kicks in (during run()), raw + tok should be
    # demoted to "rerun_for_downstream" because they don't persist rows.
    assert not by_name["validated"].hit  # fingerprint really did change

    # And run() must succeed without throwing the "0 rows" error.
    results = runner2.run()
    assert "validated" in results

    val = results["validated"]
    # Pre-fix this would be 0 (validate saw upstream.rows = None and failed
    # the "need >= 1 row" guard). Post-fix it reflects the 6 corpus rows.
    report = val.extras.get("report", {})
    assert report.get("rows", 0) == 6, f"unexpected report: {report}"
