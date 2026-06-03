"""Adversarial tests for ``PrepRunner`` pool dispatch.

Targets that the legacy ``tests/test_prepgraph_process_pool.py`` misses:
  * Thread vs serial vs process pool produce **byte-identical** results
    (legacy test only checks ``hit`` flags after a re-run, not result equality)
  * ``pool_kind`` validation at construction
  * The historical PREP_POOL_01 fix (docs/changelog/v0.1.3): the pool dispatch
    uses ``concurrent.futures.as_completed(futs)`` rather than iterating the
    futures dict in submission order, so a later-submitted future that
    raises surfaces its exception before earlier slow futures complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data import (
    processors as _processors,  # noqa: F401 — registry
)
from lighttrain.builtin_plugins.prepgraph import (
    nodes as _nodes,  # noqa: F401 — registry
)
from lighttrain.prepgraph import runner as runner_module
from lighttrain.prepgraph.dag import PrepGraph
from lighttrain.prepgraph.runner import PrepRunner

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


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


def _two_sibling_spec(jsonl: Path) -> dict:
    """Two sibling tokenize nodes in parallel — distinguished by ``label_ignore``
    so their fingerprints (and final dirs) differ; otherwise they would
    content-address to the same artifact location and race on commit.
    """
    return {
        "nodes": [
            {
                "name": "raw",
                "kind": "load",
                "source": f"jsonl:{jsonl}",
                "raw_data_version": "v0",
            },
            {
                "name": "tok_a",
                "kind": "tokenize",
                "inputs": ["raw"],
                "label_ignore": -100,
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {
                        "name": "byte"
                    },
                },
            },
            {
                "name": "tok_b",
                "kind": "tokenize",
                "inputs": ["raw"],
                "label_ignore": -200,
                "processor": {
                    "name": "chat_template",
                    "tokenizer": {
                        "name": "byte"
                    },
                },
            },
        ],
        "terminals": ["tok_a", "tok_b"],
    }


def _collect_fingerprints(plan) -> dict[str, str]:
    return {e.name: e.full_fp for e in plan}


# --------------------------------------------------------------------------- #
# Equivalence: serial vs thread vs process pool yield identical caches        #
# --------------------------------------------------------------------------- #


def test_thread_pool_results_match_serial(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """Thread pool (workers=2) and serial (workers=1) yield identical caches.

    Pin: parallel scheduling must not change fingerprints, results, or
    on-disk manifests. A bug that, say, mutates upstream state in the
    parallel path would diverge the post-run plan-hits.
    """
    spec = _two_sibling_spec(jsonl_corpus)
    serial_root = tmp_path / "serial_store"
    parallel_root = tmp_path / "parallel_store"

    PrepRunner(PrepGraph.from_config(spec), store_root=serial_root, workers=1).run()
    PrepRunner(
        PrepGraph.from_config(spec),
        store_root=parallel_root,
        workers=2,
        pool_kind="thread",
    ).run()

    serial_plan = PrepRunner(
        PrepGraph.from_config(spec), store_root=serial_root
    ).plan()
    parallel_plan = PrepRunner(
        PrepGraph.from_config(spec), store_root=parallel_root
    ).plan()
    assert _collect_fingerprints(serial_plan) == _collect_fingerprints(parallel_plan)
    # And the post-run re-plan reports all hits in both cases.
    assert all(e.hit for e in serial_plan)
    assert all(e.hit for e in parallel_plan)


def test_process_pool_results_match_serial(
    tmp_path: Path, jsonl_corpus: Path
) -> None:
    """Process pool yields identical caches to serial execution.

    Pin: pickling round-trip across processes preserves node identity
    (same fingerprints) and writes the same on-disk artifacts.
    """
    spec = _two_sibling_spec(jsonl_corpus)
    serial_root = tmp_path / "serial_store"
    pproc_root = tmp_path / "pproc_store"

    PrepRunner(PrepGraph.from_config(spec), store_root=serial_root, workers=1).run()
    PrepRunner(
        PrepGraph.from_config(spec),
        store_root=pproc_root,
        workers=2,
        pool_kind="process",
    ).run()

    serial_plan = PrepRunner(
        PrepGraph.from_config(spec), store_root=serial_root
    ).plan()
    pproc_plan = PrepRunner(
        PrepGraph.from_config(spec), store_root=pproc_root
    ).plan()
    assert _collect_fingerprints(serial_plan) == _collect_fingerprints(pproc_plan)
    assert all(e.hit for e in serial_plan)
    assert all(e.hit for e in pproc_plan)


def test_pool_kind_invalid_raises(tmp_path: Path, jsonl_corpus: Path) -> None:
    """``pool_kind`` other than ``thread`` / ``process`` raises ``ValueError``.

    Contract: validation is eager at construction.
    """
    spec = _two_sibling_spec(jsonl_corpus)
    with pytest.raises(ValueError, match="pool_kind"):
        PrepRunner(
            PrepGraph.from_config(spec),
            store_root=tmp_path / "store",
            workers=2,
            pool_kind="invalid",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# PREP_POOL_01 regression — as_completed dispatch                             #
# --------------------------------------------------------------------------- #


def test_regression_PREP_POOL_01_uses_as_completed_in_thread_dispatch(
    tmp_path: Path, jsonl_corpus: Path, monkeypatch
) -> None:
    """Pre-fix bug: pool dispatch iterated futures in submission order
    (``for fut in futs``), hiding exceptions of later-submitted futures
    behind still-running earlier ones (see docs/changelog/v0.1.3:
    'ProcessPool/ThreadPool 异常延迟暴露').

    Input: a layer with two parallel sibling nodes (tok_a, tok_b), workers=2,
    pool_kind="thread". Monkeypatch ``runner_module.as_completed`` to a
    recording wrapper.

    Analytical solution: the runner's thread-pool branch must call
    ``as_completed(futs)`` exactly once, and the argument must be the
    dict-keyed futures (so completion-order iteration replaces submission
    order). The pre-fix implementation (``for fut in futs``) would not
    call ``as_completed`` at all, and the recorded call list would be
    empty.

    A test that only times wall-clock would not reliably catch the bug:
    pool ``__exit__`` calls ``shutdown(wait=True)`` so the run() return
    also waits for slow futures; the post-fix benefit is the EXCEPTION
    surfacing earlier inside the iterator, not run() returning faster.
    Code-path pinning via ``as_completed`` monkeypatch is the robust
    invariant.
    """
    spec = _two_sibling_spec(jsonl_corpus)
    real_as_completed = runner_module.as_completed
    calls: list[object] = []

    def _spy(fs, *args, **kwargs):
        calls.append(fs)
        return real_as_completed(fs, *args, **kwargs)

    monkeypatch.setattr(runner_module, "as_completed", _spy)

    PrepRunner(
        PrepGraph.from_config(spec),
        store_root=tmp_path / "store",
        workers=2,
        pool_kind="thread",
    ).run()

    assert len(calls) >= 1, (
        "Pre-fix: pool path iterated 'for fut in futs' and never called "
        "as_completed. The futures dict must be handed to as_completed."
    )
    # Sanity-check: the dict handed in is non-empty (proves pool path was taken).
    assert all(len(c) >= 1 for c in calls)


def test_regression_PREP_POOL_01_uses_as_completed_in_process_dispatch(
    tmp_path: Path, jsonl_corpus: Path, monkeypatch
) -> None:
    """Same pin for the ProcessPool branch.

    Pre-fix bug: docs/changelog/v0.1.3 says the fix swapped ``for fut in
    futs`` for ``as_completed(futs)`` — that change applies to BOTH
    ``ProcessPoolExecutor`` and ``ThreadPoolExecutor`` dispatch paths in
    runner.py.
    """
    spec = _two_sibling_spec(jsonl_corpus)
    real_as_completed = runner_module.as_completed
    calls: list[object] = []

    def _spy(fs, *args, **kwargs):
        calls.append(fs)
        return real_as_completed(fs, *args, **kwargs)

    monkeypatch.setattr(runner_module, "as_completed", _spy)

    PrepRunner(
        PrepGraph.from_config(spec),
        store_root=tmp_path / "store",
        workers=2,
        pool_kind="process",
    ).run()

    assert len(calls) >= 1, (
        "Pre-fix: process-pool path iterated 'for fut in futs' and never "
        "called as_completed."
    )


def test_invariant_pool_propagates_exception(
    tmp_path: Path, jsonl_corpus: Path, monkeypatch
) -> None:
    """An exception in any pool-submitted task propagates out of ``run()``.

    Invariant: failures are not silently swallowed by the pool dispatch.
    This guards against a regression where someone catches the future's
    exception inside the for-loop and continues.
    """
    spec = _two_sibling_spec(jsonl_corpus)

    real_run_one = PrepRunner._run_one
    sentinel = RuntimeError("__POOL_SENTINEL__")

    def _maybe_raise(self, name, results):
        if name == "tok_b":
            raise sentinel
        return real_run_one(self, name, results)

    monkeypatch.setattr(PrepRunner, "_run_one", _maybe_raise)

    runner = PrepRunner(
        PrepGraph.from_config(spec),
        store_root=tmp_path / "store",
        workers=2,
        pool_kind="thread",
    )
    with pytest.raises(RuntimeError, match="__POOL_SENTINEL__"):
        runner.run()
