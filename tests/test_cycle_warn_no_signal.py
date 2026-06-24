"""F6 — cycle detection warns when ``self_feeding`` is undeclared.

M3 already ships the ``cycle_check`` machinery; M4 just confirms that
the default ``warn`` policy emits a Python warning when an artifact
loops back into the current run and the user hasn't opted in.
"""

from __future__ import annotations

import warnings

import pytest

from lighttrain.observability.lineage.dag import apply_cycle_policy, cycle_check
from lighttrain.observability.lineage.store import LineageStore


def _seed_cycle(store: LineageStore, run_id: str) -> int:
    run_node = store.upsert_node(kind="run", name=run_id, version=run_id, run_id=run_id)
    art = store.upsert_node(
        kind="artifact",
        name="art",
        version="v1",
        run_id=run_id,
        payload_path="/tmp/x",
    )
    # produced_by: the run produced the artifact; the artifact also derived
    # from the run's earlier ckpt — closes the loop the check looks for.
    store.add_edge(run_node, art, "produced_by")
    store.add_edge(art, run_node, "derived_from")
    return art


def test_cycle_warns_by_default(tmp_path):
    store = LineageStore(tmp_path / "lineage.sqlite")
    art = _seed_cycle(store, run_id="R")
    hits = cycle_check(store, art, current_run_id="R", k=4)
    assert hits, "expected a cycle hit"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        apply_cycle_policy(hits, self_feeding="warn")
    assert any("cycle" in str(x.message).lower() or "self" in str(x.message).lower() for x in w)
    store.close()


def test_cycle_forbid_raises(tmp_path):
    store = LineageStore(tmp_path / "lineage.sqlite")
    art = _seed_cycle(store, run_id="R")
    hits = cycle_check(store, art, current_run_id="R", k=4)
    with pytest.raises(RuntimeError):
        apply_cycle_policy(hits, self_feeding="forbid")
    store.close()
