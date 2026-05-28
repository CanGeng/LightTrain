"""Cycle detection + policy — DESIGN §12.7."""

from __future__ import annotations

import warnings

import pytest

from lighttrain.lineage import LineageStore, cycle_check
from lighttrain.lineage.dag import apply_cycle_policy


def _build_loop(store: LineageStore, run_id: str = "R1"):
    run = store.upsert_node(kind="run", name=run_id, version="1", run_id=run_id)
    a1 = store.upsert_node(kind="artifact", name="A", version="v1",
                           run_id=run_id, schema_kind="artifact_header")
    a2 = store.upsert_node(kind="artifact", name="A", version="v2",
                           run_id=run_id, schema_kind="artifact_header")
    store.add_edge(run, a1, "produced_by")
    store.add_edge(a2, a1, "derived_from")
    store.add_edge(run, a2, "produced_by")
    return run, a1, a2


def test_cycle_check_finds_self_feeding(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    _, _, a2 = _build_loop(store, run_id="R1")
    hits = cycle_check(store, a2, current_run_id="R1", k=4)
    # a2's ancestors include a1 (run_id=R1) and the run itself.
    assert hits, "expected at least one cycle hit"
    assert all(h.via_run_id == "R1" for h in hits)


def test_cycle_check_no_hit_on_external_run(tmp_path):
    store = LineageStore(tmp_path / "l.sqlite")
    _, _, a2 = _build_loop(store, run_id="R1")
    hits = cycle_check(store, a2, current_run_id="R_other", k=4)
    assert hits == []


def test_apply_cycle_policy_warn_emits_warning():
    from lighttrain.lineage.dag import CycleHit

    hits = [CycleHit(node_id=1, via_run_id="R1", depth=1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy(hits, self_feeding="warn")
    assert any("self-feeding" in str(w.message) for w in caught)


def test_apply_cycle_policy_forbid_raises():
    from lighttrain.lineage.dag import CycleHit

    hits = [CycleHit(node_id=1, via_run_id="R1", depth=1)]
    with pytest.raises(RuntimeError):
        apply_cycle_policy(hits, self_feeding="forbid")


def test_apply_cycle_policy_allowed_is_silent():
    from lighttrain.lineage.dag import CycleHit

    hits = [CycleHit(node_id=1, via_run_id="R1", depth=1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_cycle_policy(hits, self_feeding="allowed")
    assert not any("self-feeding" in str(w.message) for w in caught)
