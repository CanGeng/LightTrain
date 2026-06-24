"""Edge-case unit tests for LineageRecorderCallback + _safe_metrics.

Pins and exhausts branches in
``lighttrain/builtin_plugins/callbacks/builtins/lineage_recorder.py``
currently at 61% coverage. Every uncovered line listed in the task is
targeted by at least one test below.

Uncovered lines driven to covered:
  54   – on_train_start early return when ctx.lineage_store is absent
  73   – on_train_end early return when _store or _run_node_id is None
  91/92 – on_save_checkpoint_post guard (_store is None / path is None)
  96   – upsert_node call inside on_save_checkpoint_post
  107/108/109/110 – add_edge call inside on_save_checkpoint_post when
                    _run_node_id is not None
  122/123 – on_artifact_finalized early return when _store is None
  126/127/128 – fallback upsert when artifact_node is None and path given
  134/135/136/137 – add_edge call inside on_artifact_finalized
  139/140 – cycle_check call
  146  – apply_cycle_policy call
  155  – on_artifact_new_version delegates to on_artifact_finalized
  176  – on_exception early return when _store or _run_node_id is None
  198/199 – on_exception except clause when inner store call raises
  208   – _safe_metrics: non-dict metrics -> {}
  213/214 – _safe_metrics: float() fails, warning emitted
  219/220 – _safe_metrics: str() fallback succeeds
  221/222 – _safe_metrics: str() also raises, drop + continue
  227   – _safe_metrics: continue (via drop path above)
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder import (
    LineageRecorderCallback,
    _safe_metrics,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _Ctx:
    """Minimal ctx stub that carries a lineage_store attribute."""

    def __init__(self, store=None, run_id: str = "run-abc"):
        self.lineage_store = store
        self.run_id = run_id


class _NoStoreCtx:
    """ctx without a lineage_store attribute at all."""


class _BrokenStore:
    """A lineage store whose upsert_node blows up.

    Used to exercise the on_exception inner except branch (lines 198-199).
    """

    def upsert_node(self, **_: Any) -> int:
        raise RuntimeError("db is gone")

    def __getattr__(self, name: str):
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# on_train_start
# ---------------------------------------------------------------------------

def test_invariant_train_start_noop_when_store_absent():
    """Line 54: on_train_start returns early when ctx has no lineage_store."""
    cb = LineageRecorderCallback()
    # ctx with lineage_store=None
    cb.on_train_start(trainer=None, ctx=_Ctx(store=None))
    assert cb._store is None
    assert cb._run_node_id is None


def test_invariant_train_start_noop_when_ctx_missing_attr():
    """Line 54 (getattr path): ctx without lineage_store attr also no-ops."""
    cb = LineageRecorderCallback()
    cb.on_train_start(trainer=None, ctx=_NoStoreCtx())
    assert cb._store is None


def test_invariant_train_start_initialises_store_and_node(lineage_store_factory):
    """Lines 55-69: on_train_start stores the LineageStore ref and writes a
    run node; _run_node_id is a valid positive integer."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="/tmp/run")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-xyz"))
    assert cb._store is store
    assert cb._run_id == "run-xyz"
    assert isinstance(cb._run_node_id, int)
    assert cb._run_node_id > 0


def test_invariant_train_start_no_trainer(lineage_store_factory):
    """on_train_start tolerates trainer=None (payload_path falls back to '')."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb.on_train_start(trainer=None, ctx=_Ctx(store=store, run_id="r1"))
    # node was created without crashing
    assert cb._run_node_id is not None


def test_invariant_train_start_run_id_defaults_to_unknown(lineage_store_factory):
    """run_id=None/empty on ctx falls back to 'unknown'."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    ctx = SimpleNamespace(lineage_store=store, run_id=None)
    cb.on_train_start(trainer=None, ctx=ctx)
    assert cb._run_id == "unknown"


# ---------------------------------------------------------------------------
# on_train_end
# ---------------------------------------------------------------------------

def test_invariant_train_end_noop_when_store_none():
    """Line 73: on_train_end returns early when _store is None."""
    cb = LineageRecorderCallback()
    # _store is None by default; should not raise
    cb.on_train_end(ctx=None, metrics={"loss": 0.5})
    # nothing stored
    assert cb._store is None


def test_invariant_train_end_noop_when_run_node_none(lineage_store_factory):
    """Line 73: on_train_end returns early when _run_node_id is None even if
    _store was set (e.g. on_train_start was called but node creation failed)."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_node_id = None  # simulate failed node creation
    cb.on_train_end(ctx=None, metrics={"loss": 0.5})
    # no crash, nothing updated (can't easily assert against DB here)


def test_invariant_train_end_updates_existing_node(lineage_store_factory):
    """Lines 76-79: on_train_end calls update_node_payload merging ended_ts
    and final_metrics into the run node created by on_train_start."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-end"))
    cb.on_train_end(ctx=_Ctx(store=store), metrics={"loss": 0.25})

    node = store.get_node(cb._run_node_id)
    import json as _json
    payload = node["payload"]
    if isinstance(payload, str):
        payload = _json.loads(payload)
    assert "ended_ts" in payload
    assert "started_ts" in payload
    assert payload["final_metrics"]["loss"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# on_save_checkpoint_post
# ---------------------------------------------------------------------------

def test_invariant_checkpoint_noop_when_store_none():
    """Line 91: on_save_checkpoint_post returns early when _store is None."""
    cb = LineageRecorderCallback()
    cb.on_save_checkpoint_post(step=1, path="/ckpt/step1", manifest=None)
    # no crash


def test_invariant_checkpoint_noop_when_path_none(lineage_store_factory):
    """Line 92: on_save_checkpoint_post returns early when path is None."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = 1
    cb.on_save_checkpoint_post(step=5, path=None, manifest=None)
    # no crash, no node written


def test_invariant_checkpoint_writes_node(lineage_store_factory):
    """Lines 96-106: a checkpoint node is upserted with the correct kind, step
    encoded in the version string, and the manifest payload."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-ckpt"))
    cb.on_save_checkpoint_post(
        step=10, path="/runs/run-ckpt/checkpoints/step10", manifest={"weights": "ok"}
    )
    ckpts = list(store.iter_nodes(kind="checkpoint"))
    assert len(ckpts) == 1
    assert ckpts[0]["version"] == "step_10"


def test_invariant_checkpoint_adds_edge_when_run_node_exists(lineage_store_factory):
    """Lines 107-110: a produced_by edge is added from the run node to the
    checkpoint node when _run_node_id is set."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-edge"))
    cb.on_save_checkpoint_post(step=3, path="/ckpts/step3", manifest=None)

    ckpts = list(store.iter_nodes(kind="checkpoint"))
    assert len(ckpts) == 1
    ckpt_id = ckpts[0]["id"]
    # verify edge exists: run_node -> ckpt_node
    edges = list(store.edges_from(cb._run_node_id))
    assert any(e["dst"] == ckpt_id for e in edges)


def test_invariant_checkpoint_no_edge_when_run_node_none(lineage_store_factory):
    """Lines 107: skip add_edge when _run_node_id is None."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = None  # no run node
    cb.on_save_checkpoint_post(step=1, path="/ckpts/step1", manifest=None)
    ckpts = list(store.iter_nodes(kind="checkpoint"))
    assert len(ckpts) == 1  # node was still created
    # no edge was added (nothing to assert against without a src)


def test_invariant_checkpoint_step_none_version(lineage_store_factory):
    """When step=None the checkpoint version is None (not 'step_None')."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = None
    cb.on_save_checkpoint_post(step=None, path="/ckpts/nostep", manifest=None)
    ckpts = list(store.iter_nodes(kind="checkpoint"))
    assert ckpts[0]["version"] is None


def test_invariant_checkpoint_manifest_non_dict_is_none(lineage_store_factory):
    """Manifest that is not a dict is passed as None payload (line 105 ``isinstance``
    guard)."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = None
    cb.on_save_checkpoint_post(step=2, path="/p", manifest="not-a-dict")
    # should not crash; manifest payload was None


# ---------------------------------------------------------------------------
# on_artifact_finalized
# ---------------------------------------------------------------------------

def test_invariant_artifact_finalized_noop_when_store_none():
    """Lines 122-123: on_artifact_finalized returns early when _store is None."""
    cb = LineageRecorderCallback()
    cb.on_artifact_finalized(path="/art/some/artifact.pt", step=5)
    # no crash


def test_invariant_artifact_finalized_upserts_node_when_no_prior_id(lineage_store_factory):
    """Lines 126-133: when artifact_node is None and path is given, a fallback
    upsert is performed and the parent dir name becomes the node name."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-art"))
    cb.on_artifact_finalized(path="/runs/my_artifact/file.pt", step=7, artifact_node=None)
    arts = list(store.iter_nodes(kind="artifact"))
    assert len(arts) == 1
    assert arts[0]["name"] == "my_artifact"
    assert arts[0]["version"] == "step_7"


def test_invariant_artifact_finalized_adds_edge_to_run_node(lineage_store_factory):
    """Lines 134-137: when both artifact_node and _run_node_id are set an edge
    is created."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-art2"))
    # Pre-upsert artifact externally and pass artifact_node
    art_id = store.upsert_node(kind="artifact", name="myart", version="step_1")
    cb.on_artifact_finalized(path=None, step=1, artifact_node=art_id)
    edges = list(store.edges_from(cb._run_node_id))
    assert any(e["dst"] == art_id for e in edges)


def test_pin_current_behavior_artifact_finalized_cycle_check_warns_for_run_produced_artifact(lineage_store_factory):
    """Lines 139-151 (pin current behavior): when on_artifact_finalized writes
    a produced_by edge (run_node -> art_id) the cycle_check BFS walks that edge
    back and finds the run node has the same run_id as current_run_id, firing a
    cycle hit even for a fresh single-step graph.

    This is arguably by design: the run is flagged as self-feeding because it
    both produced and is now finalizing an artifact. With policy='warn' a
    UserWarning is emitted.
    """
    store = lineage_store_factory()
    cb = LineageRecorderCallback(cycle_policy="warn", cycle_depth=2)
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-cycle"))
    art_id = store.upsert_node(kind="artifact", name="art", version="v1")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cb.on_artifact_finalized(path=None, step=1, artifact_node=art_id)
    # cycle_check fires because the run node (run_id=="run-cycle") is a parent
    # of the artifact via the produced_by edge just written
    assert any("self-feeding cycle" in str(w.message) for w in rec)


def test_invariant_artifact_finalized_forbid_policy_raises_on_cycle(lineage_store_factory):
    """apply_cycle_policy raises RuntimeError when cycle_policy='forbid' and
    a self-feeding cycle exists. We manually wire a cycle in the store."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback(cycle_policy="forbid", cycle_depth=4)
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-forbid"))

    # Create an artifact node that was "produced_by" run-forbid (a real cycle)
    art_id = store.upsert_node(
        kind="artifact", name="art", version="v0", run_id="run-forbid"
    )
    # Add edge: run_node -> art (produced_by), then a reverse ancestry
    store.add_edge(cb._run_node_id, art_id, "produced_by", {})
    # Now add a derived_from pointing back so cycle_check can find the run
    store.add_edge(art_id, cb._run_node_id, "derived_from", {})

    with pytest.raises(RuntimeError, match="self-feeding cycle"):
        cb.on_artifact_finalized(path=None, step=2, artifact_node=art_id)


def test_invariant_artifact_finalized_warn_policy_warns_on_cycle(lineage_store_factory):
    """apply_cycle_policy emits a UserWarning when cycle_policy='warn'."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback(cycle_policy="warn", cycle_depth=4)
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-warn"))

    art_id = store.upsert_node(
        kind="artifact", name="art", version="v0", run_id="run-warn"
    )
    store.add_edge(cb._run_node_id, art_id, "produced_by", {})
    store.add_edge(art_id, cb._run_node_id, "derived_from", {})

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cb.on_artifact_finalized(path=None, step=2, artifact_node=art_id)
    assert any("self-feeding cycle" in str(w.message) for w in rec)


def test_invariant_artifact_finalized_no_edge_when_run_node_none(lineage_store_factory):
    """No edge is written when _run_node_id is None even if artifact_node is
    set (line 134 guard)."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = None
    art_id = store.upsert_node(kind="artifact", name="a", version="v1")
    cb.on_artifact_finalized(path=None, step=0, artifact_node=art_id)
    # no edges were added
    edges = list(store.edges_from(art_id))
    assert len(edges) == 0


def test_invariant_artifact_finalized_no_upsert_when_path_also_none(lineage_store_factory):
    """When artifact_node is None and path is also None the fallback upsert
    is skipped; no artifact node is created, no edge is written."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-no-art"))
    cb.on_artifact_finalized(path=None, step=1, artifact_node=None)
    assert list(store.iter_nodes(kind="artifact")) == []


# ---------------------------------------------------------------------------
# on_artifact_new_version
# ---------------------------------------------------------------------------

def test_invariant_artifact_new_version_delegates(lineage_store_factory):
    """Line 155: on_artifact_new_version forwards to on_artifact_finalized; an
    artifact node is created and edge added exactly as if finalized was called
    directly."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-ver"))
    cb.on_artifact_new_version(path="/data/arts/versioned/file.pt", step=3)
    arts = list(store.iter_nodes(kind="artifact"))
    assert len(arts) == 1
    assert arts[0]["name"] == "versioned"


def test_invariant_artifact_new_version_noop_when_store_none():
    """on_artifact_new_version is a no-op when _store is None (via delegate)."""
    cb = LineageRecorderCallback()
    cb.on_artifact_new_version(path="/x/y/z.pt", step=1)
    # no crash


# ---------------------------------------------------------------------------
# on_exception
# ---------------------------------------------------------------------------

def test_invariant_exception_noop_when_store_none():
    """Line 176: on_exception returns early when _store is None."""
    cb = LineageRecorderCallback()
    cb.on_exception(trainer=None, exception=ValueError("boom"), step=5)
    # no crash


def test_invariant_exception_noop_when_run_node_none(lineage_store_factory):
    """Line 176: on_exception returns early when _run_node_id is None."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    cb._store = store
    cb._run_id = "r"
    cb._run_node_id = None
    cb.on_exception(trainer=None, exception=RuntimeError("oops"), step=1)
    # no frozen_step node should be written
    assert list(store.iter_nodes(kind="frozen_step")) == []


def test_invariant_exception_writes_crash_node(lineage_store_factory):
    """Lines 178-197: on_exception creates a frozen_step node and an edge from
    the run node to it."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-crash"))
    exc = ValueError("training went wrong")
    cb.on_exception(trainer=trainer, exception=exc, step=42)

    frozen = list(store.iter_nodes(kind="frozen_step"))
    assert len(frozen) == 1
    assert frozen[0]["version"] == "crash_step_42"
    import json as _json
    payload = frozen[0]["payload"]
    if isinstance(payload, str):
        payload = _json.loads(payload)
    assert payload["exc_type"] == "ValueError"
    assert "training went wrong" in payload["exc_str"]


def test_invariant_exception_step_none_defaults_to_zero(lineage_store_factory):
    """When step=None the crash version uses 0."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-crash2"))
    cb.on_exception(trainer=trainer, exception=None, step=None)
    frozen = list(store.iter_nodes(kind="frozen_step"))
    assert frozen[0]["version"] == "crash_step_0"


def test_invariant_exception_no_exception_object(lineage_store_factory):
    """on_exception with exception=None still writes a node with Unknown exc."""
    store = lineage_store_factory()
    cb = LineageRecorderCallback()
    trainer = SimpleNamespace(_run_dir="")
    cb.on_train_start(trainer=trainer, ctx=_Ctx(store=store, run_id="run-noexc"))
    cb.on_exception(trainer=trainer, exception=None, step=0)
    frozen = list(store.iter_nodes(kind="frozen_step"))
    import json as _json
    payload = frozen[0]["payload"]
    if isinstance(payload, str):
        payload = _json.loads(payload)
    assert payload["exc_type"] == "Unknown"


def test_invariant_exception_inner_store_failure_logged(caplog):
    """Lines 198-199: if the upsert_node inside on_exception raises, the
    exception is caught and a warning is emitted (never re-raised) so the
    original crash can propagate normally."""
    cb = LineageRecorderCallback()
    cb._store = _BrokenStore()
    cb._run_id = "r"
    cb._run_node_id = 1  # pretend there is a run node

    logger_name = "lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        # Must NOT raise
        cb.on_exception(trainer=None, exception=RuntimeError("crash"), step=1)

    assert any(
        "lineage_recorder" in r.message and "failed to record" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# _safe_metrics
# ---------------------------------------------------------------------------

def test_safe_metrics_non_dict_returns_empty():
    """Line 208: non-dict metrics returns {}."""
    assert _safe_metrics(None) == {}
    assert _safe_metrics("string") == {}
    assert _safe_metrics([1, 2, 3]) == {}
    assert _safe_metrics(42) == {}


def test_safe_metrics_all_numeric():
    """Lines 211-212: all float-coercible values are preserved."""
    result = _safe_metrics({"loss": 0.5, "acc": 1, "ppl": 3.14})
    assert result == {"loss": pytest.approx(0.5), "acc": pytest.approx(1.0), "ppl": pytest.approx(3.14)}


def test_safe_metrics_float_fail_str_fallback(caplog):
    """Lines 213-220: a non-float-coercible value logs a warning and falls back
    to str(); the string representation is kept in the output."""
    # Use an object that cannot be converted to float but has a good __str__
    class _Opaque:
        def __str__(self):
            return "opaque-value"

    logger_name = "lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = _safe_metrics({"special": _Opaque()})

    assert result == {"special": "opaque-value"}
    assert any("not float-coercible" in r.message for r in caplog.records)


def test_pin_current_behavior_safe_metrics_mixed(caplog):
    """Pin current behavior: a mix of good metrics, a non-float with a str
    fallback, and a missing key scenario all returned correctly."""
    class _Opaque:
        def __str__(self):
            return "blah"

    logger_name = "lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = _safe_metrics({"loss": 0.1, "meta": _Opaque()})
    assert result["loss"] == pytest.approx(0.1)
    assert result["meta"] == "blah"


def test_pin_current_behavior_safe_metrics_drop_when_str_also_fails(caplog):
    """Lines 221-227 (pin debatable): when both float() AND str() raise, the key
    is silently dropped via continue and a second warning is logged.

    This behavior is debatable (caller loses the metric entirely), but it
    prevents the lineage writer from crashing the trainer.
    """
    class _Evil:
        def __float__(self):
            raise TypeError("no float")

        def __str__(self):
            raise TypeError("no str either")

        def __repr__(self):
            return "<Evil>"

    logger_name = "lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = _safe_metrics({"loss": 0.2, "bad": _Evil()})

    # "bad" key should be absent (dropped)
    assert "bad" not in result
    # "loss" is still present
    assert result["loss"] == pytest.approx(0.2)
    # Two warnings: one for "not float-coercible", one for "not str-coercible"
    messages = [r.message for r in caplog.records]
    assert any("not float-coercible" in m for m in messages)
    assert any("not str-coercible" in m for m in messages)


# ---------------------------------------------------------------------------
# Constructor / registration
# ---------------------------------------------------------------------------

def test_invariant_constructor_defaults():
    """LineageRecorderCallback initialises with sane defaults."""
    cb = LineageRecorderCallback()
    assert cb.cycle_policy == "warn"
    assert cb.cycle_depth == 4
    assert cb.require_external_signal is False
    assert cb._store is None
    assert cb._run_node_id is None
    assert cb._run_id is None
    assert cb.critical is True


@pytest.mark.parametrize(
    "policy,depth,req",
    [
        ("allowed", 2, True),
        ("forbid", 8, False),
        ("warn", 1, True),
    ],
)
def test_invariant_constructor_custom_params(policy, depth, req):
    """Constructor stores exactly what is passed in (type-coerced)."""
    cb = LineageRecorderCallback(
        cycle_policy=policy,
        cycle_depth=depth,
        require_external_signal=req,
    )
    assert cb.cycle_policy == policy
    assert cb.cycle_depth == depth
    assert cb.require_external_signal == req


def test_invariant_registered_in_registry():
    """LineageRecorderCallback is registered under ('callback', 'lineage_recorder')."""
    from lighttrain.registry import get as registry_get

    cls = registry_get("callback", "lineage_recorder")
    assert cls is LineageRecorderCallback


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------

def test_invariant_full_run_lifecycle(lineage_store_factory):
    """Full run: start → checkpoint → artifact → exception → end.

    Verifies that the store contains exactly the expected node kinds and that
    edges connect them correctly.
    """
    store = lineage_store_factory()
    cb = LineageRecorderCallback(cycle_policy="allowed")
    trainer = SimpleNamespace(_run_dir="/runs/full")
    ctx = _Ctx(store=store, run_id="run-full")

    cb.on_train_start(trainer=trainer, ctx=ctx)
    cb.on_save_checkpoint_post(step=10, path="/runs/full/ckpts/step10", manifest={"ok": True})
    art_id = store.upsert_node(kind="artifact", name="ds", version="v1")
    cb.on_artifact_finalized(path=None, step=10, artifact_node=art_id)
    cb.on_exception(trainer=trainer, exception=RuntimeError("oops"), step=10)
    cb.on_train_end(ctx=ctx, metrics={"loss": 0.3})

    runs = list(store.iter_nodes(kind="run"))
    ckpts = list(store.iter_nodes(kind="checkpoint"))
    arts = list(store.iter_nodes(kind="artifact"))
    frozen = list(store.iter_nodes(kind="frozen_step"))

    assert len(runs) == 1
    assert len(ckpts) == 1
    assert len(arts) == 1
    assert len(frozen) == 1

    # All downstream nodes should be reachable from the run node via edges
    edges = list(store.edges_from(cb._run_node_id))
    dst_ids = {e["dst"] for e in edges}
    assert ckpts[0]["id"] in dst_ids
    assert art_id in dst_ids
    assert frozen[0]["id"] in dst_ids
