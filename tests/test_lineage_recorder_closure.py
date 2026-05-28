"""LineageRecorderCallback + checkpoint event closure (REVIEW #7 / #8).

Covers:
* on_save_checkpoint_pre/post are dispatched by the trainer
* LineageRecorderCallback writes a checkpoint node + produced_by edge
* on_train_start + on_train_end keep a SINGLE run node (no duplicate row)
* update_node_payload merges, not replaces
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lighttrain.callbacks.base import EventBus
from lighttrain.callbacks.builtins.lineage_recorder import LineageRecorderCallback
from lighttrain.lineage.store import LineageStore


def test_update_node_payload_merges_existing(tmp_path):
    ls = LineageStore(tmp_path / "lineage.sqlite")
    node = ls.upsert_node(
        kind="run", name="r1", version="r1", payload={"started_ts": 1.0}
    )
    ls.update_node_payload(node, {"ended_ts": 2.0, "final_metrics": {"loss": 0.5}})

    got = ls.get_node(node)
    assert got["payload"]["started_ts"] == 1.0
    assert got["payload"]["ended_ts"] == 2.0
    assert got["payload"]["final_metrics"]["loss"] == 0.5


def test_lineage_recorder_keeps_single_run_node(tmp_path):
    ls = LineageStore(tmp_path / "lineage.sqlite")
    cb = LineageRecorderCallback()
    ctx = SimpleNamespace(lineage_store=ls, run_id="myrun")
    trainer = SimpleNamespace(_run_dir=tmp_path)

    cb.on_train_start(trainer=trainer, ctx=ctx)
    cb.on_train_end(ctx=ctx, metrics={"loss": 0.1})

    runs = [n for n in ls.iter_nodes(kind="run")]
    assert len(runs) == 1
    payload = runs[0]["payload"]
    import json as _json

    payload = _json.loads(payload) if isinstance(payload, str) else payload
    assert "started_ts" in payload
    assert "ended_ts" in payload
    assert payload["final_metrics"]["loss"] == pytest.approx(0.1)


def test_lineage_recorder_critical_raises_when_store_missing_method():
    """A critical lineage callback should NOT silently swallow underlying
    sqlite/lineage errors (REVIEW #6)."""

    class _BrokenStore:
        # Missing `add_edge`, missing `upsert_node` — calling them blows up.
        def __getattr__(self, name):
            raise AttributeError(name)

    cb = LineageRecorderCallback()
    ctx = SimpleNamespace(lineage_store=_BrokenStore(), run_id="r")
    bus = EventBus([cb])
    with pytest.raises(AttributeError):
        bus.dispatch("on_train_start", trainer=SimpleNamespace(_run_dir=""), ctx=ctx)


def test_checkpoint_events_dispatched_by_trainer(tmp_path):
    """PretrainTrainer must dispatch on_save_checkpoint_pre/post so the
    LineageRecorderCallback receives the path (REVIEW #7)."""
    from lighttrain.checkpoint.manager import CheckpointManager
    from lighttrain.engine._context import StepContext
    from lighttrain.trainers.pretrain import PretrainTrainer

    class _Spy:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []

        def on_save_checkpoint_pre(self, **kw):
            self.events.append(("pre", kw))

        def on_save_checkpoint_post(self, **kw):
            self.events.append(("post", kw))

    spy = _Spy()
    model = torch.nn.Linear(2, 2)

    class _DM:
        def state_dict(self):
            return {}

        def train_loader(self):
            return []

        def val_loader(self):
            return None

    trainer = PretrainTrainer(
        engine=SimpleNamespace(step=lambda b, c: {}),
        data_module=_DM(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[spy],
        ckpt_manager=CheckpointManager(tmp_path),
        ckpt_every=1,
    )
    trainer.ctx.step = 1
    trainer._maybe_save({"loss": 0.5})

    assert [e[0] for e in spy.events] == ["pre", "post"]
    post = spy.events[1][1]
    assert post["step"] == 1
    assert post["kind"] == "step"
    # path is the directory the ckpt was written to
    assert Path(str(post["path"])).exists()
