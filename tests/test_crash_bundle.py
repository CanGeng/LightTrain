"""F2 derived — crash bundle write + lineage frozen_step node (DESIGN §20.4)."""

from __future__ import annotations

import json

import torch

from lighttrain.builtin_plugins.callbacks.builtins.lineage_recorder import (
    LineageRecorderCallback,
)
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.diagnostics.crash_bundle import write_crash_bundle
from lighttrain.engine._context import StepContext
from lighttrain.lineage.store import LineageStore


def test_crash_bundle_contents(tmp_path):
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.randint(0, 32, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    exc = RuntimeError("synthetic")
    bundle = write_crash_bundle(
        tmp_path,
        exception=exc,
        step=99,
        model=model,
        batch=batch,
        optimizer=optimizer,
        metrics={"loss": 1.23},
    )
    assert bundle.exists()
    for name in (
        "traceback.txt",
        "env.json",
        "batch.pt",
        "model_state.safetensors",
        "optimizer_state.pt",
        "rng.pt",
        "model_spec.json",
        "metrics_recent.jsonl",
    ):
        assert (bundle / name).exists(), f"missing {name}"
    # env.json mentions the exception type.
    env = json.loads((bundle / "env.json").read_text(encoding="utf-8"))
    assert env["exception_type"] == "RuntimeError"


class _StubTrainer:
    pass


def test_lineage_recorder_on_exception_writes_frozen_step(tmp_path):
    store = LineageStore(tmp_path / "lineage.sqlite")
    ctx = StepContext(run_id="run-x", lineage_store=store)
    cb = LineageRecorderCallback()
    cb.on_train_start(trainer=_StubTrainer(), ctx=ctx)
    cb.on_exception(
        trainer=_StubTrainer(),
        exception=ValueError("synthetic"),
        step=42,
    )
    # A frozen_step node should now exist with version crash_step_42.
    nodes = list(store.iter_nodes(kind="frozen_step"))
    assert any(n["version"] == "crash_step_42" for n in nodes)
    store.close()
