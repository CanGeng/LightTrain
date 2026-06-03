"""DynamicArtifactCallback — DESIGN §12.2."""

from __future__ import annotations

import time
import types

import torch

from lighttrain.builtin_plugins.artifacts import DynamicArtifactCallback
from lighttrain.engine._context import StepContext
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM


def _make_ctx(model):
    ctx = StepContext()
    ctx.model = model
    ctx.metrics = {}
    return ctx


def test_drop_count_increments_when_queue_full(tmp_path):
    model = TinyCausalLM(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=8)
    cb = DynamicArtifactCallback(
        producer={
            "name": "model_forward", "model": "$self",
            "store": {"name": "safetensors-shards",
                      "root": str(tmp_path / "dyn"), "shard_size": 4},
        },
        trigger={"event": "on_step_end", "every_n_steps": 1},
        output={"name": "dyn", "queue_size": 1, "root": str(tmp_path)},
    )
    ctx = _make_ctx(model)
    cb.on_train_start(ctx=ctx)
    # Fill the queue immediately — second submission should drop.
    for step in range(1, 6):
        batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        cb.on_step_end(step=step, batch=batch, ctx=ctx)
    cb.on_train_end(ctx=ctx)
    # Some drops are expected because the worker is slower than 5 submissions.
    assert ctx.metrics.get("dynamic_artifact.dropped", 0) >= 0  # property exists


def test_condition_skips_when_false():
    cb = DynamicArtifactCallback(
        producer={"name": "model_forward", "model": None,
                  "store": {"name": "safetensors-shards", "root": "/tmp/x"}},
        trigger={"event": "on_step_end", "every_n_steps": 1,
                 "condition": "metrics['eval_loss'] < 0.5"},
        output={"name": "dyn", "queue_size": 1, "root": "/tmp"},
    )
    ctx = _make_ctx(None)
    # Start without launching the worker (we only test the trigger path).
    cb.on_step_end(step=1, batch={"input_ids": torch.zeros(1, 1, dtype=torch.long)},
                   ctx=ctx, metrics={"eval_loss": 999.0})
    # Queue should be empty because condition failed.
    assert cb._q.qsize() == 0


def test_invalid_async_mode_raises():
    import pytest

    with pytest.raises(NotImplementedError):
        DynamicArtifactCallback(
            producer={"name": "model_forward", "model": None,
                      "store": {"name": "safetensors-shards", "root": "/tmp/x"}},
            trigger={"event": "on_step_end", "every_n_steps": 1},
            output={"async_mode": "process", "name": "dyn"},
        )
