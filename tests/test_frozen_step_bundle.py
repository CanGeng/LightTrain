"""Frozen Step Bundle write + read (DESIGN §18.1)."""

from __future__ import annotations

import torch

from lighttrain.diagnostics.frozen_step import (
    FrozenStepWriter,
    read_frozen_step_bundle,
)
from lighttrain.engine._context import StepContext
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM


def test_writer_commit_creates_zip(tmp_path):
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = StepContext(step=7, epoch=0)
    writer = FrozenStepWriter(tmp_path, mode="lab", run_id="run42")
    writer.snapshot(
        step=7,
        ctx=ctx,
        batch={
            "input_ids": torch.randint(0, 32, (1, 4)),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        model=model,
        optimizer=optimizer,
        config_resolved_yaml="mode: lab\n",
    )
    path = writer.commit(reason="scheduled")
    assert path is not None and path.exists()
    bundle = read_frozen_step_bundle(path)
    assert bundle.step == 7
    assert bundle.reason == "scheduled"
    assert "input_ids" in bundle.batch
    assert bundle.config_resolved_yaml.startswith("mode")
    assert bundle.model_spec.get("name") == "tiny_lm"


def test_exception_reason_is_independent_file(tmp_path):
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = StepContext(step=10, epoch=0)
    writer = FrozenStepWriter(tmp_path, mode="lab", run_id="r")
    writer.snapshot(
        step=10,
        ctx=ctx,
        batch={"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long)},
        model=model,
        optimizer=optimizer,
    )
    scheduled = writer.commit(reason="scheduled")
    exception = writer.commit(reason="exception")
    assert scheduled != exception
    assert scheduled.exists() and exception.exists()
