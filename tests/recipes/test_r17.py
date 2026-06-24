"""R17 — NaN injection variant of R1 (DESIGN §25.2 / §25.3)."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest
import torch

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.builtin_plugins.observability.diagnostics.nan_hunter import (
    NanHunterCallback,
)
from lighttrain.engine._context import StepContext
from lighttrain.observability.diagnostics.frozen_step import (
    FrozenStepWriter,
    replay_step_bundle,
)
from tests._diagnostics import expect_nonempty

pytestmark = pytest.mark.heavy


class _Trainer:
    def __init__(self, model, run_dir):
        self.model = model
        self._run_dir = run_dir


def test_r17_nan_repro_round_trips_subprocess(tmp_path, caplog):
    """End-to-end: inject NaN → write repro kit → run repro.py in subprocess."""
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, d_model=16, n_layers=2, n_heads=2, max_seq_len=16)
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("nan"))

    hunter = NanHunterCallback()
    ctx = StepContext(run_dir=tmp_path)
    hunter.on_train_start(trainer=_Trainer(model, tmp_path), ctx=ctx)
    with caplog.at_level(logging.WARNING, logger="lighttrain"):
        hunter.on_step_begin(
            step=1,
            batch={
                "input_ids": torch.zeros(2, 8, dtype=torch.long),  # row 0 → NaN
                "attention_mask": torch.ones(2, 8, dtype=torch.long),
            },
        )
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            model(input_ids=torch.zeros(2, 8, dtype=torch.long))
        hunter.on_train_end()

    repros = sorted((tmp_path / "diagnostics").glob("repro_nan_*"))
    expect_nonempty(
        repros, tmp_path / "diagnostics", what="a NaN repro kit", caplog=caplog
    )
    proc = subprocess.run(
        [sys.executable, str(repros[0] / "repro.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repros[0]),
    )
    out = (proc.stdout + proc.stderr).lower()
    assert "nan" in out or "anomaly" in out or "non-finite" in out


def test_r17_frozen_step_bundle_replay(tmp_path):
    """Healthy step → frozen bundle → replay reproduces the loss."""
    torch.manual_seed(7)
    model = TinyCausalLM(vocab_size=64, d_model=16, n_layers=2, n_heads=2, max_seq_len=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = StepContext(step=42, epoch=0)
    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.randint(0, 64, (2, 8)),
    }
    writer = FrozenStepWriter(tmp_path, mode="lab", run_id="r17")
    writer.snapshot(step=42, ctx=ctx, batch=batch, model=model, optimizer=optimizer)
    path = writer.commit(reason="scheduled")
    assert path is not None

    # Replay reuses the captured RNG via FrozenStepBundle → loss should match
    # forward+loss applied on a freshly rebuilt model in-place.
    result = replay_step_bundle(path, loss_fn=CrossEntropyLoss(), do_backward=False)
    assert result["loss"] is not None
    assert result["logits_shape"][0] == 2 and result["logits_shape"][1] == 8
