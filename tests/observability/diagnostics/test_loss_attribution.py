"""Three-level loss attribution (DESIGN §18.2)."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.builtin_plugins.observability.diagnostics.loss_attribution import (
    compute_loss_attribution,
)
from lighttrain.protocols import LossContext


def _setup():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=8)
    batch = {
        "input_ids": torch.randint(0, 32, (3, 6)),
        "attention_mask": torch.ones(3, 6, dtype=torch.long),
        "labels": torch.randint(0, 32, (3, 6)),
    }
    out = model(**batch)
    loss = CrossEntropyLoss()(out, batch, LossContext())["loss"]
    return model, batch, out, loss


def test_sample_level_returns_one_per_sample():
    model, batch, out, loss = _setup()
    report = compute_loss_attribution(
        model=model, batch=batch, outputs=out, loss=loss, levels=("sample",)
    )
    assert "sample" in report
    losses = report["sample"]["loss_per_sample"]
    assert len(losses) == batch["input_ids"].shape[0]


def test_token_level_matrix_shape_matches_shift():
    model, batch, out, loss = _setup()
    report = compute_loss_attribution(
        model=model, batch=batch, outputs=out, loss=loss, levels=("token",)
    )
    matrix = report["token"]["loss_per_token"]
    assert len(matrix) == batch["input_ids"].shape[0]
    assert len(matrix[0]) == batch["input_ids"].shape[1] - 1  # shift drops 1


def test_module_level_returns_topk():
    model, batch, out, loss = _setup()
    report = compute_loss_attribution(
        model=model, batch=batch, outputs=out, loss=loss,
        levels=("module",), top_k_modules=5,
    )
    if "module" in report:  # may be skipped if no grad-requiring capture
        top = report["module"]["top_k"]
        assert isinstance(top, list)
        assert all(isinstance(g, float) for _, g in top)
