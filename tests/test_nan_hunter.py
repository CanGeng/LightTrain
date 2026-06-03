"""NanHunterCallback hooks + repro kit emission (F1 — DESIGN §18.3/§18.7)."""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.diagnostics.nan_hunter import NanHunterCallback
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.engine._context import StepContext


class _Trainer:
    def __init__(self, model, run_dir):
        self.model = model
        self._run_dir = run_dir


def test_nan_hunter_dumps_and_writes_repro(tmp_path):
    model = TinyCausalLM(vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    # Make the embeddings emit NaN by stuffing one row with NaN.
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("nan"))
    cb = NanHunterCallback()
    ctx = StepContext(run_dir=tmp_path)
    cb.on_train_start(trainer=_Trainer(model, tmp_path), ctx=ctx)
    cb.on_step_begin(
        step=1,
        batch={
            "input_ids": torch.zeros(1, 4, dtype=torch.long),  # row 0 ⇒ NaN
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    cb.on_train_end()
    diag = tmp_path / "diagnostics"
    repros = sorted(diag.glob("repro_nan_*"))
    assert len(repros) == 1, f"expected one repro kit, got {repros}"
    assert (repros[0] / "repro.py").exists()
    assert (repros[0] / "batch.pt").exists()
    assert (repros[0] / "model_state.safetensors").exists()
    nan_dumps = sorted((diag / "nan_dumps").rglob("*.pt"))
    assert nan_dumps, "expected at least one module dump"


def test_repro_py_is_under_80_lines(tmp_path):
    model = TinyCausalLM(vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("inf"))
    cb = NanHunterCallback()
    ctx = StepContext(run_dir=tmp_path)
    cb.on_train_start(trainer=_Trainer(model, tmp_path), ctx=ctx)
    cb.on_step_begin(
        step=1,
        batch={
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    with pytest.raises(RuntimeError):
        model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    cb.on_train_end()
    repros = sorted((tmp_path / "diagnostics").glob("repro_nan_*"))
    assert repros
    repro = (repros[0] / "repro.py").read_text(encoding="utf-8").splitlines()
    assert len(repro) <= 80, f"repro.py is {len(repro)} lines, DESIGN §18.3 says ≤80"
