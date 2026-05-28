"""Shared preference trainer pipeline tests (M6) — DPO / IPO / SimPO / ORPO / KTO."""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.trainers._preference_base import PreferenceTrainer, _seq_logps_and_nll
from lighttrain.trainers.dpo import DPOTrainer
from lighttrain.trainers.ipo import IPOTrainer
from lighttrain.trainers.kto import KTOTrainer
from lighttrain.trainers.orpo import ORPOTrainer
from lighttrain.trainers.simpo import SimPOTrainer


# ---- Minimal helpers -------------------------------------------------------

class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _FakeDataModule:
    def __init__(self, V: int = 16, T: int = 5, B: int = 2) -> None:
        self.V, self.T, self.B = V, T, B

    def train_loader(self):
        V, T, B = self.V, self.T, self.B
        while True:
            yield {
                "chosen_input_ids": torch.randint(0, V, (B, T)),
                "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
                "chosen_labels": torch.randint(0, V, (B, T)),
                "rejected_input_ids": torch.randint(0, V, (B, T)),
                "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
                "rejected_labels": torch.randint(0, V, (B, T)),
                # Injected artifact ref logprobs (normally from ArtifactJoinedDataset)
                "aux.ref.chosen_logprobs": torch.randn(B) - 1.0,
                "aux.ref.rejected_logprobs": torch.randn(B) - 2.0,
            }


class _FakeEngine:
    mixed_precision = "no"


def _make(cls, **kw):
    V = 16
    model = _TinyLM(V=V)
    dm = _FakeDataModule(V=V)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return cls(engine=_FakeEngine(), data_module=dm, optimizer=opt, model=model, max_steps=3, **kw)


# ---- _seq_logps_and_nll ---------------------------------------------------

def test_seq_logps_shape():
    B, T, V = 2, 5, 16
    model = _TinyLM(V=V)
    ids = torch.randint(0, V, (B, T))
    labels = ids.clone()
    logps, nll = _seq_logps_and_nll(model, ids, None, labels)
    assert logps.shape == (B,)
    assert nll.shape == (B,)


def test_seq_logps_finite():
    B, T, V = 3, 4, 16
    model = _TinyLM(V=V)
    ids = torch.randint(0, V, (B, T))
    labels = ids.clone()
    logps, _ = _seq_logps_and_nll(model, ids, None, labels)
    assert torch.isfinite(logps).all()


def test_seq_nll_nonneg():
    B, T, V = 2, 4, 16
    model = _TinyLM(V=V)
    ids = torch.randint(0, V, (B, T))
    _, nll = _seq_logps_and_nll(model, ids, None, ids.clone())
    assert (nll >= 0).all()


# ---- DPO -----------------------------------------------------------------

def test_dpo_trainer_registers():
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "dpo") is DPOTrainer


def test_dpo_preference_step_returns_loss():
    t = _make(DPOTrainer, beta=0.1)
    V, T, B = 16, 5, 2
    batch = next(t.data_module.train_loader())
    metrics = t._preference_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] > 0.0


# ---- IPO -----------------------------------------------------------------

def test_ipo_preference_step_runs():
    t = _make(IPOTrainer, beta=0.1)
    batch = next(t.data_module.train_loader())
    metrics = t._preference_step(batch)
    assert "loss" in metrics


# ---- SimPO (no ref needed) -----------------------------------------------

def test_simpo_no_ref_logprobs_needed():
    t = _make(SimPOTrainer, beta=2.5, gamma=1.0)
    V, T, B = 16, 5, 2
    # Batch without aux ref keys — SimPO should still work
    dm_no_ref = _FakeDataModule(V=V, T=T, B=B)
    batch = next(dm_no_ref.train_loader())
    batch_stripped = {k: v for k, v in batch.items() if not k.startswith("aux.")}
    metrics = t._preference_step(batch_stripped)
    assert "loss" in metrics


# ---- ORPO ----------------------------------------------------------------

def test_orpo_returns_sft_loss():
    t = _make(ORPOTrainer, lam=0.1)
    batch = next(t.data_module.train_loader())
    metrics = t._preference_step(batch)
    assert "loss" in metrics


# ---- KTO -----------------------------------------------------------------

def test_kto_preference_step_runs():
    t = _make(KTOTrainer, beta=0.1)
    batch = next(t.data_module.train_loader())
    metrics = t._preference_step(batch)
    assert "loss" in metrics


# ---- callback wiring fix (bug fix verification) ----------------------------

def test_preference_step_fires_full_callback_chain():
    """_preference_step must dispatch on_step_begin/end and on_clip_grad."""
    fired = []

    class _Recorder:
        def on_step_begin(self, **kw): fired.append("on_step_begin")
        def on_backward_pre(self, **kw): fired.append("on_backward_pre")
        def on_backward_post(self, **kw): fired.append("on_backward_post")
        def on_clip_grad(self, **kw): fired.append("on_clip_grad")
        def on_optimizer_step_pre(self, **kw): fired.append("on_optimizer_step_pre")
        def on_optimizer_step_post(self, **kw): fired.append("on_optimizer_step_post")
        def on_zero_grad(self, **kw): fired.append("on_zero_grad")
        def on_step_end(self, **kw): fired.append("on_step_end")

    t = _make(DPOTrainer, callbacks=[_Recorder()])
    batch = next(t.data_module.train_loader())
    t._preference_step(batch)

    for event in [
        "on_step_begin", "on_backward_pre", "on_backward_post",
        "on_clip_grad", "on_optimizer_step_pre", "on_optimizer_step_post",
        "on_zero_grad", "on_step_end",
    ]:
        assert event in fired, f"{event} not fired by _preference_step"
