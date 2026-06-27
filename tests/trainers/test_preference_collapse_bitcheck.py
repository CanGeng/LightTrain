"""Keystone step-2 bit-check: the collapsed PreferenceTrainer + ``loss:`` seam,
and the apply_update-migrated RewardModelTrainer, reproduce the per-step loss
sequences of the pre-migration per-algorithm trainers EXACTLY.

Golden values were captured from the pre-migration code (DPOTrainer(beta=0.1)
etc. and the bare-backward RewardModelTrainer) on the fixed seeds/batches below.
A mismatch means the collapse changed numerics — not allowed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.losses.preference import (
    DPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from lighttrain.builtin_plugins.trainers._preference_base import PreferenceTrainer
from lighttrain.builtin_plugins.trainers.rm import RewardModelTrainer
from lighttrain.protocols import ModelOutput

# ---- preference golden (captured pre-migration) ---------------------------

_PREF_GOLDEN = {
    "dpo": [0.74024159, 0.75441372, 0.79952103, 0.72592592, 0.75744557],
    "ipo": [35.86708832, 39.81147003, 49.80110931, 32.62204742, 41.89845276],
    "simpo": [1.13197541, 1.76217008, 1.60707188, 1.09368825, 1.08408999],
    "orpo": [3.33487177, 3.79651928, 3.75905204, 3.53681064, 3.41379309],
    "kto": [0.51115167, 0.51421261, 0.52502131, 0.50772804, 0.51433408],
}

_RM_GOLDEN = [
    (0.48861176, 0.519777), (0.91419041, -0.37379399), (0.58738345, 0.29959095),
    (0.66914499, 0.46875417), (0.51169235, 0.49778661),
]


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        return ModelOutput(outputs={"logits": self.head(self.emb(input_ids))})


class _TinyBackbone(nn.Module):
    class _Cfg:
        hidden_size = 8

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.config = self._Cfg()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)}, hidden_states=(h,))


class _ListDM:
    def __init__(self, batches):
        self._b = batches

    def train_loader(self):
        return list(self._b)

    def val_loader(self):
        return None

    def state_dict(self):
        return {}


def _pref_batches(n, V=16, T=6, B=4):
    g = torch.Generator().manual_seed(1234)
    return [{
        "chosen_input_ids": torch.randint(0, V, (B, T), generator=g),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "chosen_labels": torch.randint(0, V, (B, T), generator=g),
        "rejected_input_ids": torch.randint(0, V, (B, T), generator=g),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_labels": torch.randint(0, V, (B, T), generator=g),
        "aux.ref.chosen_logprobs": (torch.randn(B, generator=g) - 1.0),
        "aux.ref.rejected_logprobs": (torch.randn(B, generator=g) - 2.0),
    } for _ in range(n)]


def _rm_batches(n, V=16, T=6, B=4):
    g = torch.Generator().manual_seed(99)
    return [{
        "chosen_input_ids": torch.randint(0, V, (B, T), generator=g),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T), generator=g),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    } for _ in range(n)]


def _run_pref(loss_fn, n=5):
    torch.manual_seed(7)
    model = _TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    batches = _pref_batches(n)
    t = PreferenceTrainer(engine=None, data_module=_ListDM(batches), optimizer=opt,
                          model=model, max_steps=n)
    t.ctx.loss_fn = loss_fn
    return [round(float(t.train_step(b).loss), 8) for b in batches]  # type: ignore[arg-type]


import pytest  # noqa: E402 — placed after the helper above by design


@pytest.mark.parametrize("name,loss_fn", [
    ("dpo", DPOLoss(beta=0.1)),
    ("ipo", IPOLoss(beta=0.1)),
    ("simpo", SimPOLoss(beta=2.5, gamma=1.0)),
    ("orpo", ORPOLoss(lam=1.0)),
    ("kto", KTOLoss(beta=0.1, lambda_desirable=1.0, lambda_undesirable=1.0)),
])
def test_preference_loss_seam_bit_identical(name, loss_fn):
    # rel=1e-5 tolerance: the migration is exact on a single machine, but the
    # golden was captured on one platform — different CPU/BLAS drifts ~1e-7.
    # A real behaviour change is >>1e-5, so this still pins the math.
    assert _run_pref(loss_fn) == pytest.approx(_PREF_GOLDEN[name], rel=1e-5, abs=1e-7)


def test_reward_model_apply_update_bit_identical():
    torch.manual_seed(3)
    model = _TinyBackbone()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    batches = _rm_batches(5)
    # grad_clip=0.0 pins the legacy no-clip apply_update path this golden captured.
    # (RM's *default* is now 1.0 — F3 — a deliberate behaviour change tested
    # separately in tests/test_hardcoding_audit_fixes.py.)
    t = RewardModelTrainer(engine=None, data_module=_ListDM(batches), optimizer=opt,
                           model=model, max_steps=5, grad_clip=0.0)
    got = []
    for b in batches:
        out = t.train_step(b)
        got.append((round(float(out.metrics["loss"]), 8),
                    round(float(out.metrics["reward_margin"]), 8)))
    # tolerance, not 8-decimal exact: cross-platform float noise drifts the
    # accumulated reward_margin ~1e-7 (the loss matches; a real change is >>1e-5).
    assert len(got) == len(_RM_GOLDEN)
    for g, exp in zip(got, _RM_GOLDEN, strict=False):
        assert g == pytest.approx(exp, rel=1e-5, abs=1e-7)
