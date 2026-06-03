"""ReferencePolicy tests (M6) — freeze_as_ref / ref_log_probs."""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.rl.ref_policy import (
    freeze_as_ref,
    ref_log_probs,
)


class _TinyModel(nn.Module):
    def __init__(self, V: int = 16, D: int = 8, T: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> dict:
        x = self.emb(input_ids)
        logits = self.proj(x)
        return {"logits": logits}


def test_freeze_as_ref_no_grad():
    model = _TinyModel()
    ref = freeze_as_ref(model)
    assert ref.model is not None
    for p in ref.model.parameters():
        assert not p.requires_grad


def test_freeze_as_ref_deep_copy():
    model = _TinyModel()
    ref = freeze_as_ref(model)
    assert ref.model is not model
    # Modifying original should not affect ref
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)
    for p_orig, p_ref in zip(model.parameters(), ref.model.parameters(), strict=False):
        assert not torch.allclose(p_orig, p_ref)


def test_ref_log_probs_shape():
    B, T, V = 2, 4, 16
    model = _TinyModel(V=V)
    ref = freeze_as_ref(model)
    input_ids = torch.randint(0, V, (B, T))
    labels = input_ids.clone()
    lp = ref_log_probs(ref, input_ids, None, labels)
    assert lp.shape == (B,)


def test_ref_log_probs_finite():
    B, T, V = 2, 5, 16
    model = _TinyModel(V=V)
    ref = freeze_as_ref(model)
    input_ids = torch.randint(0, V, (B, T))
    labels = input_ids.clone()
    lp = ref_log_probs(ref, input_ids, None, labels)
    assert torch.isfinite(lp).all()
