"""Auxiliary loss tests (M6) — InfoNCELoss / MoEBalanceLoss."""

from __future__ import annotations

import pytest
import torch

from lighttrain.losses.aux import InfoNCELoss, MoEBalanceLoss
from lighttrain.protocols import LossContext, ModelOutput

_DUMMY = ModelOutput(outputs={})
_CTX = LossContext()


# ---- InfoNCE -------------------------------------------------------------

def test_info_nce_loss_positive_for_random():
    B, D = 4, 8
    batch = {
        "embeddings_anchor": torch.randn(B, D),
        "embeddings_positive": torch.randn(B, D),
    }
    out = InfoNCELoss()(_DUMMY, batch, _CTX)
    assert float(out["loss"]) > 0.0


def test_info_nce_loss_near_zero_when_identical():
    B, D = 4, 8
    z = torch.randn(B, D)
    batch = {"embeddings_anchor": z.clone(), "embeddings_positive": z.clone()}
    out = InfoNCELoss(temperature=0.01)(_DUMMY, batch, _CTX)
    assert float(out["loss"]) < 1e-3


def test_info_nce_missing_key_raises():
    with pytest.raises(KeyError):
        InfoNCELoss()(_DUMMY, {"embeddings_anchor": torch.randn(2, 4)}, _CTX)


# ---- MoEBalance ----------------------------------------------------------

def test_moe_balance_positive():
    B, T, E = 2, 4, 4
    router_probs = torch.softmax(torch.randn(B, T, E), dim=-1)
    ctx = LossContext(extras={"router_probs": router_probs})
    out = MoEBalanceLoss()(_DUMMY, {}, ctx)
    assert float(out["loss"]) > 0.0


def test_moe_balance_missing_key_raises():
    with pytest.raises(KeyError):
        MoEBalanceLoss()(_DUMMY, {}, _CTX)
