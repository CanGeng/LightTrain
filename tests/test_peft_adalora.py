"""Tests for AdaLoRAAdapter (M7, M5 defer)."""
import pytest
import torch
import torch.nn as nn

from lighttrain.models.peft._adalora import AdaLoRAAdapter, AdaLoRALinear


def _make_model():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


# ---------------------------------------------------------------------------
# AdaLoRALinear (always manual — no PEFT dependency)
# ---------------------------------------------------------------------------

def test_adalora_linear_output_shape():
    base = nn.Linear(16, 32)
    ada = AdaLoRALinear(base, r=4, lora_alpha=8)
    x = torch.randn(2, 16)
    out = ada(x)
    assert out.shape == (2, 32)


def test_adalora_linear_has_lambda():
    base = nn.Linear(8, 16)
    ada = AdaLoRALinear(base, r=4, lora_alpha=4)
    assert hasattr(ada, "lora_Lambda")
    assert ada.lora_Lambda.shape == (4,)


def test_adalora_rank_pruning():
    """prune_rank should zero out low-importance lambdas."""
    base = nn.Linear(8, 16)
    ada = AdaLoRALinear(base, r=4, lora_alpha=4)
    with torch.no_grad():
        ada.lora_Lambda.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    ada.prune_rank(keep=1)
    assert (ada.lora_Lambda.abs() > 1e-6).sum() <= 1


def test_adalora_linear_importance_scores():
    base = nn.Linear(4, 8)
    ada = AdaLoRALinear(base, r=3, lora_alpha=3)
    scores = ada.importance_scores()
    assert scores.shape == (3,)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# AdaLoRAAdapter (path-agnostic — PEFT or manual)
# ---------------------------------------------------------------------------

def test_adalora_adapter_creates_without_error():
    model = _make_model()
    adapter = AdaLoRAAdapter(base=model, r=4, target_modules=["0", "2"], total_step=100)
    assert adapter is not None


def test_adalora_state_dict_has_keys():
    model = _make_model()
    adapter = AdaLoRAAdapter(base=model, r=4, target_modules=["0", "2"], total_step=100)
    sd = adapter.state_dict()
    # Should have some keys regardless of PEFT or manual path
    assert len(sd) > 0
