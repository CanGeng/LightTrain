"""Tests for ArchitectureProfile and ObjectiveProfile (M7)."""
import pytest
import torch.nn as nn

from lighttrain.architectures import ArchitectureProfile
from lighttrain.builtin_plugins.architectures.transformer import transformer_profile


def _make_tiny_transformer():
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 16)
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
            self.lm_head = nn.Linear(16, 100)
        def forward(self, input_ids, **kw):
            x = self.embed_tokens(input_ids)
            for ln in self.layers:
                x = ln(x)
            return {"logits": self.lm_head(x)}
    return TinyTransformer()


def test_architecture_profile_basic():
    p = ArchitectureProfile(name="test", loss_family="next_token")
    assert p.name == "test"
    assert p.state_mode == "stateless"


def test_architecture_profile_stateful():
    p = ArchitectureProfile(name="rwkv", loss_family="next_token", state_mode="stateful")
    assert p.state_mode == "stateful"


def test_architecture_profile_reset_state_fn():
    calls = []
    p = ArchitectureProfile(
        name="test", loss_family="next_token", state_mode="stateful",
        reset_state_fn=lambda m: calls.append(1)
    )
    m = nn.Linear(4, 4)
    p.reset_state(m)
    assert len(calls) == 1


def test_architecture_profile_no_reset_raises():
    p = ArchitectureProfile(name="test", loss_family="next_token")
    m = nn.Linear(4, 4)
    with pytest.raises(NotImplementedError):
        p.reset_state(m)


def test_transformer_profile_loss_family():
    p = transformer_profile()
    assert p.loss_family == "next_token"
    assert p.state_mode == "stateless"


def test_transformer_profile_mlm():
    p = transformer_profile(loss_family="mlm")
    assert p.loss_family == "mlm"


def test_transformer_profile_block_iterator():
    model = _make_tiny_transformer()
    p = transformer_profile()
    blocks = list(p.iter_blocks(model))
    assert len(blocks) == 2


def test_transformer_profile_embedding_head():
    model = _make_tiny_transformer()
    p = transformer_profile()
    emb = p.get_embedding(model)
    head = p.get_head(model)
    assert isinstance(emb, nn.Embedding)
    assert isinstance(head, nn.Linear)
