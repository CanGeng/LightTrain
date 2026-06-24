"""Tests for TinyMamba architecture (M7)."""
import torch

from lighttrain.builtin_plugins.optim.architectures.mamba import (
    TinyMambaConfig,
    TinyMambaModel,
    mamba_profile,
)


def _model(vocab_size=32, d_model=16, d_state=8, num_layers=2):
    cfg = TinyMambaConfig(
        vocab_size=vocab_size, d_model=d_model,
        d_state=d_state, num_layers=num_layers,
    )
    return TinyMambaModel(cfg)


def test_mamba_forward_shape():
    model = _model()
    ids = torch.randint(0, 32, (2, 8))
    out = model(input_ids=ids)
    assert out.outputs["logits"].shape == (2, 8, 32)


def test_mamba_ssm_state_shape():
    model = _model(d_model=16, d_state=8, num_layers=2)
    ids = torch.randint(0, 32, (1, 4))
    model(input_ids=ids)
    # After forward, state should be set
    assert model._state is not None
    assert len(model._state) == 2  # num_layers


def test_mamba_state_changes_per_step():
    model = _model()
    ids1 = torch.randint(0, 32, (1, 4))
    model(input_ids=ids1)
    s1 = [s.clone() for s in model._state]

    ids2 = torch.randint(0, 32, (1, 4))
    model(input_ids=ids2)
    s2 = [s.clone() for s in model._state]

    assert any(not torch.allclose(a, b) for a, b in zip(s1, s2, strict=False))


def test_mamba_profile_stateful():
    profile = mamba_profile()
    assert profile.state_mode == "stateful"
    assert profile.loss_family == "next_token"


def test_mamba_profile_reset():
    model = _model()
    profile = mamba_profile()
    ids = torch.randint(0, 32, (1, 4))
    model(input_ids=ids)
    profile.reset_state(model)
    # reset_state fills with zeros
    assert model._state is not None
    assert all(torch.all(s == 0) for s in model._state)
