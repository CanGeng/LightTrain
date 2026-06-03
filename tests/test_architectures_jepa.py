"""Tests for JEPA architecture (M7)."""
import torch

from lighttrain.builtin_plugins.architectures.jepa import (
    EMATargetEncoder,
    JEPAEncoder,
    JEPAModelConfig,
    JEPAPredictor,
    jepa_profile,
)


def _cfg(patch_dim=16, embed_dim=32, num_heads=2, depth=2, predictor_depth=1):
    return JEPAModelConfig(
        patch_dim=patch_dim, embed_dim=embed_dim,
        num_heads=num_heads, depth=depth, predictor_depth=predictor_depth,
    )


def test_jepa_encoder_forward_shape():
    cfg = _cfg()
    enc = JEPAEncoder(cfg)
    patches = torch.randn(2, 10, 16)
    out = enc(patches)
    assert out.shape == (2, 10, 32)


def test_ema_target_encoder_no_grad():
    cfg = _cfg()
    enc = JEPAEncoder(cfg)
    target = EMATargetEncoder(enc, momentum=0.99)
    for p in target.parameters():
        assert not p.requires_grad


def test_ema_update_changes_weights():
    cfg = _cfg()
    enc = JEPAEncoder(cfg)
    target = EMATargetEncoder(enc, momentum=0.9)
    # Modify encoder weights significantly
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(1.0)
    before = [p.data.clone() for p in target.parameters()]
    target.update(enc)
    after = [p.data.clone() for p in target.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after, strict=False))


def test_ema_momentum_respected():
    """With momentum=0.0, target should fully copy source."""
    cfg = _cfg()
    enc = JEPAEncoder(cfg)
    target = EMATargetEncoder(enc, momentum=0.0)
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(5.0)
    target.update(enc)
    for p in target.parameters():
        assert torch.allclose(p, torch.full_like(p, 5.0))


def test_jepa_predictor_output_shape():
    cfg = _cfg(embed_dim=32, num_heads=2)
    predictor = JEPAPredictor(cfg)
    context = torch.randn(2, 6, 32)
    target_pos = torch.randn(2, 4, 32)
    out = predictor(context, target_pos)
    assert out.shape == (2, 4, 32)


def test_jepa_profile():
    profile = jepa_profile()
    assert profile.loss_family == "jepa"
    assert profile.state_mode == "stateless"
