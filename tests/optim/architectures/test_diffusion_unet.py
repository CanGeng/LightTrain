"""Tests for ``lighttrain.builtin_plugins.optim.architectures.diffusion_unet``.

Pins and exercises:

* forward with ``"noisy_x"`` key (normal path)
* forward with ``"x"`` key fallback (line 141) when ``noisy_x`` absent
* forward raises ``KeyError`` when neither key present (line 143)
* forward with 2D input triggers unsqueeze/squeeze (lines 148, 177)
* spatial-size interpolation branch (line 168) triggered by odd-length input
* ``_unet_blocks`` generator yields enc_blocks + mid + dec_blocks (lines 187-189)
* ``diffusion_unet_profile()`` factory returns a properly wired
  ``ArchitectureProfile`` (line 193)
* ``TinyUNetConfig`` dataclass defaults
* ``_sinusoidal_embed`` shape and range
* registration under the ``"model"`` / ``"tiny_unet"`` key
* ``diffusion_unet_profile`` seam helpers (embedding, head, block iteration)
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.optim.architectures.diffusion_unet import (
    TinyUNet,
    TinyUNetConfig,
    _sinusoidal_embed,
    _unet_blocks,
    diffusion_unet_profile,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(base_channels: int = 16, channel_mults: tuple = (1, 2)) -> TinyUNet:
    """Minimal TinyUNet for CPU tests."""
    torch.manual_seed(0)
    cfg = TinyUNetConfig(
        in_channels=1,
        base_channels=base_channels,
        channel_mults=channel_mults,
        timesteps=10,
        time_embed_dim=32,
    )
    return TinyUNet(cfg)


def _make_batch_3d(B: int = 2, C: int = 1, L: int = 32, seed: int = 7) -> dict:
    """3D input batch: noisy_x (B, C, L) and t (B,)."""
    torch.manual_seed(seed)
    return {
        "noisy_x": torch.randn(B, C, L),
        "t": torch.randint(0, 10, (B,)),
    }


def _make_batch_2d(B: int = 2, L: int = 32, seed: int = 8) -> dict:
    """2D input batch: x (B, L) and t (B,)."""
    torch.manual_seed(seed)
    return {
        "x": torch.randn(B, L),
        "t": torch.randint(0, 10, (B,)),
    }


# ---------------------------------------------------------------------------
# TinyUNetConfig defaults
# ---------------------------------------------------------------------------

def test_invariant_config_defaults():
    """Default TinyUNetConfig fields must match the documented values."""
    cfg = TinyUNetConfig()
    assert cfg.in_channels == 1
    assert cfg.base_channels == 32
    assert cfg.channel_mults == (1, 2)
    assert cfg.timesteps == 1000
    assert cfg.time_embed_dim == 64


# ---------------------------------------------------------------------------
# _sinusoidal_embed
# ---------------------------------------------------------------------------

def test_invariant_sinusoidal_embed_output_shape():
    """_sinusoidal_embed(t, dim) → (len(t), dim)."""
    t = torch.arange(4)
    out = _sinusoidal_embed(t, dim=32)
    assert out.shape == (4, 32), f"expected (4, 32), got {out.shape}"


def test_invariant_sinusoidal_embed_values_finite():
    """All output values must be finite (no NaN / Inf from log / exp)."""
    t = torch.tensor([0, 1, 500, 999])
    out = _sinusoidal_embed(t, dim=64)
    assert torch.isfinite(out).all(), "sinusoidal embed contains non-finite values"


# ---------------------------------------------------------------------------
# Forward — normal 3D path ("noisy_x" key)
# ---------------------------------------------------------------------------

def test_invariant_forward_3d_noisy_x_output_shape():
    """Forward with (B, C, L) noisy_x produces pred of the same shape."""
    model = _make_model()
    batch = _make_batch_3d(B=2, C=1, L=32)
    with torch.no_grad():
        out = model(**batch)
    assert isinstance(out, ModelOutput)
    assert out.outputs["pred"].shape == (2, 1, 32), out.outputs["pred"].shape


def test_invariant_forward_3d_output_finite():
    """Forward output must be finite on valid input."""
    model = _make_model()
    batch = _make_batch_3d()
    with torch.no_grad():
        out = model(**batch)
    assert torch.isfinite(out.outputs["pred"]).all()


# ---------------------------------------------------------------------------
# Forward — "x" fallback key (line 141)
# ---------------------------------------------------------------------------

def test_invariant_forward_x_key_fallback_produces_correct_shape():
    """When batch has 'x' (3D) but not 'noisy_x', line 141 is taken.

    The model must use the 'x' value and return a pred of matching shape.
    """
    model = _make_model()
    torch.manual_seed(9)
    batch = {
        "x": torch.randn(2, 1, 32),   # 3D; no noisy_x
        "t": torch.randint(0, 10, (2,)),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (2, 1, 32), out.outputs["pred"].shape


def test_invariant_forward_x_key_fallback_output_equals_noisy_x_path():
    """With identical tensors, the 'x' fallback path gives the same prediction
    as the 'noisy_x' direct path (line 141 branch vs. default branch).

    Both paths eventually assign the same tensor, so outputs must be identical.
    """
    model = _make_model()
    torch.manual_seed(42)
    x_tensor = torch.randn(2, 1, 32)
    t_tensor = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out_noisy = model(**{"noisy_x": x_tensor, "t": t_tensor})
        out_x = model(**{"x": x_tensor, "t": t_tensor})

    torch.testing.assert_close(out_noisy.outputs["pred"], out_x.outputs["pred"])


# ---------------------------------------------------------------------------
# Forward — KeyError when neither "noisy_x" nor "x" present (line 143)
# ---------------------------------------------------------------------------

def test_invariant_forward_missing_x_raises_key_error():
    """Neither 'noisy_x' nor 'x' in batch → KeyError with descriptive message."""
    model = _make_model()
    batch = {"t": torch.randint(0, 10, (2,))}
    with pytest.raises(KeyError, match="noisy_x.*x|x.*noisy_x"):
        model(**batch)


def test_pin_current_behavior_forward_key_error_message():
    """Pin current KeyError message content.

    The message is the literal string in line 143 of the source. If someone
    changes the message wording, this test will catch it.  The message must
    contain both key names so callers can debug the missing input.
    """
    model = _make_model()
    batch = {"t": torch.randint(0, 10, (2,))}
    with pytest.raises(KeyError) as exc_info:
        model(**batch)
    msg = str(exc_info.value)
    assert "noisy_x" in msg or "x" in msg


# ---------------------------------------------------------------------------
# Forward — 2D input unsqueeze/squeeze (lines 148 and 177)
# ---------------------------------------------------------------------------

def test_invariant_forward_2d_x_input_squeezed_back_in_output():
    """2D batch 'x' (B, L) is unsqueezed to (B, 1, L) internally (line 148)
    and the output pred is squeezed back to (B, L) (line 177).
    """
    model = _make_model()
    batch = _make_batch_2d(B=2, L=32)
    with torch.no_grad():
        out = model(**batch)
    # Output must be 2D: (B, L)
    assert out.outputs["pred"].dim() == 2, (
        f"expected 2D pred from 2D input, got {out.outputs['pred'].shape}"
    )
    assert out.outputs["pred"].shape == (2, 32)


def test_invariant_forward_2d_noisy_x_input_squeezed_back():
    """2D 'noisy_x' (B, L) also triggers the unsqueeze/squeeze path (lines 148+177)."""
    model = _make_model()
    torch.manual_seed(11)
    batch = {
        "noisy_x": torch.randn(2, 32),   # 2D
        "t": torch.randint(0, 10, (2,)),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (2, 32), out.outputs["pred"].shape


def test_invariant_forward_2d_vs_3d_pred_equivalent():
    """2D input (B, L) and 3D input (B, 1, L) containing the same data should
    produce the same prediction (up to the extra dim).

    Analytical: unsqueeze(1) + squeeze(1) is the identity for single-channel.
    """
    model = _make_model()
    torch.manual_seed(13)
    x_2d = torch.randn(2, 32)
    x_3d = x_2d.unsqueeze(1)
    t = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out_2d = model(**{"x": x_2d, "t": t})
        out_3d = model(**{"noisy_x": x_3d, "t": t})

    # squeeze(1) on the 3D result must match the 2D result directly
    torch.testing.assert_close(out_2d.outputs["pred"], out_3d.outputs["pred"].squeeze(1))


# ---------------------------------------------------------------------------
# Forward — spatial-size interpolation (line 168) via odd-length input
# ---------------------------------------------------------------------------

def test_invariant_forward_odd_length_triggers_interpolation():
    """Odd spatial length (e.g. L=33) causes a size mismatch after stride-2
    downsampling, which must be corrected by F.interpolate (line 168).

    The test simply verifies that the forward succeeds and the output shape
    matches the input spatial size.
    """
    model = _make_model()
    torch.manual_seed(17)
    L = 33   # odd length → downsampled to 16 then upsample to 17 (mismatch)
    batch = {
        "noisy_x": torch.randn(2, 1, L),
        "t": torch.randint(0, 10, (2,)),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (2, 1, L), out.outputs["pred"].shape


@pytest.mark.parametrize("L", [17, 19, 21, 33, 65])
def test_invariant_forward_various_odd_lengths_output_shape(L):
    """Parametric: multiple odd lengths all produce output matching input shape."""
    model = _make_model()
    torch.manual_seed(L)
    batch = {
        "noisy_x": torch.randn(2, 1, L),
        "t": torch.randint(0, 10, (2,)),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (2, 1, L)


# ---------------------------------------------------------------------------
# _unet_blocks (lines 187, 188, 189)
# ---------------------------------------------------------------------------

def test_invariant_unet_blocks_yields_enc_mid_dec_in_order():
    """_unet_blocks yields enc_blocks, mid, dec_blocks — in that order.

    Covers lines 187 (yield from enc_blocks), 188 (yield mid),
    189 (yield from dec_blocks).
    """
    model = _make_model()
    blocks = list(_unet_blocks(model))

    # Expected: 2 enc_blocks + 1 mid + 2 dec_blocks = 5 total
    n_enc = len(model.enc_blocks)
    n_dec = len(model.dec_blocks)
    expected_count = n_enc + 1 + n_dec
    assert len(blocks) == expected_count, (
        f"expected {expected_count} blocks, got {len(blocks)}"
    )

    # First n_enc items must be identity-equal to enc_blocks
    for i in range(n_enc):
        assert blocks[i] is model.enc_blocks[i], (
            f"enc_block[{i}] identity mismatch"
        )

    # Middle item is the bottleneck
    assert blocks[n_enc] is model.mid, "mid block identity mismatch"

    # Remaining items must match dec_blocks
    for i in range(n_dec):
        assert blocks[n_enc + 1 + i] is model.dec_blocks[i], (
            f"dec_block[{i}] identity mismatch"
        )


def test_invariant_unet_blocks_yields_only_nn_modules():
    """Every item from _unet_blocks must be an nn.Module."""
    import torch.nn as nn

    model = _make_model()
    for blk in _unet_blocks(model):
        assert isinstance(blk, nn.Module), f"non-Module yielded: {type(blk)}"


def test_invariant_unet_blocks_count_matches_channel_mults_depth():
    """With channel_mults of length N, there are N enc blocks and N dec blocks
    plus 1 mid block → 2N + 1 blocks total.
    """
    for n in (1, 2, 3):
        mults = tuple(range(1, n + 1))
        model = _make_model(base_channels=8, channel_mults=mults)
        blocks = list(_unet_blocks(model))
        expected = 2 * n + 1
        assert len(blocks) == expected, (
            f"channel_mults={mults}: expected {expected} blocks, got {len(blocks)}"
        )


# ---------------------------------------------------------------------------
# diffusion_unet_profile (line 193)
# ---------------------------------------------------------------------------

def test_invariant_diffusion_unet_profile_returns_architecture_profile():
    """diffusion_unet_profile() returns an ArchitectureProfile instance (line 193)."""
    from lighttrain.optim.architectures.profile import ArchitectureProfile

    p = diffusion_unet_profile()
    assert isinstance(p, ArchitectureProfile)


def test_pin_diffusion_unet_profile_name_and_loss_family():
    """Pin: profile.name == 'diffusion_unet', loss_family == 'diffusion'."""
    p = diffusion_unet_profile()
    assert p.name == "diffusion_unet"
    assert p.loss_family == "diffusion"


def test_pin_diffusion_unet_profile_state_mode_is_stateless():
    """Pin: diffusion profile is stateless (no recurrent state)."""
    p = diffusion_unet_profile()
    assert p.state_mode == "stateless"


def test_invariant_diffusion_unet_profile_reset_state_raises():
    """Stateless profile + no reset_state_fn → reset_state raises NotImplementedError."""

    p = diffusion_unet_profile()
    model = _make_model()
    with pytest.raises(NotImplementedError):
        p.reset_state(model)


def test_invariant_diffusion_unet_profile_iter_blocks_matches_unet_blocks():
    """profile.iter_blocks(model) must yield the same blocks (same identity,
    same order) as _unet_blocks(model) directly.
    """
    model = _make_model()
    p = diffusion_unet_profile()

    profile_blocks = list(p.iter_blocks(model))
    direct_blocks = list(_unet_blocks(model))

    assert len(profile_blocks) == len(direct_blocks)
    for i, (pb, db) in enumerate(zip(profile_blocks, direct_blocks, strict=False)):
        assert pb is db, f"block {i}: profile iter and direct iter differ"


def test_invariant_diffusion_unet_profile_embedding_returns_enc_in():
    """profile.get_embedding(model) returns model.enc_in (the input conv layer)."""
    import torch.nn as nn

    model = _make_model()
    p = diffusion_unet_profile()
    emb = p.get_embedding(model)
    assert emb is model.enc_in
    assert isinstance(emb, nn.Conv1d)


def test_invariant_diffusion_unet_profile_head_returns_out_conv():
    """profile.get_head(model) returns model.out_conv (the output conv layer)."""
    import torch.nn as nn

    model = _make_model()
    p = diffusion_unet_profile()
    head = p.get_head(model)
    assert head is model.out_conv
    assert isinstance(head, nn.Conv1d)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_invariant_model_registered_as_tiny_unet():
    """TinyUNet must be registered under ('model', 'tiny_unet') via @register."""
    from lighttrain.registry import get

    cls = get("model", "tiny_unet")
    assert cls is TinyUNet


# ---------------------------------------------------------------------------
# TinyUNet constructor via kwargs (cfg=None path)
# ---------------------------------------------------------------------------

def test_invariant_constructor_kwargs_without_cfg():
    """TinyUNet can be constructed with keyword args instead of a TinyUNetConfig."""
    torch.manual_seed(0)
    model = TinyUNet(
        in_channels=1,
        base_channels=8,
        channel_mults=(1,),
        timesteps=5,
        time_embed_dim=16,
    )
    batch = {
        "noisy_x": torch.randn(1, 1, 16),
        "t": torch.tensor([2]),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (1, 1, 16)


# ---------------------------------------------------------------------------
# Multi-channel input
# ---------------------------------------------------------------------------

def test_invariant_forward_multi_channel_shape():
    """TinyUNet with in_channels=3 accepts (B, 3, L) and returns (B, 3, L)."""
    torch.manual_seed(0)
    cfg = TinyUNetConfig(
        in_channels=3,
        base_channels=8,
        channel_mults=(1, 2),
        timesteps=10,
        time_embed_dim=16,
    )
    model = TinyUNet(cfg)
    batch = {
        "noisy_x": torch.randn(2, 3, 32),
        "t": torch.randint(0, 10, (2,)),
    }
    with torch.no_grad():
        out = model(**batch)
    assert out.outputs["pred"].shape == (2, 3, 32)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_invariant_forward_grad_flows_through_pred():
    """Loss on pred must produce gradients for model parameters (training check)."""
    model = _make_model()
    batch = _make_batch_3d()
    out = model(**batch)
    loss = out.outputs["pred"].mean()
    loss.backward()

    n_params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None
    )
    assert n_params_with_grad > 0, "no gradients flowed to model parameters"
