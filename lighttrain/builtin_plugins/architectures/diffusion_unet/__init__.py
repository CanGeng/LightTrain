"""TinyUNet — simple 1D/2D diffusion U-Net for CPU demo.

A toy U-Net architecture suitable for verifying the DiffusionObjective +
ArchitectureProfile integration (R8).  Supports 1D signals (default) and
can be extended for 2D images by swapping Conv1d → Conv2d.

Model contract (aligns with DiffusionObjective):
    Input:  batch["noisy_x"] — (B, C, L)  noisy signal
            batch["t"]       — (B,)        timestep indices
    Output: ModelOutput.outputs["pred"] — (B, C, L)  eps / x0 / v prediction

Registered as ``@register("model", "tiny_unet")``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.architectures.profile import ArchitectureProfile
from lighttrain.protocols import ModelOutput
from lighttrain.registry import register

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TinyUNetConfig:
    in_channels: int = 1
    base_channels: int = 32
    channel_mults: tuple = (1, 2)      # depth of U-Net
    timesteps: int = 1000
    time_embed_dim: int = 64


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps t (B,) to sinusoidal embeddings (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# TinyUNet
# ---------------------------------------------------------------------------

@register("model", "tiny_unet")
class TinyUNet(nn.Module):
    """Toy 1D U-Net for diffusion model smoke tests.

    Accepts ``batch["noisy_x"]`` (B, C, L) and ``batch["t"]`` (B,).
    Returns ModelOutput with outputs["pred"] of the same shape.
    """

    def __init__(self, cfg: TinyUNetConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = TinyUNetConfig(**{k: v for k, v in kwargs.items() if hasattr(TinyUNetConfig, k)})
        self.cfg = cfg
        T = cfg.time_embed_dim
        C = cfg.base_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(T, T * 4),
            nn.SiLU(),
            nn.Linear(T * 4, T),
        )

        # Encoder path
        self.enc_in = nn.Conv1d(cfg.in_channels, C, 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = C
        enc_chs = [ch]
        for mult in cfg.channel_mults:
            out_ch = C * mult
            self.enc_blocks.append(ResBlock(ch, out_ch, T))
            enc_chs.append(out_ch)
            self.downs.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        # Bottleneck
        self.mid = ResBlock(ch, ch, T)

        # Decoder path
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for mult in reversed(cfg.channel_mults):
            skip_ch = enc_chs.pop()
            out_ch = C * mult
            self.ups.append(nn.ConvTranspose1d(ch, out_ch, 2, stride=2))
            self.dec_blocks.append(ResBlock(out_ch + skip_ch, out_ch, T))
            ch = out_ch

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv1d(ch, cfg.in_channels, 3, padding=1)

        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.dec_blocks = nn.ModuleList(self.dec_blocks)

    def forward(self, **batch: Any) -> ModelOutput:
        x = batch.get("noisy_x")
        if x is None:
            x = batch.get("x")
        if x is None:
            raise KeyError("TinyUNet expects 'noisy_x' or 'x' in batch.")
        t = batch["t"]   # (B,)

        # Ensure 3D input (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        t_emb = _sinusoidal_embed(t, self.cfg.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.enc_in(x)
        skips = [h]

        for enc_block, down in zip(self.enc_blocks, self.downs, strict=False):
            h = enc_block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, t_emb)

        for up, dec_block in zip(self.ups, self.dec_blocks, strict=False):
            h = up(h)
            skip = skips.pop()
            # Align sizes if pooling introduced mismatch
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = dec_block(h, t_emb)

        h = F.silu(self.out_norm(h))
        pred = self.out_conv(h)

        # Match input shape
        if batch.get("noisy_x", batch.get("x")).dim() == 2:
            pred = pred.squeeze(1)

        return ModelOutput(outputs={"pred": pred})


# ---------------------------------------------------------------------------
# ArchitectureProfile
# ---------------------------------------------------------------------------

def _unet_blocks(model: nn.Module) -> Iterator[nn.Module]:
    yield from model.enc_blocks
    yield model.mid
    yield from model.dec_blocks


def diffusion_unet_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        name="diffusion_unet",
        loss_family="diffusion",
        state_mode="stateless",
        block_iterator_fn=_unet_blocks,
        embedding_layer_fn=lambda m: m.enc_in,
        head_layer_fn=lambda m: m.out_conv,
        reset_state_fn=None,
    )


__all__ = ["TinyUNet", "TinyUNetConfig", "diffusion_unet_profile"]
