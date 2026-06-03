"""JEPA architecture — encoder + EMA target encoder + predictor.

Implements the I-JEPA architecture (Assran et al., 2023):

    JEPAEncoder       — processes context patches, returns embeddings
    EMATargetEncoder  — exponential-moving-average copy of JEPAEncoder;
                        produces stop-gradient target embeddings
    JEPAPredictor     — cross-attends context embeddings to target positions

The combined ``JEPAModel`` wraps all three and integrates with
``JEPAObjective`` for end-to-end training.

ArchitectureProfile is exported as ``jepa_profile()``.

Usage::

    from lighttrain.builtin_plugins.architectures.jepa import JEPAModel, JEPAModelConfig, jepa_profile

    cfg = JEPAModelConfig(patch_dim=64, embed_dim=256, num_heads=4, depth=4)
    model = JEPAModel(cfg)
    profile = jepa_profile(model)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register
from lighttrain.architectures.profile import ArchitectureProfile


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JEPAModelConfig:
    patch_dim: int = 64
    embed_dim: int = 256
    num_heads: int = 4
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    predictor_depth: int = 2


# ---------------------------------------------------------------------------
# Tiny transformer building blocks (shared by encoder and predictor)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        src = kv if kv is not None else x
        Nkv = src.shape[1]
        if kv is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.unbind(2)
        else:
            q = x @ self.qkv.weight[:C].T  # crude split
            q = q.reshape(B, N, self.num_heads, C // self.num_heads)
            kv_out = (src @ self.qkv.weight[C:].T).reshape(B, Nkv, 2, self.num_heads, C // self.num_heads)
            k, v = kv_out.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv=kv)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class JEPAEncoder(nn.Module):
    """Patch encoder for JEPA context and target branches."""

    def __init__(self, cfg: JEPAModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.patch_dim, cfg.embed_dim)
        self.blocks = nn.ModuleList([
            _Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, patch_dim)
        Returns:
            embeddings: (B, N, embed_dim)
        """
        x = self.proj(patches)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# EMA Target Encoder
# ---------------------------------------------------------------------------

class EMATargetEncoder(nn.Module):
    """EMA copy of JEPAEncoder — parameters are not trained directly."""

    def __init__(self, encoder: JEPAEncoder, momentum: float = 0.996) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.momentum = momentum
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: JEPAEncoder) -> None:
        m = self.momentum
        for pt, ps in zip(self.encoder.parameters(), source.parameters()):
            pt.data.mul_(m).add_((1.0 - m) * ps.data)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(patches)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Cross-attention predictor: context embeddings → target patch predictions."""

    def __init__(self, cfg: JEPAModelConfig) -> None:
        super().__init__()
        self.query_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.blocks = nn.ModuleList([
            _Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.predictor_depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, context_emb: torch.Tensor, target_pos_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_emb:    (B, num_context, embed_dim)  — encoder output
            target_pos_emb: (B, num_target, embed_dim)   — positional queries for target patches
        Returns:
            predictions: (B, num_target, embed_dim)
        """
        q = self.query_proj(target_pos_emb)
        for blk in self.blocks:
            q = blk(q, kv=context_emb)
        return self.norm(q)


# ---------------------------------------------------------------------------
# Combined JEPAModel
# ---------------------------------------------------------------------------

@register("model", "jepa")
class JEPAModel(nn.Module):
    """Full JEPA model: context encoder + EMA target encoder + predictor.

    Forward contract::

        batch["context_patches"]  (B, num_context, patch_dim)
        batch["target_patches"]   (B, num_target,  patch_dim)  — for target encoder
        batch.get("target_idx")   (B, num_target)              — optional positional ids

    Returns ModelOutput with::

        outputs["pred_embeddings"]    (B, num_target, embed_dim)
        extras["target_embeddings"]   (B, num_target, embed_dim)  — EMA targets
    """

    def __init__(self, cfg: JEPAModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = JEPAModelConfig(**{k: v for k, v in kwargs.items() if hasattr(JEPAModelConfig, k)})
        self.cfg = cfg
        self.encoder = JEPAEncoder(cfg)
        self.target_encoder = EMATargetEncoder(self.encoder, momentum=0.996)
        self.predictor = JEPAPredictor(cfg)
        # Positional embedding table (max 1024 patches)
        self.pos_embed = nn.Embedding(1024, cfg.embed_dim)

    def update_ema(self) -> None:
        self.target_encoder.update(self.encoder)

    def forward(self, **batch: Any) -> ModelOutput:
        ctx_patches = batch["context_patches"]          # (B, nc, patch_dim)
        tgt_patches = batch.get("target_patches", ctx_patches)  # (B, nt, patch_dim)
        tgt_idx = batch.get("target_idx")               # (B, nt) int

        # Encode context
        ctx_emb = self.encoder(ctx_patches)             # (B, nc, embed_dim)

        # Target encoder (no grad)
        tgt_emb = self.target_encoder(tgt_patches)      # (B, nt, embed_dim)

        # Positional queries for target positions
        if tgt_idx is not None:
            pos_q = self.pos_embed(tgt_idx)             # (B, nt, embed_dim)
        else:
            B, nt = tgt_patches.shape[:2]
            pos_q = self.pos_embed(
                torch.arange(nt, device=tgt_patches.device).unsqueeze(0).expand(B, -1)
            )

        pred = self.predictor(ctx_emb, pos_q)           # (B, nt, embed_dim)

        return ModelOutput(
            outputs={"pred_embeddings": pred},
            extras={"target_embeddings": tgt_emb},
        )


# ---------------------------------------------------------------------------
# ArchitectureProfile factory
# ---------------------------------------------------------------------------

def _jepa_blocks(model: nn.Module) -> Iterator[nn.Module]:
    if hasattr(model, "encoder"):
        yield from model.encoder.blocks
    if hasattr(model, "predictor"):
        yield from model.predictor.blocks


def jepa_profile(model: JEPAModel | None = None) -> ArchitectureProfile:
    return ArchitectureProfile(
        name="jepa",
        loss_family="jepa",
        state_mode="stateless",
        block_iterator_fn=_jepa_blocks,
        embedding_layer_fn=lambda m: m.encoder.proj if hasattr(m, "encoder") else None,
        head_layer_fn=lambda m: m.predictor.norm if hasattr(m, "predictor") else None,
        reset_state_fn=None,
    )


__all__ = [
    "EMATargetEncoder",
    "JEPAEncoder",
    "JEPAModel",
    "JEPAModelConfig",
    "JEPAPredictor",
    "jepa_profile",
]
