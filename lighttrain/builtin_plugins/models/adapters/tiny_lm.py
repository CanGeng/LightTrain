"""Tiny causal LM — hand-rolled pre-norm transformer (~30M params at defaults).

Used by smoke tests so they don't depend on HF model weights or network access.
Architecture follows the now-standard pre-norm
GPT-2 pattern with learned positional embeddings and tied input/output
projections.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            # Convert (B, T) -> additive mask (B, 1, 1, T) compatible with SDPA.
            pad = (1 - attention_mask.to(q.dtype)).unsqueeze(1).unsqueeze(2)
            attn_mask = pad * torch.finfo(q.dtype).min

        if return_attentions:
            # Explicit matmul path so we can return attention probabilities.
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = scores.softmax(dim=-1)
            if self.dropout and self.training:
                probs_drop = F.dropout(probs, p=self.dropout, training=True)
            else:
                probs_drop = probs
            out = torch.matmul(probs_drop, v)
            attn_weights = probs
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), attn_weights


class _MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_probs = self.attn(self.norm1(x), mask, return_attentions=return_attentions)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_probs


@register("model", "tiny_lm")
class TinyCausalLM(nn.Module):
    """A tied-embeddings pre-norm transformer."""

    def __init__(
        self,
        vocab_size: int = 260,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        tie_weights: bool = True,
        init_std: float = 0.02,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        self._default_output_hidden_states = bool(output_hidden_states)
        self._default_output_attentions = bool(output_attentions)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(self.d_model, n_heads, dropout) for _ in range(self.n_layers)]
        )
        self.norm_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(lambda m: _init_weights(m, init_std))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **_: Any,
    ) -> ModelOutput:
        if output_hidden_states is None:
            output_hidden_states = self._default_output_hidden_states
        if output_attentions is None:
            output_attentions = self._default_output_attentions
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (B, T); got shape {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        hidden_states: list[torch.Tensor] = [x] if output_hidden_states else []
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, attention_mask, return_attentions=output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions and attn is not None:
                attentions.append(attn)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return ModelOutput(
            outputs={"logits": logits},
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        num_return_sequences: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        """Minimal generate for acceptance-test rollouts (GenerativeModelProtocol).

        Returns prompt concatenated with uniform-random response tokens.
        forward() still drives log-prob computation and the policy-update loop.
        """
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        B, T = input_ids.shape
        n = min(max_new_tokens, self.max_seq_len - T)
        if n <= 0:
            return input_ids
        rand_tokens = torch.randint(
            0, self.vocab_size, (B, n),
            device=input_ids.device, dtype=input_ids.dtype,
        )
        return torch.cat([input_ids, rand_tokens], dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _init_weights(module: nn.Module, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# Suppress unused warning on math.
_ = math.pi


__all__ = ["TinyCausalLM"]
