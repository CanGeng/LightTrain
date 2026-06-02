"""TinyRWKVModel — CPU-compatible RWKV-6 demo.

Implements a simplified RWKV recurrent language model in pure PyTorch.
This demo version uses loops (not CUDA-optimised kernels) and is designed
to verify the ArchitectureProfile + stateful training integration (R7).

Key RWKV concepts implemented here:
    * Token Shift  — linear interpolation with previous token embedding
    * WKV state    — running key-value accumulation (attention-free)
    * Channel Mix  — FFN with receptance gating

State tuple per layer: (shift_state, wkv_state)
    shift_state: (B, C)     — last token embedding
    wkv_state:   (B, C, 1)  — running sum state for wkv

Cross-document reset:  call ``model.reset_state()`` or set ``batch["_reset_state"] = True``.

ArchitectureProfile exported as ``rwkv_profile()``.
Registered as ``@register("model", "tiny_rwkv")``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class TinyRWKVConfig:
    vocab_size: int = 256
    embed_dim: int = 64
    num_layers: int = 2
    chunk_size: int = 64  # sequence length per forward chunk


# ---------------------------------------------------------------------------
# RWKV Block
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """Single RWKV-6 block (simplified time-mixing + channel-mixing)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        # Time-mixing params
        self.mix_k = nn.Parameter(torch.zeros(dim))
        self.mix_v = nn.Parameter(torch.zeros(dim))
        self.mix_r = nn.Parameter(torch.zeros(dim))
        self.time_decay = nn.Parameter(torch.zeros(dim) - 4.0)  # log(-w)
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        # Channel-mixing
        self.mix_k2 = nn.Parameter(torch.zeros(dim))
        self.mix_r2 = nn.Parameter(torch.zeros(dim))
        self.W_r2 = nn.Linear(dim, dim, bias=False)
        self.W_k2 = nn.Linear(dim, int(dim * 3.5), bias=False)
        self.W_v2 = nn.Linear(int(dim * 3.5), dim, bias=False)

    def time_mix(
        self,
        x: torch.Tensor,       # (B, T, C)
        state: torch.Tensor,   # (B, C) — last token state
        wkv_state: torch.Tensor,  # (B, C, 1) — running accumulator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        out = torch.zeros_like(x)
        new_state = state
        new_wkv = wkv_state

        w = -torch.exp(self.time_decay)  # (C,) decay

        for t in range(T):
            xt = x[:, t, :]                     # (B, C)
            xk = xt * torch.sigmoid(self.mix_k) + new_state * (1 - torch.sigmoid(self.mix_k))
            xv = xt * torch.sigmoid(self.mix_v) + new_state * (1 - torch.sigmoid(self.mix_v))
            xr = xt * torch.sigmoid(self.mix_r) + new_state * (1 - torch.sigmoid(self.mix_r))
            new_state = xt

            r = torch.sigmoid(self.W_r(xr))     # (B, C)
            k = self.W_k(xk)                    # (B, C)
            v = self.W_v(xv)                    # (B, C)

            # WKV accumulation: numerically stable approximation
            kv = k.unsqueeze(-1) * v.unsqueeze(-1)   # (B, C, 1) approx
            new_wkv = new_wkv * w.unsqueeze(0).unsqueeze(-1).exp() + kv

            wkv_out = (new_wkv / (new_wkv.abs().max(dim=1, keepdim=True).values + 1e-8)).squeeze(-1)
            yt = r * wkv_out
            out[:, t, :] = self.W_o(yt)

        return out, new_state, new_wkv

    def channel_mix(
        self,
        x: torch.Tensor,       # (B, T, C)
        state: torch.Tensor,   # (B, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        out = torch.zeros_like(x)
        new_state = state
        for t in range(T):
            xt = x[:, t, :]
            xk = xt * torch.sigmoid(self.mix_k2) + new_state * (1 - torch.sigmoid(self.mix_k2))
            xr = xt * torch.sigmoid(self.mix_r2) + new_state * (1 - torch.sigmoid(self.mix_r2))
            new_state = xt
            r = torch.sigmoid(self.W_r2(xr))
            k = torch.relu(self.W_k2(xk)) ** 2
            out[:, t, :] = r * self.W_v2(k)
        return out, new_state

    def forward(
        self,
        x: torch.Tensor,
        time_state: torch.Tensor,
        wkv_state: torch.Tensor,
        cm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Time-mixing
        tm_out, new_time_state, new_wkv = self.time_mix(self.ln1(x), time_state, wkv_state)
        x = x + tm_out
        # Channel-mixing
        cm_out, new_cm_state = self.channel_mix(self.ln2(x), cm_state)
        x = x + cm_out
        return x, new_time_state, new_wkv, new_cm_state


# ---------------------------------------------------------------------------
# TinyRWKVModel
# ---------------------------------------------------------------------------

@register("model", "tiny_rwkv")
class TinyRWKVModel(nn.Module):
    """Tiny RWKV language model — stateful (RWKV-6 simplified).

    State per layer stored in ``ModelOutput.state`` as a list of 3-tuples:
        [(time_state, wkv_state, cm_state), ...]  — one per layer

    Reset state between documents by calling ``.reset_state(batch_size, device)``
    or setting ``batch["_reset_state"] = True``.
    """

    def __init__(self, cfg: TinyRWKVConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = TinyRWKVConfig(**{k: v for k, v in kwargs.items() if hasattr(TinyRWKVConfig, k)})
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList([RWKVBlock(cfg.embed_dim) for _ in range(cfg.num_layers)])
        self.ln_out = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self._state: list | None = None

    def reset_state(self, batch_size: int = 1, device: Any = "cpu") -> None:
        C = self.cfg.embed_dim
        self._state = [
            (
                torch.zeros(batch_size, C, device=device),
                torch.zeros(batch_size, C, 1, device=device),
                torch.zeros(batch_size, C, device=device),
            )
            for _ in range(self.cfg.num_layers)
        ]

    def forward(self, **batch: Any) -> ModelOutput:
        input_ids: torch.Tensor = batch["input_ids"]
        B, T = input_ids.shape
        device = input_ids.device

        # Carry over state from previous chunk, or reset
        reset = batch.get("_reset_state", False)
        prev_state = batch.get("_arch_state")
        if reset or prev_state is None or self._state is None:
            self.reset_state(B, device)
        else:
            self._state = prev_state

        # Ensure state batch size matches current input
        if self._state[0][0].shape[0] != B:
            self.reset_state(B, device)

        x = self.embed(input_ids)   # (B, T, C)

        new_state = []
        for i, block in enumerate(self.blocks):
            ts, ws, cs = self._state[i]
            x, ts, ws, cs = block(x, ts.to(device), ws.to(device), cs.to(device))
            new_state.append((ts.detach(), ws.detach(), cs.detach()))

        self._state = new_state

        x = self.ln_out(x)
        logits = self.head(x)   # (B, T, vocab_size)

        return ModelOutput(
            outputs={"logits": logits},
            state=new_state,
        )


# ---------------------------------------------------------------------------
# ArchitectureProfile
# ---------------------------------------------------------------------------

def _rwkv_blocks(model: nn.Module) -> Iterator[nn.Module]:
    yield from model.blocks


@register("architecture", "rwkv")
def rwkv_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        name="rwkv",
        loss_family="next_token",
        state_mode="stateful",
        block_iterator_fn=_rwkv_blocks,
        embedding_layer_fn=lambda m: m.embed,
        head_layer_fn=lambda m: m.head,
        reset_state_fn=lambda m: m.reset_state(1),
    )


__all__ = ["TinyRWKVConfig", "TinyRWKVModel", "rwkv_profile"]
