"""TinyMambaModel — CPU-compatible Mamba SSM demo.

Implements a simplified Mamba-style selective state-space model in pure PyTorch
(no CUDA-optimised selective scan kernel — uses loops for correctness).

Mamba core (Gu & Dao 2023):
    * Input-dependent selection of A, B, C matrices
    * Continuous-time SSM discretised with ZOH:
        h_t = Ā·h_{t-1} + B̄·x_t
        y_t = C·h_t + D·x_t
    * Convolution short branch (causal 1D conv)

State tuple per layer: h_t — shape (B, D_state, D_inner)
Cross-document reset: batch["_reset_state"] = True or model.reset_state().

ArchitectureProfile exported as ``mamba_profile()``.
Registered as ``@register("model", "tiny_mamba")``.
"""

from __future__ import annotations

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
class TinyMambaConfig:
    vocab_size: int = 256
    d_model: int = 64       # model dimension
    d_state: int = 16       # SSM state dimension N
    d_conv: int = 4         # local convolution width
    expand: int = 2         # inner expansion factor
    num_layers: int = 2


# ---------------------------------------------------------------------------
# Mamba SSM block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Simplified Mamba block (selective SSM + short convolution)."""

    def __init__(self, cfg: TinyMambaConfig) -> None:
        super().__init__()
        D = cfg.d_model
        D_inner = D * cfg.expand
        N = cfg.d_state

        self.norm = nn.LayerNorm(D)
        self.in_proj = nn.Linear(D, D_inner * 2, bias=False)   # x and z branches
        self.conv1d = nn.Conv1d(D_inner, D_inner, cfg.d_conv, padding=cfg.d_conv - 1, groups=D_inner)
        self.x_proj = nn.Linear(D_inner, N + N + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, D_inner, bias=True)

        # SSM parameters
        log_A = torch.log(torch.arange(1, N + 1, dtype=torch.float32)).unsqueeze(0).expand(D_inner, -1)
        self.A_log = nn.Parameter(log_A)
        self.D = nn.Parameter(torch.ones(D_inner))

        self.out_proj = nn.Linear(D_inner, D, bias=False)

        self._d_state = N
        self._d_inner = D_inner
        self._d_conv = cfg.d_conv

    def ssm_step(
        self,
        x: torch.Tensor,       # (B, D_inner)
        h: torch.Tensor,       # (B, D_inner, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._d_state

        # Compute time-varying B, C, dt
        bcd = self.x_proj(x)                        # (B, N+N+1)
        B_ssm = bcd[:, :N]                          # (B, N)
        C_ssm = bcd[:, N: 2 * N]                    # (B, N)
        dt_raw = bcd[:, 2 * N:]                     # (B, 1)
        dt = F.softplus(self.dt_proj(dt_raw))       # (B, D_inner)

        # Discretise A: Ā = exp(Δ * A)
        A = -self.A_log.exp()                       # (D_inner, N)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, D_inner, N)

        # Discretise B: B̄ = Δ * B  (simplified ZOH)
        B_bar = dt.unsqueeze(-1) * B_ssm.unsqueeze(1)  # (B, D_inner, N)

        # SSM step
        h_new = A_bar * h + B_bar * x.unsqueeze(-1)    # (B, D_inner, N)

        # Output
        y = (h_new * C_ssm.unsqueeze(1)).sum(-1) + self.D * x  # (B, D_inner)
        return y, h_new

    def forward(
        self,
        x: torch.Tensor,       # (B, T, D)
        h: torch.Tensor,       # (B, D_inner, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Split into x and z branches
        xz = self.in_proj(x)                        # (B, T, D_inner*2)
        xi, z = xz.chunk(2, dim=-1)                 # each (B, T, D_inner)

        # Short 1D convolution (causal)
        xi_conv = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xi_conv = F.silu(xi_conv)

        # SSM loop over time
        D_inner = self._d_inner
        outs = torch.zeros(B, T, D_inner, device=x.device)
        for t in range(T):
            yt, h = self.ssm_step(xi_conv[:, t, :], h)
            outs[:, t, :] = yt

        y = outs * F.silu(z)
        y = self.out_proj(y)
        return residual + y, h


# ---------------------------------------------------------------------------
# TinyMambaModel
# ---------------------------------------------------------------------------

@register("model", "tiny_mamba")
class TinyMambaModel(nn.Module):
    """Tiny Mamba language model — stateful SSM.

    State per layer in ``ModelOutput.state``: list of h tensors (B, D_inner, N).
    """

    def __init__(self, cfg: TinyMambaConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = TinyMambaConfig(**{k: v for k, v in kwargs.items() if hasattr(TinyMambaConfig, k)})
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._state: list[torch.Tensor] | None = None

    def reset_state(self, batch_size: int = 1, device: Any = "cpu") -> None:
        D_inner = self.cfg.d_model * self.cfg.expand
        N = self.cfg.d_state
        self._state = [
            torch.zeros(batch_size, D_inner, N, device=device)
            for _ in range(self.cfg.num_layers)
        ]

    def forward(self, **batch: Any) -> ModelOutput:
        input_ids: torch.Tensor = batch["input_ids"]
        B, T = input_ids.shape
        device = input_ids.device

        reset = batch.get("_reset_state", False)
        prev_state = batch.get("_arch_state")
        if reset or prev_state is None or self._state is None:
            self.reset_state(B, device)
        else:
            self._state = prev_state

        if self._state[0].shape[0] != B:
            self.reset_state(B, device)

        x = self.embed(input_ids)   # (B, T, d_model)
        new_state = []
        for i, block in enumerate(self.blocks):
            h = self._state[i].to(device)
            x, h = block(x, h)
            new_state.append(h.detach())

        self._state = new_state
        x = self.ln_out(x)
        logits = self.head(x)

        return ModelOutput(outputs={"logits": logits}, state=new_state)


# ---------------------------------------------------------------------------
# ArchitectureProfile
# ---------------------------------------------------------------------------

def _mamba_blocks(model: nn.Module) -> Iterator[nn.Module]:
    yield from model.blocks


def mamba_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        name="mamba",
        loss_family="next_token",
        state_mode="stateful",
        block_iterator_fn=_mamba_blocks,
        embedding_layer_fn=lambda m: m.embed,
        head_layer_fn=lambda m: m.head,
        reset_state_fn=lambda m: m.reset_state(1),
    )


__all__ = ["MambaBlock", "TinyMambaConfig", "TinyMambaModel", "mamba_profile"]
