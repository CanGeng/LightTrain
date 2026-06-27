"""NanoGPT model — faithful port of Andrej Karpathy's nanoGPT for lighttrain.

Architecture is identical to the original: pre-norm GPT-2 style transformer
with optional bias in LayerNorm / Linear, flash attention, weight tying, and
the GPT-2 conv-weight naming convention (for pretrained weight compatibility).

References:
  https://github.com/karpathy/nanoGPT
  https://github.com/openai/gpt-2/blob/master/src/model.py
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register


class _LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch's built-in lacks bias=False)."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _CausalSelfAttention(nn.Module):

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _MLP(nn.Module):

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class _Block(nn.Module):

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = _LayerNorm(config.n_embd, bias=config.bias)
        self.attn = _CausalSelfAttention(config)
        self.ln_2 = _LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPTConfig:
    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


@register("model", "nano_gpt")
class NanoGPT(nn.Module):
    """Faithful nanoGPT port registered as a lighttrain model.

    Extra kwargs vs. the original:
      pretrained (str | None): if set to 'gpt2', 'gpt2-medium', 'gpt2-large',
        or 'gpt2-xl', load OpenAI GPT-2 weights via HuggingFace after init.
    """

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        pretrained: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = NanoGPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([_Block(self._cfg) for _ in range(n_layer)]),
                ln_f=_LayerNorm(n_embd, bias=bias),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        if pretrained is not None:
            self._load_pretrained(pretrained, dropout)

        print(f"NanoGPT: {self.get_num_params() / 1e6:.2f}M parameters")

    # ------------------------------------------------------------------
    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # unused, accepted for compat
        **_: Any,
    ) -> ModelOutput:
        B, T = input_ids.size()
        assert T <= self._cfg.block_size, (
            f"sequence length {T} > block_size {self._cfg.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return ModelOutput(outputs={"logits": logits})

    # ------------------------------------------------------------------
    def crop_block_size(self, block_size: int) -> None:
        """Shrink block_size post-hoc (e.g. after loading GPT-2 weights)."""
        assert block_size <= self._cfg.block_size
        self._cfg.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # ------------------------------------------------------------------
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """AdamW with weight-decay split (2-D tensors decay; biases/norms don't)."""
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **({"fused": True} if use_fused else {}),
        )

    # ------------------------------------------------------------------
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Model FLOPs utilization (MFU) relative to A100 bf16 peak (312 TFLOPS)."""
        N = self.get_num_params()
        L, H, Q, T = (
            self._cfg.n_layer, self._cfg.n_head,
            self._cfg.n_embd // self._cfg.n_head, self._cfg.block_size,
        )
        flops_per_iter = (6 * N + 12 * L * H * Q * T) * T * fwdbwd_per_iter
        return flops_per_iter / dt / 312e12

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self._cfg.block_size
                else idx[:, -self._cfg.block_size:]
            )
            logits = self(idx_cond).outputs["logits"][:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

    # ------------------------------------------------------------------
    def _load_pretrained(self, model_type: str, dropout: float) -> None:
        """Load OpenAI GPT-2 weights from HuggingFace (requires `transformers`)."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, (
            f"Unknown pretrained model: {model_type}"
        )
        from transformers import GPT2LMHeadModel  # type: ignore[import-untyped]

        print(f"Loading pretrained GPT-2 weights: {model_type}")
        hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf.state_dict()
        sd = self.state_dict()

        # HF uses Conv1D (transposed) for attn/mlp projections
        transposed = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        ]
        keys_hf = [
            k for k in sd_hf
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]
        keys = [k for k in sd if not k.endswith(".attn.bias")]
        assert len(keys_hf) == len(keys), (
            f"Key count mismatch: HF {len(keys_hf)} vs local {len(keys)}"
        )
        for k in keys_hf:
            if any(k.endswith(s) for s in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        # Crop to match current block_size / vocab_size if needed
        if self._cfg.block_size < 1024:
            self.crop_block_size(self._cfg.block_size)


__all__ = ["NanoGPT", "NanoGPTConfig"]
