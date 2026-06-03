"""Safe embedding resize.

Grows or shrinks the token embedding matrix (and the LM head, if tied or
present) without dropping existing weights. New rows can be initialized:

* ``"mean"`` (default) — average of the existing token embeddings; matches
  the strategy commonly used when adding chat / vision tokens to an LLM
  vocab without disturbing the geometry of the original token cloud;
* ``"zeros"`` — exactly zero rows (useful for debugging);
* ``"normal"`` — sample from ``N(0, 0.02)`` (lighttrain's default
  initialization).

Recognizes the two built-in adapters by attribute layout:

* :class:`lighttrain.builtin_plugins.models.adapters.tiny_lm.TinyCausalLM` exposes
  ``tok_emb`` / ``lm_head`` directly (and ties weights via shared storage).
* :class:`lighttrain.builtin_plugins.models.adapters.hf_causal.HFCausalLM` delegates to its
  ``inner`` HuggingFace model, which provides ``resize_token_embeddings``;
  we forward to that.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

InitMode = Literal["mean", "zeros", "normal"]


def _init_rows(matrix: torch.Tensor, *, mode: InitMode, std: float = 0.02) -> None:
    if mode == "mean":
        # Mean across the *old* rows (assumes caller passed only the new slice).
        # Caller does: old_rows = emb.weight[:old_size]; new = emb.weight[old_size:];
        # we just init `new` here when mode != "mean" (mean handled in resize_embedding).
        raise RuntimeError("`mean` init handled by resize_embedding directly")
    if mode == "zeros":
        matrix.zero_()
    elif mode == "normal":
        matrix.normal_(mean=0.0, std=std)
    else:
        raise ValueError(f"Unknown init mode: {mode!r}")


def resize_embedding(
    model: nn.Module,
    new_vocab_size: int,
    *,
    init: InitMode = "mean",
) -> None:
    """Resize embeddings (and the tied / present LM head) to ``new_vocab_size``.

    No-op when ``new_vocab_size`` equals the current ``vocab_size``. Shrinking
    keeps the first ``new_vocab_size`` rows.
    """
    # HF adapter path — delegate to transformers' resizer (handles tie / pad).
    inner = getattr(model, "inner", None)
    if inner is not None and hasattr(inner, "resize_token_embeddings"):
        inner.resize_token_embeddings(new_vocab_size)
        if hasattr(model, "vocab_size"):
            model.vocab_size = int(new_vocab_size)
        return

    # tiny_lm / generic adapters that expose ``tok_emb`` (+ optional ``lm_head``).
    tok_emb = getattr(model, "tok_emb", None)
    if tok_emb is None or not isinstance(tok_emb, nn.Embedding):
        raise TypeError(
            f"resize_embedding: cannot find ``tok_emb: nn.Embedding`` on "
            f"{type(model).__name__}; HFCausalLM uses ``inner`` instead."
        )
    old_size, dim = tok_emb.weight.shape
    if new_vocab_size == old_size:
        return

    lm_head: nn.Linear | None = getattr(model, "lm_head", None)
    tied = isinstance(lm_head, nn.Linear) and (lm_head.weight is tok_emb.weight)

    # New embedding tensor.
    new_emb = nn.Embedding(new_vocab_size, dim).to(
        device=tok_emb.weight.device, dtype=tok_emb.weight.dtype
    )
    with torch.no_grad():
        keep = min(old_size, new_vocab_size)
        new_emb.weight[:keep] = tok_emb.weight[:keep]
        if new_vocab_size > old_size:
            extra = new_vocab_size - old_size
            if init == "mean":
                row_mean = tok_emb.weight[:old_size].mean(dim=0)
                new_emb.weight[old_size:] = row_mean.unsqueeze(0).expand(extra, dim)
            else:
                _init_rows(new_emb.weight[old_size:], mode=init)
    model.tok_emb = new_emb

    if lm_head is not None and isinstance(lm_head, nn.Linear):
        bias = lm_head.bias is not None
        if tied:
            # Re-tie head weight to the new embedding storage.
            new_head = nn.Linear(dim, new_vocab_size, bias=bias)
            new_head.weight = new_emb.weight
            if bias:
                with torch.no_grad():
                    keep = min(old_size, new_vocab_size)
                    new_head.bias[:keep] = lm_head.bias[:keep]
                    if new_vocab_size > old_size:
                        new_head.bias[old_size:] = 0.0
            model.lm_head = new_head
        else:
            new_head = nn.Linear(dim, new_vocab_size, bias=bias).to(
                device=lm_head.weight.device, dtype=lm_head.weight.dtype
            )
            with torch.no_grad():
                keep = min(old_size, new_vocab_size)
                new_head.weight[:keep] = lm_head.weight[:keep]
                if new_vocab_size > old_size:
                    if init == "mean":
                        row_mean = lm_head.weight[:old_size].mean(dim=0)
                        new_head.weight[old_size:] = row_mean.unsqueeze(0).expand(
                            new_vocab_size - old_size, dim
                        )
                    else:
                        _init_rows(new_head.weight[old_size:], mode=init)
                if bias:
                    new_head.bias[:keep] = lm_head.bias[:keep]
                    if new_vocab_size > old_size:
                        new_head.bias[old_size:] = 0.0
            model.lm_head = new_head

    if hasattr(model, "vocab_size"):
        model.vocab_size = int(new_vocab_size)


__all__ = ["resize_embedding"]
