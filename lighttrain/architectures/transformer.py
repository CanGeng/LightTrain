"""Transformer ArchitectureProfile factory.

Provides a ready-made ArchitectureProfile for standard autoregressive and
masked-language-model Transformers backed by HuggingFace or TinyLM.

Usage::

    from lighttrain.architectures import transformer_profile

    profile = transformer_profile()                   # next_token default
    profile = transformer_profile(loss_family="mlm")  # BERT-style

The seam functions use heuristic attribute traversal so they work for both
``HFCausalLMAdapter`` and ``TinyLMAdapter`` without knowing the concrete type.
"""

from __future__ import annotations

import torch.nn as nn

from .profile import ArchitectureProfile


# ---------------------------------------------------------------------------
# Seam helpers (heuristic attribute traversal)
# ---------------------------------------------------------------------------

def _transformer_blocks(model: nn.Module):
    """Yield transformer blocks via common attribute names."""
    for attr in ("layers", "blocks", "h", "transformer_blocks"):
        seq = getattr(model, attr, None)
        if seq is None:
            # one level deeper (e.g. model.transformer.h for GPT-2 style)
            inner = getattr(model, "transformer", None) or getattr(model, "model", None)
            if inner is not None:
                seq = getattr(inner, attr, None)
        if seq is not None and hasattr(seq, "__iter__"):
            yield from seq
            return
    # Fallback: every direct child that looks like a block (has "self_attn" or "attn")
    for child in model.children():
        if hasattr(child, "self_attn") or hasattr(child, "attn"):
            yield child


def _transformer_embedding(model: nn.Module) -> nn.Module:
    for attr in ("embed_tokens", "wte", "tok_embeddings", "embedding"):
        layer = getattr(model, attr, None)
        if layer is None:
            inner = getattr(model, "transformer", None) or getattr(model, "model", None)
            if inner is not None:
                layer = getattr(inner, attr, None)
        if layer is not None:
            return layer
    raise AttributeError(f"Cannot locate embedding layer on {type(model).__name__}")


def _transformer_head(model: nn.Module) -> nn.Module:
    for attr in ("lm_head", "output", "head"):
        layer = getattr(model, attr, None)
        if layer is not None:
            return layer
    raise AttributeError(f"Cannot locate head layer on {type(model).__name__}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def transformer_profile(
    loss_family: str = "next_token",
    *,
    name: str | None = None,
) -> ArchitectureProfile:
    """Return an ArchitectureProfile suitable for standard Transformers.

    Args:
        loss_family: ``"next_token"`` (default) or ``"mlm"``.
        name: Profile name (defaults to ``"transformer_<loss_family>"``).
    """
    if name is None:
        name = f"transformer_{loss_family}"
    return ArchitectureProfile(
        name=name,
        loss_family=loss_family,
        state_mode="stateless",
        block_iterator_fn=_transformer_blocks,
        embedding_layer_fn=_transformer_embedding,
        head_layer_fn=_transformer_head,
        reset_state_fn=None,
    )


__all__ = ["transformer_profile"]
