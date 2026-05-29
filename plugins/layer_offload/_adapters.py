"""``layered_view()`` for lighttrain's built-in adapters.

A model's ``layered_view()`` may live on the model class itself (preferred —
keeps offload logic local to the architecture), but we don't want to make
all adapters depend on this frontier plugin. So we register external
``layered_view`` builders here, keyed by ``type(model).__name__``.

Coverage:

* ``TinyCausalLM`` — the built-in reference model.
* ``HFCausalLM`` — best-effort over the LLaMA / Mistral / Qwen / GPT-2
  / GPT-NeoX families via the standard ``model.model.layers`` /
  ``model.transformer.h`` accessors. Unknown HF families raise
  ``LayerOffloadNotSupported``.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from ._layer_handle import LayerHandle, LayerOffloadNotSupported, _DefaultLayeredView


def _tiny_layered_view(model) -> _DefaultLayeredView:
    """TinyCausalLM → layered view.

    ``embed`` packages tok_emb + pos_emb + drop so the engine can keep them
    glued together; ``layers`` is one handle per transformer block;
    ``head`` packages norm_f + lm_head.
    """

    class _Embed(nn.Module):
        def __init__(self, tok, pos, drop):
            super().__init__()
            self.tok = tok
            self.pos = pos
            self.drop = drop

        def forward(self, input_ids, positions):
            return self.drop(self.tok(input_ids) + self.pos(positions))

    class _Head(nn.Module):
        def __init__(self, norm, head):
            super().__init__()
            self.norm = norm
            self.head = head

        def forward(self, x):
            return self.head(self.norm(x))

    embed = _Embed(model.tok_emb, model.pos_emb, model.drop)
    head = _Head(model.norm_f, model.lm_head)
    layers = [
        LayerHandle(name=f"block.{i}", module=blk)
        for i, blk in enumerate(model.blocks)
    ]
    return _DefaultLayeredView(embed=embed, layers=layers, head=head)


def _hf_layered_view(model) -> _DefaultLayeredView:
    """HFCausalLM → layered view via well-known HF attribute layouts."""
    inner = getattr(model, "inner", None)
    if inner is None:
        raise LayerOffloadNotSupported(
            "HFCausalLM has no `.inner` — base wasn't constructed."
        )
    # Try LLaMA / Mistral / Qwen / Falcon family (most common modern decoders)
    layers_mod = None
    embed_mod = None
    head_mod = None
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        layers_mod = inner.model.layers
        embed_mod = getattr(inner.model, "embed_tokens", None)
        head_mod = getattr(inner, "lm_head", None)
    elif hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
        # GPT-2 / GPT-J family
        layers_mod = inner.transformer.h
        embed_mod = getattr(inner.transformer, "wte", None)
        head_mod = getattr(inner, "lm_head", None)
    elif hasattr(inner, "gpt_neox") and hasattr(inner.gpt_neox, "layers"):
        layers_mod = inner.gpt_neox.layers
        embed_mod = getattr(inner.gpt_neox, "embed_in", None)
        head_mod = getattr(inner, "embed_out", None)
    if layers_mod is None:
        raise LayerOffloadNotSupported(
            f"HFCausalLM family {type(inner).__name__} doesn't expose a "
            f"standard layer list (.model.layers / .transformer.h / "
            f".gpt_neox.layers) — set engine.name=standard or implement "
            f"`{type(inner).__name__}.layered_view()` upstream."
        )
    embed_holder = embed_mod if embed_mod is not None else nn.Identity()
    head_holder = head_mod if head_mod is not None else nn.Identity()
    layers = [
        LayerHandle(name=f"layer.{i}", module=layers_mod[i])
        for i in range(len(layers_mod))
    ]
    return _DefaultLayeredView(embed=embed_holder, layers=layers, head=head_holder)


_REGISTRY: dict[str, Callable] = {
    "TinyCausalLM": _tiny_layered_view,
    "HFCausalLM": _hf_layered_view,
}


def register_layered_view(cls_name: str, fn: Callable) -> None:
    """User extension: register a ``layered_view`` for a custom model class."""
    _REGISTRY[cls_name] = fn


def get_layered_view(model) -> _DefaultLayeredView:
    """Return a ``LayeredView`` for ``model`` or raise ``LayerOffloadNotSupported``."""
    # 1) Method on the model wins (preferred, lives with the architecture)
    if hasattr(model, "layered_view") and callable(model.layered_view):
        return model.layered_view()
    # 2) Built-in adapters lookup
    cls_name = type(model).__name__
    if cls_name in _REGISTRY:
        return _REGISTRY[cls_name](model)
    # 3) PEFT wrap: drill down to the underlying base
    for attr in ("get_base_model", "base_model"):
        next_base = getattr(model, attr, None)
        if callable(next_base):
            return get_layered_view(next_base())
        if hasattr(next_base, "named_parameters") and next_base is not model:
            return get_layered_view(next_base)
    raise LayerOffloadNotSupported(
        f"No layered_view registered for {cls_name!r}. Add one via "
        "`from plugins.layer_offload._adapters import register_layered_view`."
    )


__all__ = ["get_layered_view", "register_layered_view"]
