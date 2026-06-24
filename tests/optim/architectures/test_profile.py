"""Adversarial tests for ``lighttrain.optim.architectures.profile`` /
``lighttrain.builtin_plugins.architectures.transformer``.

Layered on top of the flat ``tests/test_architectures_profile.py`` smoke
tests (which cover the happy path: ``embed_tokens`` + ``layers`` + ``lm_head``).
This file pins:

* **Block-iterator attribute search depth** — every candidate attribute
  name on the top level AND one level deeper (`.transformer` / `.model`)
  is checked in order.
* **Block-iterator self_attn/attn child fallback** — when no canonical
  attribute matches, ``_transformer_blocks`` falls back to scanning
  direct children for ``self_attn`` / ``attn`` markers.
* **Block iterator yields in declaration order** — parametrized over
  n_layers ∈ {2, 3, 4, 5, 6}.
* **Embedding / head lookup raises AttributeError when not found** —
  with a clear message.
* **Profile name default** — ``transformer_profile(loss_family="mlm").name
  == "transformer_mlm"``.
* **Stateless profile reset_state raises NotImplementedError** with the
  profile name in the message.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from lighttrain.builtin_plugins.architectures.transformer import (
    _transformer_blocks,
    _transformer_embedding,
    _transformer_head,
    transformer_profile,
)
from lighttrain.optim.architectures import ArchitectureProfile

# ---------------------------------------------------------------------------
# Helpers — adversarial model topologies
# ---------------------------------------------------------------------------

class _ToyBlock(nn.Module):
    """A block that exposes ``self_attn`` so child-scanning recognizes it."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x): return self.mlp(self.self_attn(x))


def _make_canonical_transformer(n_layers: int = 2):
    """Canonical layout: model.embed_tokens / model.layers / model.lm_head."""
    class Canonical(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            self.layers = nn.ModuleList([_ToyBlock(8) for _ in range(n_layers)])
            self.lm_head = nn.Linear(8, 32)
        def forward(self, ids):
            x = self.embed_tokens(ids)
            for blk in self.layers:
                x = blk(x)
            return {"logits": self.lm_head(x)}
    return Canonical()


def _make_gpt2_style(n_layers: int = 2):
    """GPT-2 style: model.transformer.h with model.wte at the outer level
    one level deeper.
    """
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(32, 8)
            self.h = nn.ModuleList([_ToyBlock(8) for _ in range(n_layers)])
    class GPT2Style(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Inner()
            self.lm_head = nn.Linear(8, 32)
    return GPT2Style()


def _make_llama_style(n_layers: int = 2):
    """LLaMA style: model.model.layers, model.model.embed_tokens, model.lm_head."""
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            self.layers = nn.ModuleList([_ToyBlock(8) for _ in range(n_layers)])
    class LLaMAStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(8, 32)
    return LLaMAStyle()


def _make_fallback_layout(n_blocks: int = 2):
    """Layout with NO canonical attribute names — only the self_attn/attn
    child-scan fallback should find blocks.
    """
    class FallbackModel(nn.Module):
        def __init__(self):
            super().__init__()
            # No ``layers`` / ``blocks`` / ``h`` / ``transformer_blocks``;
            # blocks attached under arbitrary names.
            for i in range(n_blocks):
                setattr(self, f"custom_block_{i}", _ToyBlock(8))
            self.embed_tokens = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32)
    return FallbackModel()


def _make_no_blocks_at_all():
    """Layout with embedding + head but ZERO recognizable block structure."""
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32)
    return Empty()


# ---------------------------------------------------------------------------
# ArchitectureProfile basics
# ---------------------------------------------------------------------------

def test_pin_architecture_profile_state_mode_default_is_stateless():
    """Pin: default ``state_mode`` is the literal string ``"stateless"``.

    If you intentionally change the default, update this test and any
    StatefulTrainer dispatch logic that branches on this exact string.
    """
    p = ArchitectureProfile(name="x", loss_family="next_token")
    assert p.state_mode == "stateless"


def test_architecture_profile_iter_blocks_raises_when_no_iterator_provided():
    """``iter_blocks`` on a profile without ``block_iterator_fn`` raises
    NotImplementedError, naming the profile.
    """
    p = ArchitectureProfile(name="bare", loss_family="next_token")
    with pytest.raises(NotImplementedError) as exc:
        list(p.iter_blocks(_make_canonical_transformer()))
    assert "bare" in str(exc.value)


def test_architecture_profile_get_embedding_raises_when_not_provided():
    """``get_embedding`` raises when the seam is unset."""
    p = ArchitectureProfile(name="bare", loss_family="next_token")
    with pytest.raises(NotImplementedError):
        p.get_embedding(_make_canonical_transformer())


def test_architecture_profile_get_head_raises_when_not_provided():
    """``get_head`` raises when the seam is unset."""
    p = ArchitectureProfile(name="bare", loss_family="next_token")
    with pytest.raises(NotImplementedError):
        p.get_head(_make_canonical_transformer())


def test_architecture_profile_reset_state_raises_with_state_mode_in_message():
    """``reset_state`` on a stateless profile raises with the state_mode
    in the message (so callers can debug the dispatch failure).
    """
    p = ArchitectureProfile(name="x", loss_family="next_token", state_mode="stateful")
    with pytest.raises(NotImplementedError) as exc:
        p.reset_state(_make_canonical_transformer())
    assert "stateful" in str(exc.value)


def test_architecture_profile_reset_state_dispatches_to_user_fn():
    """When ``reset_state_fn`` is supplied, ``reset_state`` calls it exactly
    once with the model as the only argument.
    """
    received = []

    def _reset(m: nn.Module) -> None:
        received.append(m)

    p = ArchitectureProfile(
        name="x", loss_family="next_token",
        state_mode="stateful", reset_state_fn=_reset,
    )
    model = _make_canonical_transformer()
    p.reset_state(model)
    assert len(received) == 1
    assert received[0] is model


# ---------------------------------------------------------------------------
# transformer_profile factory
# ---------------------------------------------------------------------------

def test_pin_transformer_profile_default_name_format():
    """Pin: ``transformer_profile()`` uses ``name = f"transformer_{loss_family}"``.

    If you change the format, update the profile registry / lookups that
    rely on this exact string. (Currently used by LayerOffloadEngine
    dispatch.)
    """
    assert transformer_profile().name == "transformer_next_token"
    assert transformer_profile(loss_family="mlm").name == "transformer_mlm"


def test_transformer_profile_name_override_takes_precedence():
    """Explicit ``name=`` overrides the default ``transformer_<loss_family>``
    string.
    """
    p = transformer_profile(loss_family="mlm", name="my_special_mlm")
    assert p.name == "my_special_mlm"
    assert p.loss_family == "mlm"


def test_pin_transformer_profile_is_stateless():
    """Pin: the transformer profile is always ``state_mode="stateless"``."""
    p = transformer_profile()
    assert p.state_mode == "stateless"


def test_transformer_profile_get_embedding_and_head_wired_through_factory():
    """End-to-end seam: a profile built by ``transformer_profile()`` resolves
    ``get_embedding`` to the model's ``nn.Embedding`` and ``get_head`` to its
    ``nn.Linear`` head on a canonical model (factory wiring, not the bare
    ``_transformer_embedding`` / ``_transformer_head`` helpers tested above).
    """
    model = _make_canonical_transformer()
    p = transformer_profile()
    emb = p.get_embedding(model)
    head = p.get_head(model)
    assert isinstance(emb, nn.Embedding)
    assert isinstance(head, nn.Linear)
    assert emb is model.embed_tokens
    assert head is model.lm_head


# ---------------------------------------------------------------------------
# _transformer_blocks — canonical name search at top level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_layers", [2, 3, 4, 5, 6])
def test_invariant_canonical_layers_attr_yields_blocks_in_declaration_order(n_layers):
    """Invariant: ``model.layers`` is iterated in registration order;
    parametrized over n_layers to catch any off-by-one in the iterator.

    Setup: canonical model with n_layers blocks.
    Expected: ``list(_transformer_blocks(model))`` length == n_layers AND
    each yielded block IS identical (by ``id``) to the corresponding
    ``model.layers[i]``.
    """
    model = _make_canonical_transformer(n_layers=n_layers)
    blocks = list(_transformer_blocks(model))
    assert len(blocks) == n_layers
    for i, blk in enumerate(blocks):
        assert blk is model.layers[i], f"block {i} identity mismatch"


def test_transformer_blocks_finds_gpt2_h_via_one_level_deeper():
    """GPT-2-style: ``model.transformer.h`` is discovered by the "one level
    deeper via .transformer" fallback (lines 33-36 of transformer.py).
    """
    model = _make_gpt2_style(n_layers=3)
    blocks = list(_transformer_blocks(model))
    assert len(blocks) == 3
    for i, blk in enumerate(blocks):
        assert blk is model.transformer.h[i]


def test_transformer_blocks_finds_llama_layers_via_model_attr():
    """LLaMA-style: ``model.model.layers`` discovered via the ``.model``
    deeper-fallback branch.
    """
    model = _make_llama_style(n_layers=4)
    blocks = list(_transformer_blocks(model))
    assert len(blocks) == 4
    for i, blk in enumerate(blocks):
        assert blk is model.model.layers[i]


def test_invariant_transformer_blocks_fallback_to_self_attn_children():
    """Invariant: when no canonical attribute matches, the iterator scans
    direct children for ``self_attn`` / ``attn`` markers (lines 41-43 of
    transformer.py).

    Setup: a model with no ``layers``/``blocks``/``h``/``transformer_blocks``
    attribute, but child modules with ``self_attn``.
    Expected: those child modules are yielded.
    """
    model = _make_fallback_layout(n_blocks=3)
    blocks = list(_transformer_blocks(model))
    # The fallback should find at least our 3 custom blocks; the embedding
    # (nn.Embedding) and lm_head (nn.Linear) have no self_attn so they're
    # skipped. We expect EXACTLY 3.
    assert len(blocks) == 3
    found_names = {id(b) for b in blocks}
    expected_names = {id(getattr(model, f"custom_block_{i}")) for i in range(3)}
    assert found_names == expected_names


def test_transformer_blocks_no_blocks_yields_empty_iterator():
    """Model with embedding + head but NO blocks → iterator is empty
    (no exception).
    """
    model = _make_no_blocks_at_all()
    blocks = list(_transformer_blocks(model))
    assert blocks == []


# ---------------------------------------------------------------------------
# _transformer_embedding — name resolution
# ---------------------------------------------------------------------------

def test_transformer_embedding_finds_embed_tokens_at_top_level():
    """Canonical ``embed_tokens`` is the first preferred name."""
    model = _make_canonical_transformer()
    emb = _transformer_embedding(model)
    assert emb is model.embed_tokens


def test_transformer_embedding_finds_wte_via_transformer_inner():
    """``model.transformer.wte`` (GPT-2 style) is found through the
    one-level-deeper fallback.
    """
    model = _make_gpt2_style()
    emb = _transformer_embedding(model)
    assert emb is model.transformer.wte


def test_transformer_embedding_finds_llama_inner_embed_tokens():
    """LLaMA style: ``model.model.embed_tokens``."""
    model = _make_llama_style()
    emb = _transformer_embedding(model)
    assert emb is model.model.embed_tokens


def test_transformer_embedding_raises_attribute_error_when_not_found():
    """No recognizable embedding attribute → AttributeError with class name."""
    class NoEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(8, 32)
    with pytest.raises(AttributeError) as exc:
        _transformer_embedding(NoEmb())
    assert "NoEmb" in str(exc.value)


# ---------------------------------------------------------------------------
# _transformer_head — name resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr_name", ["lm_head", "output", "head"])
def test_transformer_head_finds_by_canonical_name(attr_name):
    """Pin: head search checks ``lm_head`` / ``output`` / ``head`` in that
    order.

    Parametrize over the 3 accepted names. Build a model where only that
    name is present; assert it's returned.
    """
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            setattr(self, attr_name, nn.Linear(8, 32))
    model = Mixed()
    head = _transformer_head(model)
    assert head is getattr(model, attr_name)


def test_pin_transformer_head_search_order_lm_head_first():
    """Pin: when multiple recognized head names coexist, ``lm_head`` wins
    (it's first in the search list).

    Setup: model with both ``lm_head`` AND ``output`` defined.
    Expected: ``lm_head`` is returned.

    If you reorder the search list, update this test.
    """
    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(8, 32)
            self.output = nn.Linear(8, 32)
    model = Both()
    assert _transformer_head(model) is model.lm_head


def test_transformer_head_raises_attribute_error_when_not_found():
    """No recognizable head attribute → AttributeError with class name."""
    class NoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
    with pytest.raises(AttributeError) as exc:
        _transformer_head(NoHead())
    assert "NoHead" in str(exc.value)
