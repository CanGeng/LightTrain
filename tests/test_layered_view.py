"""LayerHandle / LayeredView — DESIGN §14.2 (M5)."""

from __future__ import annotations

import pytest
import torch.nn as nn

from plugins.layer_offload import (
    LayerHandle,
    LayerOffloadNotSupported,
    get_layered_view,
)
from lighttrain.models.adapters.tiny_lm import TinyCausalLM


def test_tiny_lm_layered_view_has_one_handle_per_block():
    model = TinyCausalLM(
        vocab_size=64, d_model=16, n_layers=3, n_heads=4, max_seq_len=16
    )
    view = get_layered_view(model)
    assert len(view.layers) == 3
    for i, h in enumerate(view.layers):
        assert isinstance(h, LayerHandle)
        assert h.name == f"block.{i}"
        assert isinstance(h.module, nn.Module)
    assert isinstance(view.embed, nn.Module)
    assert isinstance(view.head, nn.Module)


def test_layered_view_falls_through_for_unknown_model():
    model = nn.Linear(4, 8)  # no layered_view registered
    with pytest.raises(LayerOffloadNotSupported):
        get_layered_view(model)


def test_layered_view_drills_through_peft_wrap():
    pytest.importorskip("peft")
    from lighttrain.models.peft import LoRAAdapter

    wrapped = LoRAAdapter(
        base={
            "name": "tiny_lm",
            "vocab_size": 64,
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 4,
            "max_seq_len": 16,
        },
        r=4,
        lora_alpha=8,
    )
    view = get_layered_view(wrapped)
    # tiny_lm has 2 blocks → 2 handles
    assert len(view.layers) == 2
