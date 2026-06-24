"""Model surgery — DESIGN §8.3 (M5).

Relocated from the flat ``tests/test_surgery.py``. No mirror under
``tests/models/`` covered ``lighttrain.models.surgery``, so the freeze /
resize / replace / add / reinit / tie behaviors are preserved.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.models.surgery import (
    add_named_module,
    count_trainable,
    freeze_modules,
    get_submodule,
    reinit_module,
    replace_module,
    resize_embedding,
    tie_weights,
    unfreeze_modules,
)


def _make_tiny() -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=64, d_model=16, n_layers=2, n_heads=4, max_seq_len=32
    )


# --- freeze ----------------------------------------------------------------


def test_freeze_modules_by_regex_returns_count_and_disables_grad():
    model = _make_tiny()
    n = freeze_modules(model, r"blocks\.0\.")
    assert n > 0
    for name, p in model.named_parameters():
        if name.startswith("blocks.0."):
            assert not p.requires_grad
        else:
            assert p.requires_grad


def test_freeze_modules_idempotent_second_call_returns_zero():
    model = _make_tiny()
    first = freeze_modules(model, r"blocks\.0\.")
    second = freeze_modules(model, r"blocks\.0\.")
    assert first > 0
    assert second == 0


def test_unfreeze_modules_restores_grad():
    model = _make_tiny()
    freeze_modules(model, r".*")
    unfrozen = unfreeze_modules(model, r"lm_head|tok_emb")
    assert unfrozen > 0
    for name, p in model.named_parameters():
        if re.search(r"lm_head|tok_emb", name):
            assert p.requires_grad


def test_count_trainable_after_partial_freeze():
    model = _make_tiny()
    total_before = sum(p.numel() for p in model.parameters())
    freeze_modules(model, r"blocks\.0\.")
    trainable, total = count_trainable(model)
    assert total == total_before
    assert trainable < total


# --- resize_embedding ------------------------------------------------------


def test_resize_embedding_grow_mean_init_preserves_old_rows():
    model = _make_tiny()
    old_size = model.tok_emb.weight.size(0)
    snapshot = model.tok_emb.weight.detach().clone()
    new_size = old_size + 8
    resize_embedding(model, new_size, init="mean")
    assert model.tok_emb.weight.shape == (new_size, model.d_model)
    # Old rows unchanged.
    assert torch.allclose(model.tok_emb.weight[:old_size], snapshot)
    # New rows are the column mean of the old rows.
    expected = snapshot.mean(dim=0)
    for i in range(old_size, new_size):
        assert torch.allclose(model.tok_emb.weight[i], expected, atol=1e-6)
    # Tied head followed along.
    assert model.lm_head.weight.shape == (new_size, model.d_model)
    assert model.lm_head.weight is model.tok_emb.weight


def test_resize_embedding_shrink_keeps_first_rows():
    model = _make_tiny()
    snapshot = model.tok_emb.weight.detach().clone()
    new_size = 16
    resize_embedding(model, new_size)
    assert model.tok_emb.weight.shape == (new_size, model.d_model)
    assert torch.allclose(model.tok_emb.weight, snapshot[:new_size])


def test_resize_embedding_noop_when_same_size():
    model = _make_tiny()
    before = model.tok_emb.weight.data_ptr()
    resize_embedding(model, model.tok_emb.weight.size(0))
    assert model.tok_emb.weight.data_ptr() == before


def test_resize_embedding_zeros_init():
    model = _make_tiny()
    old_size = model.tok_emb.weight.size(0)
    resize_embedding(model, old_size + 4, init="zeros")
    for i in range(old_size, old_size + 4):
        assert torch.all(model.tok_emb.weight[i] == 0.0)


def test_resize_embedding_rejects_unrecognized_model():
    model = nn.Linear(8, 16)
    with pytest.raises(TypeError):
        resize_embedding(model, 32)


# --- replace_module --------------------------------------------------------


def test_replace_module_with_factory_callable_passes_old():
    model = _make_tiny()
    captured: list[nn.Module] = []

    def factory(old):
        captured.append(old)
        return nn.Identity()

    new = replace_module(model, "blocks.0.mlp", factory)
    assert captured and not isinstance(captured[0], nn.Identity)
    assert isinstance(new, nn.Identity)
    assert isinstance(model.blocks[0].mlp, nn.Identity)


def test_replace_module_with_module_verbatim():
    model = _make_tiny()
    replacement = nn.Linear(model.d_model, model.d_model)
    new = replace_module(model, "blocks.0.attn.proj", replacement)
    assert model.blocks[0].attn.proj is new


def test_replace_module_raises_on_missing_path():
    model = _make_tiny()
    with pytest.raises(AttributeError):
        replace_module(model, "blocks.0.nonexistent", nn.Identity())


# --- add_named_module ------------------------------------------------------


def test_add_named_module_creates_intermediate_containers():
    model = _make_tiny()
    lin = nn.Linear(8, 16)
    add_named_module(model, "_distill_projections.layer_0", lin)
    assert get_submodule(model, "_distill_projections.layer_0") is lin
    # Parameter shows up in named_parameters.
    names = [n for n, _ in model.named_parameters()]
    assert "_distill_projections.layer_0.weight" in names


def test_add_named_module_param_followed_by_state_dict_and_to_device():
    model = _make_tiny()
    lin = nn.Linear(4, 8)
    add_named_module(model, "extras.proj", lin)
    sd = model.state_dict()
    assert "extras.proj.weight" in sd
    if torch.cuda.is_available():  # smoke
        model.cuda()
        assert next(lin.parameters()).is_cuda


def test_add_named_module_rejects_non_module_intermediate():
    model = _make_tiny()
    # Inject a non-Module attr in the way and verify we refuse to overwrite.
    model.bad = 42  # type: ignore[assignment]
    with pytest.raises(TypeError):
        add_named_module(model, "bad.child", nn.Linear(2, 2))


# --- reinit_module ---------------------------------------------------------


def test_reinit_module_changes_weights_and_zeros_bias():
    model = _make_tiny()
    before = model.blocks[0].mlp.fc1.weight.detach().clone()
    before_bias = model.blocks[0].mlp.fc1.bias.detach().clone()
    # Use a non-default std so the change is unambiguous.
    n = reinit_module(model, r"blocks\.0\.mlp\.fc1$", dist={"kind": "normal", "std": 5.0})
    assert n == 1
    after = model.blocks[0].mlp.fc1.weight
    assert not torch.allclose(after, before)
    assert torch.all(model.blocks[0].mlp.fc1.bias == 0.0)
    _ = before_bias


def test_reinit_module_no_match_returns_zero():
    model = _make_tiny()
    n = reinit_module(model, r"definitely_not_present")
    assert n == 0


# --- tie / untie -----------------------------------------------------------


def test_tie_weights_makes_storage_shared():
    model = _make_tiny()
    # tiny_lm ties by default; untie first so we can re-tie.
    from lighttrain.models.surgery import untie_weights as _untie

    _untie(model, "lm_head")
    assert model.lm_head.weight is not model.tok_emb.weight
    tie_weights(model, "tok_emb", "lm_head")
    assert model.lm_head.weight is model.tok_emb.weight


def test_untie_weights_clones_storage_but_keeps_values():
    model = _make_tiny()
    assert model.lm_head.weight is model.tok_emb.weight
    before = model.lm_head.weight.detach().clone()
    from lighttrain.models.surgery import untie_weights as _untie

    _untie(model, "lm_head")
    assert model.lm_head.weight is not model.tok_emb.weight
    assert torch.allclose(model.lm_head.weight, before)
