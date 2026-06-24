"""Tests for TinyRWKV architecture (M7)."""
import torch

from lighttrain.builtin_plugins.optim.architectures.rwkv import (
    TinyRWKVConfig,
    TinyRWKVModel,
    rwkv_profile,
)


def _model(vocab_size=32, embed_dim=16, num_layers=2, chunk_size=8):
    cfg = TinyRWKVConfig(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_layers=num_layers, chunk_size=chunk_size,
    )
    return TinyRWKVModel(cfg)


def test_rwkv_forward_shape():
    model = _model()
    ids = torch.randint(0, 32, (2, 8))
    out = model(input_ids=ids)
    assert out.outputs["logits"].shape == (2, 8, 32)


def _clone_state(state):
    """Clone list-of-tuples RWKV state."""
    return [tuple(t.clone() for t in layer_state) for layer_state in state]


def _states_differ(s1, s2):
    """Return True if any tensor in state changed."""
    return any(
        not torch.allclose(t1, t2)
        for ls1, ls2 in zip(s1, s2, strict=False)
        for t1, t2 in zip(ls1, ls2, strict=False)
    )


def test_rwkv_state_persists_across_chunks():
    model = _model()
    ids1 = torch.randint(0, 32, (1, 8))
    model(input_ids=ids1)
    state_after_chunk1 = _clone_state(model._state)

    ids2 = torch.randint(0, 32, (1, 8))
    model(input_ids=ids2)
    state_after_chunk2 = _clone_state(model._state)

    # State should change between chunks
    assert _states_differ(state_after_chunk1, state_after_chunk2)


def test_rwkv_state_reset_on_doc_boundary():
    model = _model()
    ids = torch.randint(0, 32, (1, 8))
    # Run first chunk to build up state
    model(input_ids=ids)
    # Pass reset signal
    model(input_ids=ids, _reset_state=True)
    # After forward with reset, state is rebuilt — confirm it doesn't crash
    assert model._state is not None
    assert len(model._state) == model.cfg.num_layers


def test_rwkv_reset_via_batch_flag():
    model = _model()
    ids = torch.randint(0, 32, (1, 8))
    model(input_ids=ids)
    # Reset via batch dict key
    batch = {"input_ids": ids, "_reset_state": True}
    out = model(**batch)
    assert out.outputs["logits"].shape == (1, 8, 32)


def test_rwkv_profile_stateful():
    profile = rwkv_profile()
    assert profile.state_mode == "stateful"
    assert profile.loss_family == "next_token"


def test_rwkv_profile_reset_fn():
    model = _model()
    profile = rwkv_profile()
    ids = torch.randint(0, 32, (1, 8))
    model(input_ids=ids)
    _clone_state(model._state)
    # Reset via profile
    profile.reset_state(model)
    # State should be reset to zeros
    assert model._state is not None
    assert len(model._state) == model.cfg.num_layers
    # All tensors should be zero after reset
    assert all(
        torch.all(t == 0)
        for layer_state in model._state
        for t in layer_state
    )
