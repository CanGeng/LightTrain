"""Tests for ``lighttrain.builtin_plugins.models.text.tiny_lm.TinyCausalLM``.

Coverage pins:
* ``_CausalSelfAttention.__init__``: d_model % n_heads != 0 raises ValueError (line 26).
* ``_CausalSelfAttention.forward`` with ``return_attentions=True``:
  - causal-mask + explicit matmul path (lines 55-68).
  - dropout + training-mode branch (line 63-64).
  - no-dropout branch / eval mode (line 66).
  - padding attention mask applied in explicit path (line 61).
* ``TinyCausalLM.forward``:
  - 1D input_ids raises ValueError (line 166).
  - sequence length > max_seq_len raises ValueError (line 168-169).
  - output_hidden_states=True populates hidden_states (line 178-179).
  - output_attentions=True populates attentions (line 180-181).
  - default flags False → None in output (lines 161-164).
* ``TinyCausalLM.generate`` (lines 203-213):
  - num_return_sequences > 1 replicates input (line 204).
  - max_new_tokens clipped to remaining capacity (line 206).
  - n<=0 returns input unchanged (line 207-208).
  - normal token generation (line 209-213).
* ``TinyCausalLM.num_parameters`` (line 216).
* Registry: model registered under 'tiny_lm' key.
* Tied-weight invariant.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.models.text.tiny_lm import (
    TinyCausalLM,
    _CausalSelfAttention,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny(
    vocab_size: int = 32,
    d_model: int = 16,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 16,
    dropout: float = 0.0,
    **kwargs,
) -> TinyCausalLM:
    torch.manual_seed(0)
    return TinyCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
        **kwargs,
    )


def _batch(B: int = 2, T: int = 4, V: int = 32) -> dict:
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# _CausalSelfAttention
# ---------------------------------------------------------------------------

def test_invariant_causal_attention_raises_when_d_model_not_divisible():
    """Line 26: d_model not divisible by n_heads raises ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        _CausalSelfAttention(d_model=10, n_heads=3, dropout=0.0)


def test_invariant_causal_attention_valid_construction():
    """Valid d_model / n_heads creates the module without error."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    assert attn.head_dim == 4
    assert attn.n_heads == 4


# ---------------------------------------------------------------------------
# return_attentions=True  (lines 55-68 explicit-matmul path)
# ---------------------------------------------------------------------------

def test_invariant_return_attentions_shape():
    """Lines 55-68: return_attentions=True returns (out, probs) both non-None."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 2, 5, 16
    x = torch.randn(B, T, C)
    out, probs = attn(x, attention_mask=None, return_attentions=True)
    # out shape must match input
    assert out.shape == (B, T, C)
    # probs: (B, n_heads, T, T)
    assert probs is not None
    assert probs.shape == (B, 4, T, T)


def test_invariant_return_attentions_probs_sum_to_one_along_last_dim():
    """Softmax over last dim of attention probs must sum to ~1."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 1, 6, 16
    x = torch.randn(B, T, C)
    _, probs = attn(x, attention_mask=None, return_attentions=True)
    row_sums = probs.sum(dim=-1)  # (B, H, T)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_invariant_return_attentions_causal_mask_upper_triangle_zero():
    """Causal mask: probs for future positions (upper triangle) are 0."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 1, 4, 16
    x = torch.randn(B, T, C)
    _, probs = attn(x, attention_mask=None, return_attentions=True)
    # probs[b, h, i, j] should be 0 for j > i (upper triangle)
    for i in range(T):
        for j in range(i + 1, T):
            assert probs[0, :, i, j].abs().max().item() < 1e-6, (
                f"probs[0, :, {i}, {j}] = {probs[0, :, i, j]} should be 0"
            )


def test_invariant_return_attentions_with_padding_mask():
    """Line 60-61: padding attention_mask sets masked positions to -inf → 0 prob."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 2, 4, 16
    x = torch.randn(B, T, C)
    # mask the last token in batch item 0
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -1] = 0
    _, probs = attn(x, attention_mask=mask, return_attentions=True)
    # position T-1 (padded) should have 0 attention weight in item 0
    assert probs[0, :, :, -1].abs().max().item() < 1e-5


def test_pin_current_behavior_return_attentions_without_padding_mask_no_attn_mask():
    """Lines 47-51, 60: when attention_mask is None, attn_mask stays None
    and the explicit path skips the attn_mask addition (line 60 branch not taken).
    Pinning current behavior: probabilities are still causal and sum to 1.
    """
    attn = _CausalSelfAttention(d_model=8, n_heads=2, dropout=0.0)
    attn.eval()
    torch.manual_seed(42)
    B, T, C = 1, 3, 8
    x = torch.randn(B, T, C)
    _, probs = attn(x, attention_mask=None, return_attentions=True)
    assert probs is not None
    # Still must sum to 1 and be causal
    assert torch.allclose(probs.sum(-1), torch.ones(B, 2, T), atol=1e-5)


def test_invariant_return_attentions_dropout_eval_skips_dropout():
    """Line 66: when model is in eval mode, probs_drop == probs (no dropout applied)."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.5)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 1, 4, 16
    x = torch.randn(B, T, C)
    _, probs = attn(x, attention_mask=None, return_attentions=True)
    # In eval mode, probs row sums should still be ~1 (no masking from dropout)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 4, T), atol=1e-5)


def test_invariant_return_attentions_dropout_training_mode():
    """Lines 63-64: in training mode with dropout > 0, probs_drop may differ.
    We only verify the output shape is correct and probabilities come out.
    """
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.9)
    attn.train()
    torch.manual_seed(99)
    B, T, C = 1, 6, 16
    x = torch.randn(B, T, C)
    out, probs = attn(x, attention_mask=None, return_attentions=True)
    assert out.shape == (B, T, C)
    assert probs is not None
    assert probs.shape == (B, 4, T, T)


def test_invariant_return_attentions_false_returns_none_probs():
    """Default path (return_attentions=False): second return is None."""
    attn = _CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    B, T, C = 2, 4, 16
    x = torch.randn(B, T, C)
    out, probs = attn(x, attention_mask=None, return_attentions=False)
    assert out.shape == (B, T, C)
    assert probs is None


# ---------------------------------------------------------------------------
# TinyCausalLM.forward — validation errors
# ---------------------------------------------------------------------------

def test_invariant_forward_raises_on_1d_input():
    """Line 166: 1D input_ids raises ValueError with shape info."""
    model = _make_tiny()
    with pytest.raises(ValueError, match="2D"):
        model(input_ids=torch.zeros(4, dtype=torch.long))


def test_invariant_forward_raises_on_3d_input():
    """Line 166: 3D input_ids also raises (not 2D)."""
    model = _make_tiny()
    with pytest.raises(ValueError, match="2D"):
        model(input_ids=torch.zeros(2, 4, 8, dtype=torch.long))


def test_invariant_forward_raises_when_seq_exceeds_max():
    """Line 168-169: T > max_seq_len raises ValueError mentioning max_seq_len."""
    model = _make_tiny(max_seq_len=8)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids=torch.zeros(1, 9, dtype=torch.long))


def test_invariant_forward_at_max_seq_len_ok():
    """T == max_seq_len is valid (boundary check is strictly greater)."""
    model = _make_tiny(max_seq_len=8)
    model.eval()
    out = model(input_ids=torch.zeros(1, 8, dtype=torch.long))
    assert out.outputs["logits"].shape == (1, 8, 32)


# ---------------------------------------------------------------------------
# TinyCausalLM.forward — normal output shape
# ---------------------------------------------------------------------------

def test_invariant_forward_output_logits_shape():
    """forward returns ModelOutput with logits of shape (B, T, vocab_size)."""
    model = _make_tiny()
    model.eval()
    b = _batch()
    out = model(**b)
    assert isinstance(out, ModelOutput)
    assert out.outputs["logits"].shape == (2, 4, 32)


def test_invariant_forward_default_no_hidden_states_no_attentions():
    """Lines 161-164: with defaults, hidden_states and attentions are None."""
    model = _make_tiny()
    model.eval()
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert out.hidden_states is None
    assert out.attentions is None


# ---------------------------------------------------------------------------
# TinyCausalLM.forward — output_hidden_states
# ---------------------------------------------------------------------------

def test_invariant_forward_output_hidden_states_true():
    """Lines 174, 178-179: output_hidden_states=True collects pre-block and per-block states."""
    model = _make_tiny(n_layers=2)
    model.eval()
    out = model(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        output_hidden_states=True,
    )
    assert out.hidden_states is not None
    # n_layers+1 states: initial embedding + 1 per block
    assert len(out.hidden_states) == 3  # 1 init + 2 blocks
    for hs in out.hidden_states:
        assert hs.shape == (1, 4, 16)  # (B, T, d_model)


def test_invariant_forward_output_hidden_states_constructor_default():
    """output_hidden_states default from constructor (output_hidden_states=True)
    is respected even when the kwarg is not passed to forward().
    """
    model = _make_tiny(n_layers=2, output_hidden_states=True)
    model.eval()
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert out.hidden_states is not None
    assert len(out.hidden_states) == 3


def test_invariant_forward_output_hidden_states_kwarg_overrides_default():
    """forward() kwarg output_hidden_states=False overrides constructor default=True."""
    model = _make_tiny(n_layers=2, output_hidden_states=True)
    model.eval()
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long), output_hidden_states=False)
    assert out.hidden_states is None


# ---------------------------------------------------------------------------
# TinyCausalLM.forward — output_attentions
# ---------------------------------------------------------------------------

def test_invariant_forward_output_attentions_true():
    """Lines 175, 180-181: output_attentions=True collects per-block attention probs."""
    model = _make_tiny(n_layers=2, n_heads=4)
    model.eval()
    out = model(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        output_attentions=True,
    )
    assert out.attentions is not None
    assert len(out.attentions) == 2  # one per block
    for attn in out.attentions:
        # (B, n_heads, T, T)
        assert attn.shape == (1, 4, 4, 4)


def test_invariant_forward_output_attentions_constructor_default():
    """output_attentions=True constructor default flows into forward."""
    model = _make_tiny(n_layers=2, n_heads=4, output_attentions=True)
    model.eval()
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert out.attentions is not None
    assert len(out.attentions) == 2


def test_invariant_forward_output_attentions_kwarg_overrides_default():
    """forward() kwarg output_attentions=False overrides constructor default=True."""
    model = _make_tiny(n_layers=2, output_attentions=True)
    model.eval()
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long), output_attentions=False)
    assert out.attentions is None


def test_invariant_forward_both_outputs_enabled():
    """Both output_hidden_states and output_attentions can be True simultaneously."""
    model = _make_tiny(n_layers=3, n_heads=4)
    model.eval()
    out = model(
        input_ids=torch.zeros(2, 6, dtype=torch.long),
        output_hidden_states=True,
        output_attentions=True,
    )
    assert out.hidden_states is not None
    assert out.attentions is not None
    assert len(out.hidden_states) == 4   # init + 3 blocks
    assert len(out.attentions) == 3       # 3 blocks


# ---------------------------------------------------------------------------
# TinyCausalLM.generate
# ---------------------------------------------------------------------------

def test_invariant_generate_basic_shape():
    """Lines 205-213: generate appends max_new_tokens tokens when capacity allows."""
    model = _make_tiny(max_seq_len=16)
    model.eval()
    torch.manual_seed(42)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=6)
    assert out.shape == (2, 10)  # 4 + 6


def test_invariant_generate_tokens_within_vocab():
    """Generated tokens must be in [0, vocab_size)."""
    model = _make_tiny(vocab_size=32, max_seq_len=32)
    model.eval()
    torch.manual_seed(7)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=10)
    generated = out[:, 3:]
    assert generated.min().item() >= 0
    assert generated.max().item() < 32


def test_invariant_generate_preserves_prompt():
    """The first T tokens in the output must match the original input_ids."""
    model = _make_tiny(max_seq_len=16)
    model.eval()
    torch.manual_seed(5)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    out = model.generate(input_ids, max_new_tokens=5)
    assert torch.equal(out[:, :4], input_ids)


def test_invariant_generate_clips_when_capacity_insufficient():
    """Line 206: n = min(max_new_tokens, max_seq_len - T) — clips to remaining capacity."""
    model = _make_tiny(max_seq_len=8)
    model.eval()
    torch.manual_seed(3)
    input_ids = torch.zeros(1, 6, dtype=torch.long)
    # max capacity = 8-6=2, but we ask for 100
    out = model.generate(input_ids, max_new_tokens=100)
    assert out.shape == (1, 8)  # clipped to 6+2=8


def test_invariant_generate_returns_input_when_no_capacity():
    """Lines 207-208: when T >= max_seq_len, generate returns input unchanged."""
    model = _make_tiny(max_seq_len=4)
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=5)
    assert torch.equal(out, input_ids)


def test_invariant_generate_returns_input_when_max_new_tokens_zero():
    """n <= 0 path: max_new_tokens=0 also returns input unchanged (line 207-208)."""
    model = _make_tiny(max_seq_len=8)
    model.eval()
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=0)
    assert torch.equal(out, input_ids)


def test_invariant_generate_num_return_sequences_replicates_input():
    """Line 203-204: num_return_sequences > 1 tiles the batch."""
    model = _make_tiny(max_seq_len=16)
    model.eval()
    torch.manual_seed(11)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=4, num_return_sequences=3)
    # 1 prompt * 3 sequences = batch size 3; prompt + 4 new tokens = 7
    assert out.shape == (3, 7)


def test_invariant_generate_num_return_sequences_one_unchanged():
    """num_return_sequences=1 (default) does not replicate the batch."""
    model = _make_tiny(max_seq_len=16)
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=3, num_return_sequences=1)
    assert out.shape == (2, 7)


# ---------------------------------------------------------------------------
# TinyCausalLM.num_parameters
# ---------------------------------------------------------------------------

def test_invariant_num_parameters_positive():
    """Line 216: num_parameters returns a positive integer."""
    model = _make_tiny()
    n = model.num_parameters()
    assert isinstance(n, int)
    assert n > 0


def test_invariant_num_parameters_excludes_frozen():
    """Freezing all params makes num_parameters() return 0."""
    model = _make_tiny()
    for p in model.parameters():
        p.requires_grad = False
    assert model.num_parameters() == 0


def test_invariant_num_parameters_tied_weights_counts_once():
    """With tied weights (default), lm_head shares tok_emb.weight — the shared
    parameter is counted only once by iterating model.parameters()."""
    model_tied = _make_tiny(tie_weights=True)
    model_untied = _make_tiny(tie_weights=False)
    # Tied model has fewer unique parameter tensors → smaller count
    assert model_tied.num_parameters() < model_untied.num_parameters()


# ---------------------------------------------------------------------------
# Tied-weight invariant
# ---------------------------------------------------------------------------

def test_invariant_tied_weights_share_data():
    """Default tie_weights=True: lm_head.weight is tok_emb.weight."""
    model = _make_tiny(tie_weights=True)
    assert model.lm_head.weight is model.tok_emb.weight


def test_invariant_untied_weights_are_independent():
    """tie_weights=False: lm_head.weight is a separate tensor."""
    model = _make_tiny(tie_weights=False)
    assert model.lm_head.weight is not model.tok_emb.weight


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_invariant_model_registered_in_registry(clean_registry):
    """@register('model', 'tiny_lm') must make it discoverable via the registry."""
    from lighttrain.registry import get_registry
    reg = get_registry()
    cls = reg.get("model", "tiny_lm")
    assert cls is TinyCausalLM


# ---------------------------------------------------------------------------
# Determinism with manual seed
# ---------------------------------------------------------------------------

def test_invariant_generate_deterministic_with_same_seed():
    """Same manual seed produces identical generate() output."""
    model = _make_tiny(max_seq_len=16)
    model.eval()
    input_ids = torch.zeros(1, 4, dtype=torch.long)

    torch.manual_seed(123)
    out1 = model.generate(input_ids, max_new_tokens=5)
    torch.manual_seed(123)
    out2 = model.generate(input_ids, max_new_tokens=5)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# _init_weights branches (coverage via construction)
# ---------------------------------------------------------------------------

def test_invariant_init_weights_layernorm_ones_weight_zeros_bias():
    """LayerNorm weight initialized to 1 and bias to 0 by _init_weights."""
    model = _make_tiny()
    for module in model.modules():
        import torch.nn as nn
        if isinstance(module, nn.LayerNorm):
            assert torch.all(module.weight == 1.0).item()
            assert torch.all(module.bias == 0.0).item()


# ---------------------------------------------------------------------------
# forward with attention_mask (non-trivial)
# ---------------------------------------------------------------------------

def test_invariant_forward_with_padding_mask_does_not_crash():
    """Passing an attention_mask with zeros (padding) to forward() succeeds."""
    model = _make_tiny()
    model.eval()
    B, T = 2, 6
    input_ids = torch.randint(0, 32, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -2:] = 0  # last 2 tokens padded in first example
    out = model(input_ids=input_ids, attention_mask=mask)
    assert out.outputs["logits"].shape == (B, T, 32)


def test_invariant_forward_with_padding_mask_and_attentions():
    """Padding mask is correctly propagated in the explicit attention path."""
    model = _make_tiny(n_layers=1, n_heads=4)
    model.eval()
    B, T = 1, 5
    input_ids = torch.randint(0, 32, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -1] = 0
    out = model(input_ids=input_ids, attention_mask=mask, output_attentions=True)
    assert out.attentions is not None
    # Padded position should have 0 attention weight (it's masked to -inf)
    attn = out.attentions[0]  # (B, n_heads, T, T)
    assert attn[0, :, :, -1].abs().max().item() < 1e-5
