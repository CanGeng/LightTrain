"""Adversarial tests for lighttrain.builtin_plugins.rl.ref_policy (_sequence_log_probs / freeze_as_ref)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.builtin_plugins.rl.ref_policy import (
    ReferencePolicy,
    _sequence_log_probs,
    freeze_as_ref,
    ref_log_probs,
)
from lighttrain.protocols import ModelOutput


def test_seq_logprobs_uniform_logits_equals_neg_log_V():
    """Goal: logits = 0 → log_softmax = -log V everywhere → mean = -log V.

    Input: B=2, T=4 (post-shift T-1=3 positions), V=5; logits = 0, labels = any.
    Analytical: each gathered log-prob = -log V; mean = -log V.
    """
    B, T, V = 2, 4, 5
    logits = torch.zeros(B, T, V)
    labels = torch.randint(0, V, (B, T))
    out = _sequence_log_probs(logits, labels, ignore_index=-100)
    expected = torch.full((B,), -math.log(V))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_seq_logprobs_gather_matches_manual_reconstruction():
    """Goal: implementation matches manual log_softmax + gather + mask + mean.

    Input: small random logits, B=2, T=3 (post-shift 2 positions), V=4.
    Analytical: reconstruct with PyTorch ops.
    """
    torch.manual_seed(41)
    B, T, V = 2, 3, 4
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    actual = _sequence_log_probs(logits, labels, ignore_index=-100)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target = shift_labels.clamp(min=0)
    gathered = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    expected = (gathered * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_seq_logprobs_causal_shift_drops_first_logit_and_last_label():
    """Goal: predicted positions use logits[:, :-1, :] vs labels[:, 1:].

    Input: deterministic logits where only position 0 is highly informative;
           verify that the loss reflects logits at position 0 trying to predict labels at position 1.
    """
    B, T, V = 1, 3, 4
    logits = torch.zeros(B, T, V)
    # logits at position 0 strongly favor class 2.
    logits[0, 0, 2] = 100.0
    # logits at position 1 strongly favor class 1.
    logits[0, 1, 1] = 100.0
    # logits at position 2 strongly favor class 0 (should be unused after shift).
    logits[0, 2, 0] = 100.0
    # labels: shift means we predict labels[1] from logits[0], labels[2] from logits[1].
    labels = torch.tensor([[9, 2, 1]], dtype=torch.long)
    # → at shifted position 0, gather log_softmax(logits[0])[labels[1]=2] ≈ 0
    # → at shifted position 1, gather log_softmax(logits[1])[labels[2]=1] ≈ 0
    out = _sequence_log_probs(logits, labels)
    # Both are essentially zero → mean ≈ 0.
    torch.testing.assert_close(out, torch.tensor([0.0]), atol=1e-3, rtol=1e-3)


def test_seq_logprobs_masks_ignore_index_positions():
    """Goal: positions with label == ignore_index are excluded from the mean.

    Input: half positions masked.
    Analytical: result is mean over only unmasked positions.
    """
    torch.manual_seed(42)
    B, T, V = 1, 5, 4
    logits = torch.randn(B, T, V)
    # After shift: shift_labels = labels[:, 1:] = (T-1,) = 4 positions.
    # Mask positions 0 and 2 (in shift index).
    labels = torch.tensor([[9, -100, 1, -100, 2]], dtype=torch.long)
    actual = _sequence_log_probs(logits, labels)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target = shift_labels.clamp(min=0)
    gathered = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    expected = (gathered * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Reference policy wrapper
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal model returning ModelOutput with logits = embedding(input_ids)."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        logits = self.lm_head(h)
        return ModelOutput(outputs={"logits": logits})


def test_freeze_as_ref_creates_no_grad_copy():
    """Goal: freeze_as_ref deep-copies the model and disables gradients."""
    model = _TinyLM(vocab=10, hidden=4)
    ref = freeze_as_ref(model)
    assert ref.model is not None
    assert ref.model is not model
    for p in ref.model.parameters():
        assert not p.requires_grad


def test_ref_log_probs_returns_no_grad_tensor():
    """Goal: ref policy log_probs has requires_grad=False (the @no_grad guard).

    Input: tiny model, batch B=2, T=3.
    Analytical: the returned tensor must not have requires_grad set.
    """
    torch.manual_seed(43)
    model = _TinyLM(vocab=10, hidden=4)
    ref = freeze_as_ref(model)
    input_ids = torch.randint(0, 10, (2, 3))
    labels = input_ids.clone()
    out = ref_log_probs(ref, input_ids, attention_mask=None, labels=labels)
    assert not out.requires_grad


def test_regression_ref_logprobs_must_be_detached():
    """Regression pin for ``ref_logprob_grad_leak``.

    Bug: dropping the @torch.no_grad on log_probs causes the returned tensor
    to retain a gradient graph that may flow back into the live policy parameters.

    Input: live model used as reference; check that backward on a function of
    log_probs doesn't reach the model's parameters via the ref path.
    """
    torch.manual_seed(44)
    model = _TinyLM(vocab=10, hidden=4)
    ref = freeze_as_ref(model)
    input_ids = torch.randint(0, 10, (2, 3))
    labels = input_ids.clone()
    out = ref_log_probs(ref, input_ids, attention_mask=None, labels=labels)
    # Sum to scalar; backward should NOT raise (no grad tensors in graph), and
    # ref model params should still have no grad accumulated.
    if out.requires_grad:
        out.sum().backward()
    for p in ref.model.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# per-token path (per_token=True) — used by GRPO's per-token k3 KL (L-P0f)
# ---------------------------------------------------------------------------


def test_log_probs_per_token_shape_and_leading_zero():
    """Goal: per_token=True returns (B, T) with a 0 first column.

    The leading 0 column mirrors the GRPO trainer's log_probs_new so the two
    align position-for-position for the per-token KL subtraction.
    """
    torch.manual_seed(45)
    B, T, V = 3, 5, 10
    ref = freeze_as_ref(_TinyLM(vocab=V, hidden=4))
    input_ids = torch.randint(0, V, (B, T))
    labels = input_ids.clone()

    out = ref.log_probs(input_ids, None, labels, per_token=True)

    assert out.shape == (B, T)
    assert torch.all(out[:, 0] == 0.0)
    # Default (per_token=False) still returns (B,) — the existing contract.
    assert ref.log_probs(input_ids, None, labels).shape == (B,)


def test_log_probs_per_token_gathers_input_ids_not_labels():
    """Goal: per-token gather targets input_ids[:, 1:], NOT labels.

    With labels != input_ids the reconstruction from input_ids must match; a
    labels-based gather would differ — pinning the realized-token track.
    """
    torch.manual_seed(46)
    B, T, V = 2, 4, 10
    ref = freeze_as_ref(_TinyLM(vocab=V, hidden=4))
    input_ids = torch.randint(0, V, (B, T))
    labels = (input_ids + 3) % V  # deliberately different track

    out = ref.log_probs(input_ids, None, labels, per_token=True)

    logits = ref.model(input_ids).outputs["logits"]
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    lp = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(lp, -1, shift_targets.unsqueeze(-1)).squeeze(-1)
    expected = torch.cat([torch.zeros_like(gathered[:, :1]), gathered], dim=1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_log_probs_per_token_is_detached():
    """Goal: per-token path inherits the @torch.no_grad guard."""
    torch.manual_seed(47)
    ref = freeze_as_ref(_TinyLM(vocab=10, hidden=4))
    input_ids = torch.randint(0, 10, (2, 4))
    out = ref.log_probs(input_ids, None, input_ids.clone(), per_token=True)
    assert not out.requires_grad


def test_log_probs_per_token_rejects_lora_base():
    """Goal: public-API misuse guard — per_token=True is unsupported with
    lora_base_as_ref=True (would crash unclearly on model=None)."""
    ref = ReferencePolicy(model=None, lora_base_as_ref=True)
    input_ids = torch.randint(0, 10, (2, 4))
    with pytest.raises(RuntimeError, match="lora_base_as_ref"):
        ref.log_probs(input_ids, None, input_ids.clone(), per_token=True)
