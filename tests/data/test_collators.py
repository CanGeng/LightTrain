"""Adversarial tests for ``lighttrain.builtin_plugins.data.collators.text``.

The collator's correctness is load-bearing for the whole training stack:
a wrong pad direction or a mis-typed attention mask silently breaks the
attention computation downstream. Legacy tests cover shapes and sums; this
file pins the value-level contract.

Coverage:

* **Right-pad direction**: legacy asserts shape; we assert which positions
  are real (1) vs pad (0) via ``assert_close`` on the mask.
* **Attention-mask dtype is torch.long** (current contract; pinned).
* **Labels = -100 on pad positions** via ``assert_close`` against a hand-
  crafted target tensor, not just ``.all()``.
* **Truncation at max_len** when a sample is longer.
* **Empty batch raises ValueError** (legacy doesn't pin this).
* **Labels fallback**: when sample has no ``"labels"`` key, ``input_ids``
  is used as labels.
* **Aux .hidden_states_layers transposed** from per-sample ``(L, T, H)`` to
  batch ``(L, B, T, H)`` — this is the highest-value collator invariant.
* **Aux .attentions_layers transposed** from per-sample ``(L, H, T, T)`` to
  batch ``(L, B, H, T, T)``.
* **Variable-shape aux silently skipped** (stack fails → key dropped).
* **None aux values silently skipped** (line 65-66 of collators.py).
* **PreferenceCollator pads chosen + rejected independently**.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.data.collators.text import (
    CausalLMCollator,
    PreferenceCollator,
)
from lighttrain.builtin_plugins.data.core.tokenizers import PAD_ID

# ---------------------------------------------------------------------------
# CausalLMCollator — pad direction, mask, labels
# ---------------------------------------------------------------------------

def test_invariant_collator_right_pads_to_max_length():
    """Right-pad invariant: the shorter sample has real tokens in positions
    [0, len-1] and ``pad_id`` in positions [len, max_len-1].

    Setup: two samples of length 3 and 5.
    Expected: input_ids[0] == [t0, t1, t2, pad, pad] (closed form).
    """
    samples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8]},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)

    expected_row0 = torch.tensor([1, 2, 3, PAD_ID, PAD_ID], dtype=torch.long)
    torch.testing.assert_close(batch["input_ids"][0], expected_row0)


def test_invariant_attention_mask_is_one_for_real_zero_for_pad():
    """Mask invariant: 1 on real tokens, 0 on pad positions.

    Closed form: row 0 of length 3 in a batch padded to 5 → mask=[1,1,1,0,0].
    """
    samples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8]},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    expected_mask_row0 = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)
    torch.testing.assert_close(batch["attention_mask"][0], expected_mask_row0)


def test_invariant_attention_mask_dtype_is_long():
    """Pin: attention_mask is torch.long (catches dtype regressions that
    silently break attention casting downstream).

    If you intentionally switch to torch.bool or torch.float, update this
    test AND verify every downstream consumer handles the new dtype.
    """
    samples = [{"input_ids": [1, 2, 3]}]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert batch["attention_mask"].dtype == torch.long


def test_invariant_input_ids_and_labels_dtype_are_long():
    """Pin: input_ids and labels are torch.long.

    Loss CE expects long-dtype labels; if dtype regresses, CE silently
    casts or raises.
    """
    samples = [{"input_ids": [1, 2, 3]}]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_invariant_labels_minus_100_on_pad_positions_exact():
    """Closed form: labels match input_ids on real positions; ``-100`` on
    pad positions.

    Setup: row 0 ids=[10, 20]; max in batch 4; row 0 labels expected =
    ``[10, 20, -100, -100]``.
    """
    samples = [
        {"input_ids": [10, 20]},
        {"input_ids": [30, 40, 50, 60]},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    expected_labels_row0 = torch.tensor([10, 20, -100, -100], dtype=torch.long)
    torch.testing.assert_close(batch["labels"][0], expected_labels_row0)


def test_collator_caps_at_max_len_truncating_long_samples():
    """Sample longer than ``max_len`` is truncated to ``max_len``.

    Setup: 1 sample of length 10, ``max_len=4``.
    Expected: ``input_ids.shape[1] == 4`` AND values are the first 4 tokens.
    """
    samples = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=4)
    batch = coll(samples)
    assert batch["input_ids"].shape == (1, 4)
    torch.testing.assert_close(
        batch["input_ids"][0],
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
    )


def test_collator_empty_batch_raises_value_error():
    """``__call__([])`` raises ValueError with a descriptive message.

    Goal: catches a regression where empty samples produce a 0×0 tensor
    that would silently propagate downstream.
    """
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    with pytest.raises(ValueError) as exc:
        coll([])
    assert "empty" in str(exc.value).lower()


def test_collator_uses_input_ids_as_labels_when_labels_key_absent():
    """When sample dict has no ``"labels"`` key, ``input_ids`` is used as
    the label source (line 49 fallback).

    Setup: sample with only ``input_ids``; verify labels equals input_ids
    on real positions.
    """
    samples = [{"input_ids": [11, 22, 33]}]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    # row 0: labels[:3] == input_ids[:3] (the explicit fallback path)
    torch.testing.assert_close(
        batch["labels"][0, :3],
        torch.tensor([11, 22, 33], dtype=torch.long),
    )


def test_collator_label_ignore_value_configurable():
    """``label_ignore`` parameter controls the pad-position label value.

    Setup: ``label_ignore=-1``; sample shorter than batch max.
    Expected: pad positions in labels are -1 (not -100).
    """
    samples = [
        {"input_ids": [1, 2]},
        {"input_ids": [3, 4, 5, 6]},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64, label_ignore=-1)
    batch = coll(samples)
    expected_row0 = torch.tensor([1, 2, -1, -1], dtype=torch.long)
    torch.testing.assert_close(batch["labels"][0], expected_row0)


# ---------------------------------------------------------------------------
# Aux key stacking — the highest-value collator invariant
# ---------------------------------------------------------------------------

def test_invariant_aux_hidden_states_layers_transposed_to_L_B_T_H():
    """Invariant: producer stores per-sample hidden_states as ``(L, T, H)``;
    the loss expects batch shape ``(L, B, T, H)``. ``torch.stack(...,dim=0)``
    yields ``(B, L, T, H)``, so the collator MUST transpose dim 0 and 1.

    Closed-form setup: two samples each with ``aux.foo.hidden_states_layers``
    of shape (L=3, T=2, H=4); compute the expected via manual stack +
    transpose.
    """
    L, T, H = 3, 2, 4
    s0 = torch.randn(L, T, H)
    s1 = torch.randn(L, T, H)
    samples = [
        {"input_ids": [1, 2], "aux.foo.hidden_states_layers": s0},
        {"input_ids": [3, 4], "aux.foo.hidden_states_layers": s1},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)

    # Final shape (L, B, T, H), NOT (B, L, T, H)
    got = batch["aux.foo.hidden_states_layers"]
    assert got.shape == (L, 2, T, H), f"expected (L, B, T, H)=(3,2,2,4); got {got.shape}"

    # Value check: equals the manual transpose
    expected = torch.stack([s0, s1], dim=0).transpose(0, 1).contiguous()
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_invariant_aux_attentions_layers_transposed_to_L_B_H_T_T():
    """Invariant: ``aux.*.attentions_layers`` is 5-D per sample
    ``(L, H, T, T)``; the collator transposes dim 0 ↔ 1 to produce
    ``(L, B, H, T, T)``.

    Closed-form: two samples with shape (L=2, H=3, T=4, T=4).
    """
    L, H, T = 2, 3, 4
    s0 = torch.randn(L, H, T, T)
    s1 = torch.randn(L, H, T, T)
    samples = [
        {"input_ids": [1, 2], "aux.bar.attentions_layers": s0},
        {"input_ids": [3, 4], "aux.bar.attentions_layers": s1},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)

    got = batch["aux.bar.attentions_layers"]
    assert got.shape == (L, 2, H, T, T)
    expected = torch.stack([s0, s1], dim=0).transpose(0, 1).contiguous()
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_invariant_generic_aux_2d_stacks_along_batch_dim_zero():
    """Aux keys that are NOT ``hidden_states_layers``/``attentions_layers``
    just stack via ``torch.stack(dim=0)`` without transpose.

    Setup: two samples each with ``aux.misc.logprobs`` of shape (T=3,).
    Expected: batched shape (B=2, T=3).
    """
    t0 = torch.tensor([1.0, 2.0, 3.0])
    t1 = torch.tensor([4.0, 5.0, 6.0])
    samples = [
        {"input_ids": [1, 2, 3], "aux.misc.logprobs": t0},
        {"input_ids": [4, 5, 6], "aux.misc.logprobs": t1},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)

    got = batch["aux.misc.logprobs"]
    assert got.shape == (2, 3)
    expected = torch.stack([t0, t1], dim=0)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_collator_aux_with_variable_shape_silently_dropped():
    """When aux tensors across samples have incompatible shapes, the
    ``torch.stack`` call raises RuntimeError and the collator silently
    drops the key (line 72-73 of collators.py).

    Goal: pin this behavior — a regression that suddenly raises would
    break running training jobs.
    """
    s0 = torch.randn(3, 4)
    s1 = torch.randn(3, 5)  # mismatched last dim
    samples = [
        {"input_ids": [1], "aux.varshape": s0},
        {"input_ids": [2], "aux.varshape": s1},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    # Key is silently absent (NOT KeyError; the loop continued).
    assert "aux.varshape" not in batch


def test_collator_aux_none_values_silently_skipped():
    """If a sample has ``aux.X = None``, that sample is skipped for that
    aux key. If ALL samples have None, the key never appears in output.

    Setup: both samples have None for the aux key.
    Expected: the key is absent from the batch.
    """
    samples = [
        {"input_ids": [1, 2], "aux.maybe": None},
        {"input_ids": [3, 4], "aux.maybe": None},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert "aux.maybe" not in batch


def test_collator_aux_keys_sorted_in_output():
    """Output aux keys come from a sorted union (line 60 of collators.py).

    Goal: pin determinism — downstream code can't rely on insertion order.
    """
    t = torch.tensor([1.0])
    samples = [
        {"input_ids": [1], "aux.zebra": t, "aux.alpha": t, "aux.middle": t},
    ]
    coll = CausalLMCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    aux_keys = [k for k in batch.keys() if k.startswith("aux.")]
    assert aux_keys == sorted(aux_keys)


# ---------------------------------------------------------------------------
# PreferenceCollator
# ---------------------------------------------------------------------------

def test_preference_collator_pads_chosen_and_rejected_independently():
    """``chosen_*`` and ``rejected_*`` are padded to their own batch-max
    independently — a long chosen with short rejected (or vice versa)
    must not be cross-influenced.

    Setup: chosen lengths [3, 5]; rejected lengths [4, 2].
    Expected: chosen.shape[1] == 5, rejected.shape[1] == 4. Each side has
    correct attention_mask + label padding.
    """
    samples = [
        {
            "chosen_input_ids": [1, 2, 3], "chosen_labels": [1, 2, 3],
            "rejected_input_ids": [10, 20, 30, 40],
            "rejected_labels": [10, 20, 30, 40],
        },
        {
            "chosen_input_ids": [5, 6, 7, 8, 9], "chosen_labels": [5, 6, 7, 8, 9],
            "rejected_input_ids": [50, 60], "rejected_labels": [50, 60],
        },
    ]
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert batch["chosen_input_ids"].shape == (2, 5)
    assert batch["rejected_input_ids"].shape == (2, 4)
    # Row 0 chosen had length 3 → labels expected [1, 2, 3, -100, -100]
    torch.testing.assert_close(
        batch["chosen_labels"][0],
        torch.tensor([1, 2, 3, -100, -100], dtype=torch.long),
    )
    # Row 1 rejected had length 2 → labels [50, 60, -100, -100]
    torch.testing.assert_close(
        batch["rejected_labels"][1],
        torch.tensor([50, 60, -100, -100], dtype=torch.long),
    )


def test_preference_collator_empty_batch_raises():
    """``PreferenceCollator([])`` raises ValueError too."""
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=64)
    with pytest.raises(ValueError):
        coll([])


def test_preference_collator_aux_keys_preserved():
    """Aux keys flow through PreferenceCollator the same as CausalLMCollator.

    Setup: each sample has a non-aux preference key set plus
    ``aux.ref.chosen_logprobs``.
    Expected: aux key appears in the output dict with shape (B, *).
    """
    samples = [
        {
            "chosen_input_ids": [1, 2], "chosen_labels": [1, 2],
            "rejected_input_ids": [3, 4], "rejected_labels": [3, 4],
            "aux.ref.chosen_logprobs": torch.tensor([0.1, 0.2]),
        },
        {
            "chosen_input_ids": [5, 6], "chosen_labels": [5, 6],
            "rejected_input_ids": [7, 8], "rejected_labels": [7, 8],
            "aux.ref.chosen_logprobs": torch.tensor([0.3, 0.4]),
        },
    ]
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=64)
    batch = coll(samples)
    assert "aux.ref.chosen_logprobs" in batch
    assert batch["aux.ref.chosen_logprobs"].shape == (2, 2)


def test_preference_collator_registered_under_preference():
    """``PreferenceCollator`` is registered as ``('collator', 'preference')``.

    Goal: pin the registry name — recipes use ``collator.name = preference``
    (e.g. dpo_offline.yaml).
    """
    from lighttrain.registry import get
    assert get("collator", "preference") is PreferenceCollator


def test_preference_collator_emits_exact_output_key_set():
    """The batch dict carries exactly the six chosen_/rejected_ keys.

    Pin: ``{chosen,rejected}_{input_ids,attention_mask,labels}`` and nothing
    else when no aux keys are present — downstream DPO loss indexes these by
    name, so a renamed/dropped key must surface as a test failure.
    """
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=16)
    batch = coll([
        {
            "chosen_input_ids": list(range(5)), "chosen_labels": list(range(5)),
            "rejected_input_ids": list(range(7)), "rejected_labels": list(range(7)),
        }
    ])
    expected = {
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
    }
    assert set(batch.keys()) == expected


def test_preference_collator_truncates_both_sides_at_max_len():
    """Chosen and rejected are independently truncated to ``max_len``.

    Setup: chosen + rejected both length 10, ``max_len=4``.
    Expected: both sides have shape (1, 4).
    """
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=4)
    batch = coll([
        {
            "chosen_input_ids": list(range(10)), "chosen_labels": list(range(10)),
            "rejected_input_ids": list(range(10)), "rejected_labels": list(range(10)),
        }
    ])
    assert batch["chosen_input_ids"].shape == (1, 4)
    assert batch["rejected_input_ids"].shape == (1, 4)


def test_preference_collator_configurable_pad_id_and_ignore_index():
    """``pad_id`` fills input pad slots and ``ignore_index`` fills label pad
    slots — both configurable away from the defaults.

    Closed form: row 0 chosen length 3 in a batch padded to 5, ``pad_id=99``,
    ``ignore_index=-100`` → mask=[1,1,1,0,0]; input_ids[3]==99; labels[3]==-100.
    """
    coll = PreferenceCollator(pad_id=99, max_len=16, ignore_index=-100)
    samples = [
        {
            "chosen_input_ids": [1, 2, 3], "chosen_labels": [1, 2, 3],
            "rejected_input_ids": [1, 2, 3], "rejected_labels": [1, 2, 3],
        },
        {
            "chosen_input_ids": [4, 5, 6, 7, 8], "chosen_labels": [4, 5, 6, 7, 8],
            "rejected_input_ids": [4, 5, 6, 7, 8], "rejected_labels": [4, 5, 6, 7, 8],
        },
    ]
    batch = coll(samples)
    assert batch["chosen_attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["chosen_attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
    assert batch["chosen_input_ids"][0, 3].item() == 99
    assert batch["chosen_labels"][0, 3].item() == -100


def test_preference_collator_scalar_aux_logprobs_passthrough():
    """0-d (scalar) ``aux.ref.*`` logprobs survive collation, stacked to (B,).

    Pin (REVIEW_ROUND3 #3): per-sample scalar reference logprobs are stacked
    along the batch dim, not dropped. Distinct from the (B, T) token-level
    aux case already pinned in ``test_preference_collator_aux_keys_preserved``.
    """
    samples = [
        {
            "chosen_input_ids": [1, 2, 3], "chosen_labels": [1, 2, 3],
            "rejected_input_ids": [4, 5], "rejected_labels": [4, 5],
            "aux.ref.chosen_logprobs": torch.tensor(-1.5),
            "aux.ref.rejected_logprobs": torch.tensor(-2.0),
        },
        {
            "chosen_input_ids": [6, 7], "chosen_labels": [6, 7],
            "rejected_input_ids": [8, 9, 10], "rejected_labels": [8, 9, 10],
            "aux.ref.chosen_logprobs": torch.tensor(-1.2),
            "aux.ref.rejected_logprobs": torch.tensor(-1.8),
        },
    ]
    coll = PreferenceCollator(pad_id=PAD_ID, max_len=16)
    batch = coll(samples)
    assert batch["aux.ref.chosen_logprobs"].shape == (2,)
    assert batch["aux.ref.rejected_logprobs"].shape == (2,)
