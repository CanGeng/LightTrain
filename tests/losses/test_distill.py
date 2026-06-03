"""Adversarial tests for lighttrain.builtin_plugins.losses.distill.

Covers KLDivLoss / HiddenStatesMSELoss / HiddenStatesCosineLoss /
AttentionTransferLoss with closed-form assertions and regression pins.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.distill import (
    AttentionTransferLoss,
    HiddenStatesCosineLoss,
    HiddenStatesMSELoss,
    KLDivLoss,
)
from lighttrain.protocols import LossContext, ModelOutput


def _make_topk_batch(B, T, K, V, teacher_logits, student_indices_pick_topk=False):
    """Build a batch with aux.teacher.logits_topk_64.{values,indices}.

    teacher_logits: (B, T, V) — full teacher logits; we extract top-K from these.
    """
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    labels = torch.zeros(B, T, dtype=torch.long)
    return {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# KLDivLoss
# ---------------------------------------------------------------------------


def test_kl_zero_when_student_logits_match_teacher_on_topk(dummy_ctx):
    """Goal: student logits identical to teacher at top-K positions → KL = 0.

    Input: full vocab logits with student = teacher in the top-K columns.
           K=4, V=8, B=1, T=2, τ=2.
    Analytical: log_softmax(s/τ) == log_softmax(t/τ) → KL = 0 per token.
    """
    B, T, K, V = 1, 2, 4, 8
    torch.manual_seed(11)
    teacher_logits = torch.randn(B, T, V)
    # Build student logits with the same values at teacher top-K positions
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.zeros(B, T, V)
    student_logits.scatter_(-1, idx, vals)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),  # labels not -100 → counted
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out = KLDivLoss(temperature=2.0, top_k=K)(mo, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_kl_matches_manual_F_kl_div_closed_form(dummy_ctx):
    """Goal: loss matches F.kl_div(log_p_student, log_p_teacher, log_target=True)
            summed over K and scaled by τ², averaged over non-ignored tokens.

    Input: small random student/teacher logits with τ=1.5.
    Analytical: hand-reconstruct the formula with F.kl_div and compare.
    """
    torch.manual_seed(12)
    B, T, K, V = 2, 3, 4, 10
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    tau = 1.5
    out = KLDivLoss(temperature=tau, top_k=K)(mo, batch, dummy_ctx)

    # Manual reconstruction.
    s_topk = torch.gather(student_logits, dim=-1, index=idx)
    log_p_s = F.log_softmax(s_topk / tau, dim=-1)
    log_p_t = F.log_softmax(vals / tau, dim=-1)
    per_token = F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True).sum(-1) * (tau * tau)
    # All labels are 0 (not -100) → mask all True → mean over B·T tokens.
    expected = per_token.mean()
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_kl_temperature_squared_scaling_at_matched_post_temp_logits(dummy_ctx):
    """Goal: scaling both student and teacher logits by τ leaves softmax(scaled/τ)
            invariant; the τ² factor then makes the loss scale exactly as τ².

    Input: build (s, t) at τ=1 reference, then at τ=2 use (2·s, 2·t).
    Analytical: KL_softmax part is identical (since (2·logits)/2 == logits/1).
                Loss ratio = (2)² / (1)² = 4.
    """
    torch.manual_seed(13)
    B, T, K, V = 1, 1, 3, 5
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch1 = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    batch2 = {
        "aux.teacher.logits_topk_64.values": vals * 2,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo1 = ModelOutput(outputs={"logits": student_logits})
    mo2 = ModelOutput(outputs={"logits": student_logits * 2})
    out1 = KLDivLoss(temperature=1.0, top_k=K)(mo1, batch1, dummy_ctx)
    out2 = KLDivLoss(temperature=2.0, top_k=K)(mo2, batch2, dummy_ctx)
    # softmax part identical; the τ² scaling makes loss2 = 4 · loss1.
    torch.testing.assert_close(out2["loss"], 4.0 * out1["loss"], atol=1e-5, rtol=1e-4)


def test_kl_mask_ignore_index_excludes_token_quantitatively(dummy_ctx):
    """Goal: a label of -100 excludes that position; the loss equals the
            mean over the remaining positions only.

    Input: B=1, T=4; mask positions 0 and 2 (set labels=-100).
    Analytical: reconstruct expected with mean over the unmasked subset.
    """
    torch.manual_seed(14)
    B, T, K, V = 1, 4, 3, 6
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    labels = torch.tensor([[-100, 1, -100, 1]], dtype=torch.long)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": labels,
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out = KLDivLoss(temperature=2.0, top_k=K)(mo, batch, dummy_ctx)

    # Manual reconstruction over only positions 1 and 3.
    tau = 2.0
    s_topk = torch.gather(student_logits, dim=-1, index=idx)
    log_p_s = F.log_softmax(s_topk / tau, dim=-1)
    log_p_t = F.log_softmax(vals / tau, dim=-1)
    per_token = F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True).sum(-1) * (tau * tau)
    mask = labels != -100
    expected = (per_token * mask.float()).sum() / mask.sum()
    torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-4)


def test_kl_reduction_sum_matches_mean_times_count(dummy_ctx):
    """Goal: reduction='sum' equals mean times the denominator.

    Input: same batch with both reduction modes.
    Analytical: mean = sum / unmasked_count.
    """
    torch.manual_seed(15)
    B, T, K, V = 1, 2, 3, 5
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out_mean = KLDivLoss(temperature=1.0, top_k=K, reduction="mean")(mo, batch, dummy_ctx)
    out_sum = KLDivLoss(temperature=1.0, top_k=K, reduction="sum")(mo, batch, dummy_ctx)
    torch.testing.assert_close(
        out_sum["loss"], out_mean["loss"] * (B * T), atol=1e-5, rtol=1e-4
    )


def test_kl_direction_teacher_to_student_not_reversed(dummy_ctx):
    """Goal: loss matches KL(teacher || student), not KL(student || teacher).

    Input: a clearly asymmetric pair (teacher concentrated, student diffuse).
    Analytical: KL(teacher || student) and KL(student || teacher) differ for
                non-symmetric distributions; we pin the teacher→student direction
                (the one the implementation uses).
    """
    torch.manual_seed(16)
    B, T, K, V = 1, 1, 3, 5
    teacher_logits = torch.zeros(B, T, V)
    teacher_logits[..., 0] = 10.0  # very peaked
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.zeros(B, T, V)  # uniform
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out = KLDivLoss(temperature=1.0, top_k=K)(mo, batch, dummy_ctx)

    s_topk = torch.gather(student_logits, dim=-1, index=idx)
    log_p_s = F.log_softmax(s_topk, dim=-1)
    log_p_t = F.log_softmax(vals, dim=-1)
    p_t = log_p_t.exp()
    p_s = log_p_s.exp()
    kl_teacher_to_student = (p_t * (log_p_t - log_p_s)).sum(-1).mean() * 1.0
    kl_student_to_teacher = (p_s * (log_p_s - log_p_t)).sum(-1).mean() * 1.0
    torch.testing.assert_close(
        out["loss"], kl_teacher_to_student, atol=1e-5, rtol=1e-4
    )
    # And the two directions must be detectably different.
    assert abs(float(kl_teacher_to_student) - float(kl_student_to_teacher)) > 0.1


def test_kl_topk_gather_isolates_at_teacher_indices(dummy_ctx):
    """Goal: changes to student logits at positions NOT in teacher top-K
            do not change the loss (the gather strips them out).

    Input: identical student/teacher on top-K positions; modify student's
           non-topk columns and verify loss is unchanged.
    """
    torch.manual_seed(17)
    B, T, K, V = 1, 2, 3, 8
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    student_modified = student_logits.clone()
    # Change positions NOT in idx; if all positions are in idx (unlikely K<V), skip.
    not_idx_mask = torch.ones(B, T, V, dtype=torch.bool)
    not_idx_mask.scatter_(-1, idx, False)
    student_modified = student_modified + 100.0 * not_idx_mask.float()
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo_a = ModelOutput(outputs={"logits": student_logits})
    mo_b = ModelOutput(outputs={"logits": student_modified})
    out_a = KLDivLoss(temperature=1.5, top_k=K)(mo_a, batch, dummy_ctx)
    out_b = KLDivLoss(temperature=1.5, top_k=K)(mo_b, batch, dummy_ctx)
    torch.testing.assert_close(out_a["loss"], out_b["loss"], atol=1e-5, rtol=1e-4)


def test_regression_kl_temperature_not_linear(dummy_ctx):
    """Regression pin for ``kl_temperature_linear``.

    Bug: using τ instead of τ² as the scale factor changes the loss linearly
    in τ instead of quadratically.

    Input: same logits, τ=1 vs τ=3.
    Analytical: correct ratio = 3²/1² = 9; bug ratio = 3.
    To make this robust we use the matched-scale construction
    (s, t at τ=1) vs (3s, 3t at τ=3): softmax parts equal → ratio = τ².
    """
    torch.manual_seed(18)
    B, T, K, V = 1, 1, 3, 5
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch1 = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    batch3 = {
        "aux.teacher.logits_topk_64.values": vals * 3.0,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo1 = ModelOutput(outputs={"logits": student_logits})
    mo3 = ModelOutput(outputs={"logits": student_logits * 3.0})
    out1 = KLDivLoss(temperature=1.0, top_k=K)(mo1, batch1, dummy_ctx)
    out3 = KLDivLoss(temperature=3.0, top_k=K)(mo3, batch3, dummy_ctx)
    torch.testing.assert_close(out3["loss"], 9.0 * out1["loss"], atol=1e-5, rtol=1e-4)
    # Defensive: ratio 9 (τ²) vs 3 (τ-linear) is a 6x gap, far above tolerance.


# ---------------------------------------------------------------------------
# HiddenStatesMSELoss
# ---------------------------------------------------------------------------


def _make_mo_with_hidden(hidden_states):
    return ModelOutput(outputs={}, hidden_states=tuple(hidden_states))


def test_hidden_mse_zero_when_student_equals_teacher(dummy_ctx):
    """Goal: identical student / teacher hidden states → loss = 0.

    Input: random hidden states, two layers; mapping {0:0, 1:1}.
    Analytical: MSE on equal tensors = 0.
    """
    B, T, H = 2, 3, 4
    s0 = torch.randn(B, T, H)
    s1 = torch.randn(B, T, H)
    teacher = torch.stack([s0, s1], dim=0)  # (L, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s0, s1])
    loss = HiddenStatesMSELoss(mapping={0: 0, 1: 1})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_hidden_mse_known_quadratic_closed_form(dummy_ctx):
    """Goal: per-layer MSE = mean((s - t)²) over (B,T,H); total averaged over layers.

    Input: single layer, student=0, teacher=1 everywhere; B=2, T=3, H=4.
    Analytical: (0-1)² = 1 everywhere → mean = 1.0; reduction='mean' over 1 layer → 1.0.
    """
    B, T, H = 2, 3, 4
    s = torch.zeros(B, T, H)
    t = torch.ones(1, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": t,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesMSELoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_hidden_mse_layer_mapping_uses_correct_indices(dummy_ctx):
    """Goal: mapping {1: 2} picks student[1] vs teacher[2], not [0] vs [0].

    Input: student has 2 layers (idx 1 is the target), teacher has 3 layers.
           Make student[1] == teacher[2] but student[0] != teacher[0].
    Analytical: mapping {1:2} → loss = 0; mapping {0:0} → loss > 0.
    """
    B, T, H = 1, 2, 3
    s0 = torch.ones(B, T, H)
    s1 = torch.zeros(B, T, H)  # matches teacher[2]
    t0 = torch.full((B, T, H), 5.0)  # very different
    t1 = torch.full((B, T, H), 7.0)
    t2 = torch.zeros(B, T, H)  # matches s1
    teacher = torch.stack([t0, t1, t2], dim=0)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s0, s1])
    loss_correct = HiddenStatesMSELoss(mapping={1: 2})(mo, batch, dummy_ctx)
    loss_wrong_idx = HiddenStatesMSELoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(
        loss_correct["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4
    )
    # Wrong layer mapping → (1-5)² = 16
    torch.testing.assert_close(
        loss_wrong_idx["loss"], torch.tensor(16.0), atol=1e-5, rtol=1e-4
    )


def test_hidden_mse_layer_mapping_dict_vs_list_equivalent(dummy_ctx):
    """Goal: LayerMapping accepts dict and list-of-pairs and they behave identically.

    Input: same data, mapping as {0:0, 1:1} and as [(0,0),(1,1)].
    Analytical: numerical loss must be identical.
    """
    B, T, H = 1, 2, 3
    s = [torch.randn(B, T, H), torch.randn(B, T, H)]
    teacher = torch.stack(s, dim=0) + 0.5
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden(s)
    loss_dict = HiddenStatesMSELoss(mapping={0: 0, 1: 1})(mo, batch, dummy_ctx)
    loss_list = HiddenStatesMSELoss(mapping=[(0, 0), (1, 1)])(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss_dict["loss"], loss_list["loss"], atol=1e-5, rtol=1e-4)


def test_hidden_mse_projection_zeros_init_initial_loss_equals_teacher_mse_with_zero():
    """Goal: zeros init projection → s_projected = 0 → loss = mean(t²).

    Input: student dim H_s=3, teacher dim H_t=5, project_init='zeros'.
           teacher_layer = ones → mean((0 - 1)²) = 1.0.
    """
    B, T = 1, 2
    Hs, Ht = 3, 5
    s = torch.randn(B, T, Hs)
    teacher = torch.ones(1, B, T, Ht)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    host = nn.Module()  # surgery target
    ctx = LossContext(extras={"model": host})
    loss_fn = HiddenStatesMSELoss(
        mapping={0: 0}, project=True, project_init="zeros", project_bias=False
    )
    loss = loss_fn(mo, batch, ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_hidden_mse_invalid_layer_index_raises(dummy_ctx):
    """Goal: a mapping referring to a layer index out of range must raise,
            not silently skip.
    """
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    teacher = torch.zeros(2, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    # student has 1 layer but mapping references idx 5 — must raise.
    with pytest.raises(IndexError):
        HiddenStatesMSELoss(mapping={5: 0})(mo, batch, dummy_ctx)


def test_hidden_mse_dim_mismatch_without_projection_raises(dummy_ctx):
    """Goal: mismatched hidden dims with project=False → explicit RuntimeError."""
    B, T = 1, 2
    s = torch.zeros(B, T, 3)
    teacher = torch.zeros(1, B, T, 5)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    with pytest.raises(RuntimeError, match="hidden dim mismatch"):
        HiddenStatesMSELoss(mapping={0: 0}, project=False)(mo, batch, dummy_ctx)


def test_regression_hidden_mse_mean_not_sum(dummy_ctx):
    """Regression pin for ``hidden_mse_sum_instead_of_mean``.

    Bug: a refactor that sums (instead of means) across hidden / sequence /
    batch dims yields B·T·H times the correct value.

    Input: B=2, T=3, H=4, s=0, t=1 → correct mean = 1.0.
           Sum would give 2·3·4 = 24 (or some multiple). Verify exactly 1.0.
    """
    B, T, H = 2, 3, 4
    s = torch.zeros(B, T, H)
    t = torch.ones(1, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": t,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesMSELoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# HiddenStatesCosineLoss
# ---------------------------------------------------------------------------


def test_hidden_cosine_orthogonal_vectors_loss_one(dummy_ctx):
    """Goal: orthogonal vectors → cosine = 0 → loss = 1.

    Input: student = [1,0,0], teacher = [0,1,0] at every (B,T).
    Analytical: 1 - 0 = 1.
    """
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    s[..., 0] = 1.0
    t_full = torch.zeros(1, B, T, H)
    t_full[..., 0, :, :, 1] = 1.0
    batch = {
        "aux.teacher.hidden_states_layers": t_full,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesCosineLoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_hidden_cosine_aligned_vectors_loss_zero(dummy_ctx):
    """Goal: parallel unit-norm vectors → cosine = 1 → loss = 0.

    Input: student = teacher = [1, 0, 0].
    """
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    s[..., 0] = 1.0
    t_full = torch.zeros(1, B, T, H)
    t_full[..., 0, :, :, 0] = 1.0
    batch = {
        "aux.teacher.hidden_states_layers": t_full,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesCosineLoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_hidden_cosine_antiparallel_vectors_loss_two(dummy_ctx):
    """Goal: antiparallel vectors → cosine = -1 → loss = 2.

    Input: student = [1,0,0], teacher = [-1,0,0].
    """
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    s[..., 0] = 1.0
    t_full = torch.zeros(1, B, T, H)
    t_full[..., 0, :, :, 0] = -1.0
    batch = {
        "aux.teacher.hidden_states_layers": t_full,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesCosineLoss(mapping={0: 0})(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)


def test_hidden_cosine_zero_vector_does_not_produce_nan(dummy_ctx):
    """Goal: zero vector input → cosine_similarity uses eps; loss stays finite.

    Input: student = 0 vector; teacher = unit vector.
    Analytical: with eps > 0, cos = 0 / max(eps, eps) = 0 → loss = 1 (finite).
    """
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    t_full = torch.zeros(1, B, T, H)
    t_full[..., 0, :, :, 0] = 1.0
    batch = {
        "aux.teacher.hidden_states_layers": t_full,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _make_mo_with_hidden([s])
    loss = HiddenStatesCosineLoss(mapping={0: 0})(mo, batch, dummy_ctx)
    assert torch.isfinite(loss["loss"])
    torch.testing.assert_close(loss["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# AttentionTransferLoss
# ---------------------------------------------------------------------------


def _mo_with_attn(attns):
    return ModelOutput(outputs={}, attentions=tuple(attns))


def test_attn_xfer_zero_when_normalized_attn_matches(dummy_ctx):
    """Goal: identical attention maps (post head-mean) → loss = 0."""
    B, Hheads, T = 1, 4, 3
    student_attn = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1)
    teacher_attn = student_attn.clone().unsqueeze(0)  # (L=1, B, H, T, T)
    batch = {"aux.teacher.attentions_layers": teacher_attn}
    mo = _mo_with_attn([student_attn])
    loss = AttentionTransferLoss(mapping={0: 0}, p=2.0)(mo, batch, dummy_ctx)
    torch.testing.assert_close(loss["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_attn_xfer_p_power_changes_loss(dummy_ctx):
    """Goal: changing p between 2 and 1 changes the loss for non-zero diffs.

    Input: student vs teacher attention differ.
    Analytical: |Δ|.pow(p).mean() differs across p; numerical value can be
                hand-checked via reconstruction.
    """
    torch.manual_seed(21)
    B, Hheads, T = 1, 2, 3
    s = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1)
    t = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1).unsqueeze(0)
    batch = {"aux.teacher.attentions_layers": t}
    mo = _mo_with_attn([s])
    loss_p2 = AttentionTransferLoss(mapping={0: 0}, p=2.0)(mo, batch, dummy_ctx)
    loss_p1 = AttentionTransferLoss(mapping={0: 0}, p=1.0)(mo, batch, dummy_ctx)

    # Manual reconstruction:
    s_mean = s.mean(dim=1)  # (B, T, T)
    t_mean = t[0].mean(dim=1).to(s.dtype)
    s_n = s_mean / s_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    t_n = t_mean / t_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    diff = (s_n - t_n).abs()
    expected_p2 = (diff.pow(2)).mean()
    expected_p1 = (diff.pow(1)).mean()
    torch.testing.assert_close(loss_p2["loss"], expected_p2, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(loss_p1["loss"], expected_p1, atol=1e-5, rtol=1e-4)


def test_attn_xfer_head_count_mismatch_averages_heads(dummy_ctx):
    """Goal: when H_s != H_t, both sides are averaged over heads before MSE.

    Input: student has 2 heads, teacher has 4 heads.
    Analytical: implementation does .mean(dim=1); compare against manual mean.
    """
    torch.manual_seed(22)
    B, T = 1, 3
    s = torch.softmax(torch.randn(B, 2, T, T), dim=-1)
    t = torch.softmax(torch.randn(B, 4, T, T), dim=-1).unsqueeze(0)
    batch = {"aux.teacher.attentions_layers": t}
    mo = _mo_with_attn([s])
    loss = AttentionTransferLoss(mapping={0: 0}, p=2.0)(mo, batch, dummy_ctx)
    s_mean = s.mean(dim=1)
    t_mean = t[0].mean(dim=1).to(s.dtype)
    s_n = s_mean / s_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    t_n = t_mean / t_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    expected = (s_n - t_n).abs().pow(2.0).mean()
    torch.testing.assert_close(loss["loss"], expected, atol=1e-5, rtol=1e-4)


def test_attn_xfer_row_normalization_makes_invariant_to_scale(dummy_ctx):
    """Goal: scaling the student attention map by a constant doesn't change loss
            (row-norm divides it out).
    """
    torch.manual_seed(23)
    B, Hheads, T = 1, 2, 3
    s_base = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1)
    t = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1).unsqueeze(0)
    batch = {"aux.teacher.attentions_layers": t}
    mo_a = _mo_with_attn([s_base])
    mo_b = _mo_with_attn([s_base * 7.5])
    loss_a = AttentionTransferLoss(mapping={0: 0})(mo_a, batch, dummy_ctx)
    loss_b = AttentionTransferLoss(mapping={0: 0})(mo_b, batch, dummy_ctx)
    torch.testing.assert_close(loss_a["loss"], loss_b["loss"], atol=1e-5, rtol=1e-4)


def test_regression_attn_xfer_normalization_applied_before_difference(dummy_ctx):
    """Regression pin for ``attn_xfer_normalization_order``.

    Bug: computing (s - t).normalize() instead of s.normalize() - t.normalize()
    changes the metric.

    Input: same non-trivial s, t.
    Analytical: the actual loss must match the formula with normalization
                applied first (per-side), independently of the difference.
    """
    torch.manual_seed(24)
    B, Hheads, T = 1, 1, 3
    s = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1)
    t = torch.softmax(torch.randn(B, Hheads, T, T), dim=-1).unsqueeze(0)
    batch = {"aux.teacher.attentions_layers": t}
    mo = _mo_with_attn([s])
    loss = AttentionTransferLoss(mapping={0: 0}, p=2.0)(mo, batch, dummy_ctx)

    s_mean = s.mean(dim=1)
    t_mean = t[0].mean(dim=1).to(s.dtype)
    s_n = s_mean / s_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    t_n = t_mean / t_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    correct = (s_n - t_n).abs().pow(2.0).mean()

    # Buggy alternative: take (s-t) then normalize → different value.
    diff = s_mean - t_mean
    diff_n = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    buggy = diff_n.abs().pow(2.0).mean()

    torch.testing.assert_close(loss["loss"], correct, atol=1e-5, rtol=1e-4)
    assert abs(float(correct) - float(buggy)) > 1e-4, (
        "Test inputs must distinguish per-side normalization from after-diff normalization."
    )
