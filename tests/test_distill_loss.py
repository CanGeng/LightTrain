"""Distillation losses — DESIGN §9.1."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.losses.distill import (
    AttentionTransferLoss,
    HiddenStatesCosineLoss,
    HiddenStatesMSELoss,
    KLDivLoss,
    LayerMapping,
)
from lighttrain.protocols import LossContext, ModelOutput


def test_kl_topk_zero_when_student_matches_teacher():
    """If student logits == teacher gathered values, KL should be ~0."""
    B, T, V, K = 2, 5, 32, 8
    student_logits = torch.randn(B, T, V)
    teacher_idx = torch.argsort(student_logits, dim=-1, descending=True)[..., :K]
    teacher_vals = torch.gather(student_logits, dim=-1, index=teacher_idx)

    model_out = ModelOutput(outputs={"logits": student_logits})
    batch = {
        "labels": torch.zeros(B, T, dtype=torch.long),
        "aux.teacher.logits_topk_64.values": teacher_vals,
        "aux.teacher.logits_topk_64.indices": teacher_idx.to(torch.int32),
    }
    loss_fn = KLDivLoss(temperature=2.0, mask_from_labels=False)
    out = loss_fn(model_out, batch, LossContext())
    assert float(out["loss"]) < 1e-5


def test_kl_topk_positive_when_student_disagrees():
    B, T, V, _K = 1, 3, 16, 4
    teacher_idx = torch.tensor([[0, 1, 2, 3]] * T).unsqueeze(0)  # (1, T, K)
    teacher_vals = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * T).unsqueeze(0)
    student_logits = torch.zeros(B, T, V)  # uniform → KL > 0
    model_out = ModelOutput(outputs={"logits": student_logits})
    batch = {
        "labels": torch.zeros(B, T, dtype=torch.long),
        "aux.teacher.logits_topk_64.values": teacher_vals,
        "aux.teacher.logits_topk_64.indices": teacher_idx,
    }
    out = KLDivLoss(temperature=1.0, mask_from_labels=False)(
        model_out, batch, LossContext()
    )
    assert float(out["loss"]) > 0.1


def test_hidden_mse_zero_when_student_equals_teacher():
    L, B, T, H = 4, 2, 3, 8
    teacher = torch.randn(L, B, T, H)
    # Build student hidden_states tuple matching mapping {1→2, 2→3}
    student_hs = [torch.zeros(B, T, H) for _ in range(4)]
    student_hs[1] = teacher[2].clone()
    student_hs[2] = teacher[3].clone()
    model_out = ModelOutput(
        outputs={"logits": torch.zeros(B, T, 5)},
        hidden_states=tuple(student_hs),
    )
    batch = {
        "labels": torch.zeros(B, T, dtype=torch.long),
        "aux.teacher.hidden_states_layers": teacher,
    }
    loss_fn = HiddenStatesMSELoss(mapping={1: 2, 2: 3}, mask_from_labels=False)
    out = loss_fn(model_out, batch, LossContext())
    assert float(out["loss"]) < 1e-6


def test_hidden_mse_dim_mismatch_raises():
    student_hs = (torch.zeros(1, 2, 4),) * 3
    teacher = torch.zeros(3, 1, 2, 16)  # H=16 vs student H=4
    model_out = ModelOutput(outputs={"logits": torch.zeros(1, 2, 4)},
                            hidden_states=student_hs)
    batch = {"aux.teacher.hidden_states_layers": teacher,
             "labels": torch.zeros(1, 2, dtype=torch.long)}
    try:
        HiddenStatesMSELoss(mapping={0: 0})(model_out, batch, LossContext())
    except RuntimeError as e:
        assert "dim mismatch" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_hidden_cosine_zero_when_directions_match():
    L, B, T, H = 3, 1, 2, 4
    teacher = torch.randn(L, B, T, H)
    student_hs = [teacher[i].clone() * 2.0 for i in range(L)]  # same direction, diff magnitude
    model_out = ModelOutput(
        outputs={"logits": torch.zeros(B, T, 5)},
        hidden_states=tuple(student_hs),
    )
    batch = {"aux.teacher.hidden_states_layers": teacher,
             "labels": torch.zeros(B, T, dtype=torch.long)}
    loss_fn = HiddenStatesCosineLoss(mapping={0: 0, 1: 1, 2: 2}, mask_from_labels=False)
    out = loss_fn(model_out, batch, LossContext())
    # cosine sim of co-directional vectors = 1 → loss = 0
    assert float(out["loss"]) < 1e-5


def test_attention_transfer_zero_when_attentions_match():
    L, B, H, T = 2, 1, 2, 4
    teacher_at = torch.softmax(torch.randn(L, B, H, T, T), dim=-1)
    student_at = tuple(teacher_at[i].clone() for i in range(L))
    model_out = ModelOutput(outputs={"logits": torch.zeros(B, T, 4)},
                            attentions=student_at)
    batch = {"aux.teacher.attentions_layers": teacher_at,
             "labels": torch.zeros(B, T, dtype=torch.long)}
    loss_fn = AttentionTransferLoss(mapping={0: 0, 1: 1})
    out = loss_fn(model_out, batch, LossContext())
    assert float(out["loss"]) < 1e-5


def test_layer_mapping_coerce_handles_dict_and_pairs():
    a = LayerMapping.coerce({1: 2, 3: 4})
    b = LayerMapping.coerce([[1, 2], [3, 4]])
    c = LayerMapping.coerce(a)
    assert a.mapping == b.mapping == c.mapping == {1: 2, 3: 4}
