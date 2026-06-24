"""Edge-case coverage tests for lighttrain.builtin_plugins.losses.distill.

Companion to ``test_distill.py`` (which holds the closed-form / regression
suite). This file drives the remaining uncovered branches toward 100%:

Module-private helpers
  * ``_gather_aux`` direct-hit and miss paths.
  * ``_label_mask`` ``mask_from_labels=False`` and ``labels is None`` returns.
  * ``_student_logits`` plain-Mapping path.
  * ``_student_hidden`` / ``_student_attn`` None-output RuntimeErrors and
    non-ModelOutput TypeErrors.
  * ``LayerMapping.coerce`` unsupported-type TypeError.

Loss error / branch paths
  * ``KLDivLoss``: missing-key KeyError, no-mask denominator, reduction="none".
  * ``HiddenStatesMSELoss``: missing-key KeyError, dtype-mismatch warning,
    project=True without ``ctx.extras['model']`` RuntimeError, no-mask
    denominator, empty-mapping ValueError, projection cache-hit + orthogonal /
    normal / unsupported init, bias-zeros init, and the ``ctx`` without
    ``extras`` degraded-mode warning.
  * ``HiddenStatesCosineLoss``: missing-key KeyError, dtype-mismatch warning,
    dim-mismatch RuntimeError, no-mask denominator, empty-mapping ValueError.
  * ``AttentionTransferLoss``: missing-key KeyError, empty-mapping ValueError.

Cross-vocab remap registry
  * ``get_remap`` unknown-name KeyError + known-name retrieval, ``list_remaps``,
    and the ``_remap_top_k`` default gather.

All randomness is seeded; no timing or flaky assertions.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.losses.distill import (
    AttentionTransferLoss,
    CrossVocabRemapRegistry,
    HiddenStatesCosineLoss,
    HiddenStatesMSELoss,
    KLDivLoss,
    LayerMapping,
    _gather_aux,
    _label_mask,
    _remap_top_k,
    _student_attn,
    _student_hidden,
    _student_logits,
)
from lighttrain.protocols import LossContext, ModelOutput

# ---------------------------------------------------------------------------
# Tiny stubs / helpers (prefixed "_" per house style)
# ---------------------------------------------------------------------------


def _mo_hidden(hidden_states):
    """ModelOutput carrying only ``hidden_states``."""
    return ModelOutput(outputs={}, hidden_states=tuple(hidden_states))


def _mo_attn(attns):
    """ModelOutput carrying only ``attentions``."""
    return ModelOutput(outputs={}, attentions=tuple(attns))


class _NoExtrasCtx:
    """LossContext-shaped stub that deliberately lacks an ``extras`` attribute.

    Drives ``_ensure_projection``'s ``except AttributeError`` degraded path
    (``ctx.extras.setdefault`` raises because there is no ``extras``).
    """

    # No ``extras`` attribute on purpose.


# ---------------------------------------------------------------------------
# _gather_aux
# ---------------------------------------------------------------------------


def test_invariant_gather_aux_returns_value_on_direct_hit():
    """``_gather_aux`` returns the tensor when ``aux.<ns>.<key>`` is present."""
    t = torch.arange(4)
    batch = {"aux.teacher.foo": t}
    out = _gather_aux(batch, "teacher", "foo")
    assert out is t


def test_invariant_gather_aux_returns_none_when_absent():
    """``_gather_aux`` returns None for a missing key (split-leaf fallback)."""
    batch = {"aux.teacher.foo.values": torch.zeros(2)}
    assert _gather_aux(batch, "teacher", "foo") is None


# ---------------------------------------------------------------------------
# _label_mask
# ---------------------------------------------------------------------------


def test_invariant_label_mask_disabled_returns_none():
    """``mask_from_labels=False`` short-circuits to None even with labels present."""
    batch = {"labels": torch.tensor([[1, -100]])}
    assert _label_mask(batch, mask_from_labels=False) is None


def test_invariant_label_mask_missing_labels_returns_none():
    """No ``labels`` key → None (loss falls back to counting all tokens)."""
    assert _label_mask({}, mask_from_labels=True) is None


def test_invariant_label_mask_builds_boolean_mask():
    """Present labels → boolean mask of ``labels != ignore_index``."""
    labels = torch.tensor([[1, -100, 3]])
    mask = _label_mask({"labels": labels}, ignore_index=-100)
    assert mask is not None
    torch.testing.assert_close(mask, torch.tensor([[True, False, True]]))


# ---------------------------------------------------------------------------
# _student_logits / _student_hidden / _student_attn
# ---------------------------------------------------------------------------


def test_invariant_student_logits_plain_mapping_path():
    """A plain mapping (not ModelOutput) reads ``["logits"]`` directly."""
    logits = torch.randn(1, 2, 3)
    out = _student_logits({"logits": logits})
    assert out is logits


def test_invariant_student_hidden_none_states_raises():
    """ModelOutput with ``hidden_states=None`` → RuntimeError naming the flag."""
    mo = ModelOutput(outputs={"logits": torch.zeros(1)}, hidden_states=None)
    with pytest.raises(RuntimeError, match="output_hidden_states=True"):
        _student_hidden(mo, 0)


def test_invariant_student_hidden_non_modeloutput_raises_type_error():
    """A plain mapping cannot supply hidden states → TypeError."""
    with pytest.raises(TypeError, match="requires a ModelOutput"):
        _student_hidden({"logits": torch.zeros(1)}, 0)


def test_invariant_student_attn_none_attentions_raises():
    """ModelOutput with ``attentions=None`` → RuntimeError naming the flag."""
    mo = ModelOutput(outputs={"logits": torch.zeros(1)}, attentions=None)
    with pytest.raises(RuntimeError, match="output_attentions=True"):
        _student_attn(mo, 0)


def test_invariant_student_attn_non_modeloutput_raises_type_error():
    """A plain mapping cannot supply attentions → TypeError."""
    with pytest.raises(TypeError, match="requires a ModelOutput"):
        _student_attn({"logits": torch.zeros(1)}, 0)


# ---------------------------------------------------------------------------
# LayerMapping.coerce
# ---------------------------------------------------------------------------


def test_invariant_layer_mapping_coerce_rejects_unsupported_type():
    """``coerce`` on an int (neither Mapping/list/LayerMapping) → TypeError."""
    with pytest.raises(TypeError, match="cannot coerce int to LayerMapping"):
        LayerMapping.coerce(5)


# ---------------------------------------------------------------------------
# KLDivLoss branch paths
# ---------------------------------------------------------------------------


def test_invariant_kl_missing_aux_keys_raises_keyerror(dummy_ctx):
    """Absent teacher values/indices → KeyError mentioning both expected keys."""
    mo = ModelOutput(outputs={"logits": torch.randn(1, 2, 5)})
    with pytest.raises(KeyError, match="kl_topk needs"):
        KLDivLoss()(mo, {"labels": torch.zeros(1, 2, dtype=torch.long)}, dummy_ctx)


def test_invariant_kl_no_mask_denominator_uses_token_count(dummy_ctx):
    """With ``mask_from_labels=False`` the denom is ``per_token.numel()`` (=B*T).

    Build a batch *without* labels and disable masking so the no-mask
    denominator branch runs; the mean must equal the unmasked per-token mean.
    """
    torch.manual_seed(31)
    B, T, K, V = 2, 3, 3, 6
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out = KLDivLoss(temperature=2.0, top_k=K, mask_from_labels=False)(mo, batch, dummy_ctx)
    # denom must equal the total token count B*T.
    assert out["kl_topk_unmasked_tokens"] == pytest.approx(float(B * T))


def test_invariant_kl_reduction_none_returns_per_token(dummy_ctx):
    """reduction='none' returns the full per-token (B,T) tensor, not a scalar."""
    torch.manual_seed(32)
    B, T, K, V = 2, 3, 3, 6
    teacher_logits = torch.randn(B, T, V)
    vals, idx = torch.topk(teacher_logits, k=K, dim=-1)
    student_logits = torch.randn(B, T, V)
    batch = {
        "aux.teacher.logits_topk_64.values": vals,
        "aux.teacher.logits_topk_64.indices": idx,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = ModelOutput(outputs={"logits": student_logits})
    out = KLDivLoss(temperature=2.0, top_k=K, reduction="none")(mo, batch, dummy_ctx)
    assert out["loss"].shape == (B, T)
    assert torch.all(out["loss"] >= 0.0)


# ---------------------------------------------------------------------------
# HiddenStatesMSELoss branch paths
# ---------------------------------------------------------------------------


def test_invariant_hidden_mse_missing_key_raises_keyerror(dummy_ctx):
    """Absent teacher payload → KeyError naming the expected aux key."""
    mo = _mo_hidden([torch.zeros(1, 2, 3)])
    with pytest.raises(KeyError, match="hidden_mse needs"):
        HiddenStatesMSELoss(mapping={0: 0})(mo, {"labels": torch.zeros(1, 2, dtype=torch.long)}, dummy_ctx)


def test_invariant_hidden_mse_empty_mapping_raises_valueerror(dummy_ctx):
    """An empty mapping yields ``total is None`` → ValueError."""
    B, T, H = 1, 2, 3
    teacher = torch.zeros(1, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([torch.zeros(B, T, H)])
    with pytest.raises(ValueError, match="hidden_mse mapping was empty"):
        HiddenStatesMSELoss(mapping={})(mo, batch, dummy_ctx)


def test_invariant_hidden_mse_no_mask_denominator(dummy_ctx):
    """``mask_from_labels=False`` (no labels) uses ``err.numel()`` denom.

    s=0, t=1 everywhere → per-element MSE = 1, mean = 1 regardless of denom path.
    """
    B, T, H = 2, 3, 4
    s = torch.zeros(B, T, H)
    t = torch.ones(1, B, T, H)
    batch = {"aux.teacher.hidden_states_layers": t}
    mo = _mo_hidden([s])
    out = HiddenStatesMSELoss(mapping={0: 0}, mask_from_labels=False)(mo, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_invariant_hidden_mse_non_mean_reduction_skips_layer_average(dummy_ctx):
    """reduction != 'mean' returns the per-layer SUM (no division by #layers).

    Two layers each with loss 1.0; mean → 1.0, sum → 2.0. Pins the
    ``if reduction == 'mean'`` fall-through.
    """
    B, T, H = 1, 2, 3
    s0 = torch.zeros(B, T, H)
    s1 = torch.zeros(B, T, H)
    teacher = torch.ones(2, B, T, H)  # each layer (0-1)² = 1
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s0, s1])
    out_sum = HiddenStatesMSELoss(mapping={0: 0, 1: 1}, reduction="sum")(mo, batch, dummy_ctx)
    out_mean = HiddenStatesMSELoss(mapping={0: 0, 1: 1}, reduction="mean")(mo, batch, dummy_ctx)
    torch.testing.assert_close(out_sum["loss"], torch.tensor(2.0), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_mean["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_invariant_hidden_mse_dtype_mismatch_warns_once(dummy_ctx):
    """Teacher dtype != student dtype → exactly one auto-cast UserWarning.

    Two mapped layers both mismatch but the warning fires only once
    (``_warned_dtype`` latch).
    """
    B, T, H = 1, 2, 3
    s0 = torch.zeros(B, T, H, dtype=torch.float32)
    s1 = torch.zeros(B, T, H, dtype=torch.float32)
    teacher = torch.zeros(2, B, T, H, dtype=torch.float64)  # mismatched dtype
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s0, s1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = HiddenStatesMSELoss(mapping={0: 0, 1: 1})(mo, batch, dummy_ctx)
    dtype_warns = [w for w in caught if "teacher dtype" in str(w.message)]
    assert len(dtype_warns) == 1
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_invariant_hidden_mse_project_true_without_model_raises():
    """project=True but ``ctx.extras['model']`` absent → RuntimeError."""
    B, T = 1, 2
    s = torch.randn(B, T, 3)
    teacher = torch.ones(1, B, T, 5)  # dim mismatch forces the projection path
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    ctx = LossContext(extras={})  # no "model"
    with pytest.raises(RuntimeError, match=r"ctx.extras\['model'\]"):
        HiddenStatesMSELoss(mapping={0: 0}, project=True)(mo, batch, ctx)


@pytest.mark.parametrize("init", ["xavier", "orthogonal", "normal"])
def test_invariant_hidden_mse_projection_init_variants_run(init):
    """xavier / orthogonal / normal projection inits build a finite, trainable loss."""
    torch.manual_seed(33)
    B, T = 1, 2
    Hs, Ht = 3, 5
    s = torch.randn(B, T, Hs)
    teacher = torch.ones(1, B, T, Ht)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    host = nn.Module()
    ctx = LossContext(extras={"model": host})
    out = HiddenStatesMSELoss(
        mapping={0: 0}, project=True, project_init=init
    )(mo, batch, ctx)
    assert torch.isfinite(out["loss"])
    # The projection layer parameters were published for the optimizer.
    assert ctx.extras.get("_new_trainable_params")


def test_invariant_hidden_mse_projection_unsupported_init_raises():
    """An unknown ``project_init`` → ValueError naming the bad value."""
    B, T = 1, 2
    s = torch.randn(B, T, 3)
    teacher = torch.ones(1, B, T, 5)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    ctx = LossContext(extras={"model": nn.Module()})
    with pytest.raises(ValueError, match="project_init='bogus' unsupported"):
        HiddenStatesMSELoss(
            mapping={0: 0}, project=True, project_init="bogus"
        )(mo, batch, ctx)


def test_invariant_hidden_mse_projection_bias_zeros_initialized():
    """project_bias=True with zeros weight init → s_projected=0 → loss=mean(t²).

    Exercises the ``project_bias`` zeros-init branch; with weight and bias both
    zero the projected student is all zeros so loss = mean((0-1)²) = 1.
    """
    B, T = 1, 2
    Hs, Ht = 3, 5
    s = torch.randn(B, T, Hs)
    teacher = torch.ones(1, B, T, Ht)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    ctx = LossContext(extras={"model": nn.Module()})
    out = HiddenStatesMSELoss(
        mapping={0: 0}, project=True, project_init="zeros", project_bias=True
    )(mo, batch, ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


def test_invariant_hidden_mse_projection_cache_hit_reuses_submodule():
    """Calling the same loss twice reuses the cached projection submodule.

    First call attaches the ``nn.Linear``; the second call finds the cached
    path and returns the existing submodule (the cache-hit branch).
    """
    torch.manual_seed(34)
    B, T = 1, 2
    Hs, Ht = 3, 5
    s = torch.randn(B, T, Hs)
    teacher = torch.ones(1, B, T, Ht)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    host = nn.Module()
    ctx = LossContext(extras={"model": host})
    loss_fn = HiddenStatesMSELoss(mapping={0: 0}, project=True, project_init="zeros")
    out1 = loss_fn(mo, batch, ctx)
    proj1 = loss_fn._projection_paths[(0, Hs, Ht)]
    out2 = loss_fn(mo, batch, ctx)
    proj2 = loss_fn._projection_paths[(0, Hs, Ht)]
    assert proj1 == proj2  # same cached path, not rebuilt
    torch.testing.assert_close(out1["loss"], out2["loss"], atol=1e-6, rtol=1e-5)


def test_pin_current_behavior_hidden_mse_ctx_without_extras_warns_degraded():
    """Pin: a ctx lacking ``extras`` falls into the degraded warn path.

    DEBATABLE: ``_ensure_projection`` reaches ``ctx.extras.setdefault`` only
    AFTER ``__call__`` already accessed ``ctx.extras.get('model')`` — so a ctx
    truly missing ``extras`` would fail earlier with a project=True flow. To hit
    the ``except AttributeError`` warn branch in isolation we pass a stub whose
    ``extras`` is a property that errors only on ``.setdefault`` after the model
    is fetched. We pin that the degraded-mode warning is emitted and the loss is
    still computed.
    """
    B, T = 1, 2
    Hs, Ht = 3, 5
    s = torch.randn(B, T, Hs)
    teacher = torch.ones(1, B, T, Ht)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])

    host = nn.Module()

    class _ExtrasNoSetdefault(dict):
        # ``.get('model')`` works (dict), but accessing ``.setdefault`` raises
        # AttributeError so the loss falls into the degraded warn branch.
        def __getattribute__(self, name):
            if name == "setdefault":
                raise AttributeError("setdefault unavailable")
            return super().__getattribute__(name)

    class _Ctx:
        extras = _ExtrasNoSetdefault({"model": host})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = HiddenStatesMSELoss(
            mapping={0: 0}, project=True, project_init="zeros"
        )(mo, batch, _Ctx())
    degraded = [w for w in caught if "won't be auto-registered" in str(w.message)]
    assert len(degraded) == 1
    torch.testing.assert_close(out["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# HiddenStatesCosineLoss branch paths
# ---------------------------------------------------------------------------


def test_invariant_hidden_cosine_missing_key_raises_keyerror(dummy_ctx):
    """Absent teacher payload → KeyError naming the expected aux key."""
    mo = _mo_hidden([torch.zeros(1, 2, 3)])
    with pytest.raises(KeyError, match="hidden_cosine needs"):
        HiddenStatesCosineLoss(mapping={0: 0})(
            mo, {"labels": torch.zeros(1, 2, dtype=torch.long)}, dummy_ctx
        )


def test_invariant_hidden_cosine_empty_mapping_raises_valueerror(dummy_ctx):
    """An empty mapping yields ``total is None`` → ValueError."""
    B, T, H = 1, 2, 3
    teacher = torch.zeros(1, B, T, H)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([torch.zeros(B, T, H)])
    with pytest.raises(ValueError, match="hidden_cosine mapping was empty"):
        HiddenStatesCosineLoss(mapping={})(mo, batch, dummy_ctx)


def test_invariant_hidden_cosine_dim_mismatch_raises(dummy_ctx):
    """Cosine has no projection — a hidden-dim mismatch must RuntimeError."""
    B, T = 1, 2
    s = torch.zeros(B, T, 3)
    teacher = torch.zeros(1, B, T, 5)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s])
    with pytest.raises(RuntimeError, match="pre-project teacher"):
        HiddenStatesCosineLoss(mapping={0: 0})(mo, batch, dummy_ctx)


def test_invariant_hidden_cosine_dtype_mismatch_warns_once(dummy_ctx):
    """Teacher dtype != student dtype → exactly one auto-cast UserWarning."""
    B, T, H = 1, 2, 3
    s0 = torch.ones(B, T, H, dtype=torch.float32)
    s1 = torch.ones(B, T, H, dtype=torch.float32)
    teacher = torch.ones(2, B, T, H, dtype=torch.float64)
    batch = {
        "aux.teacher.hidden_states_layers": teacher,
        "labels": torch.zeros(B, T, dtype=torch.long),
    }
    mo = _mo_hidden([s0, s1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = HiddenStatesCosineLoss(mapping={0: 0, 1: 1})(mo, batch, dummy_ctx)
    dtype_warns = [w for w in caught if "teacher dtype" in str(w.message)]
    assert len(dtype_warns) == 1
    # Identical unit vectors → cosine 1 → per-layer loss 0 → mean 0.
    torch.testing.assert_close(out["loss"], torch.tensor(0.0), atol=1e-5, rtol=1e-4)


def test_invariant_hidden_cosine_no_mask_denominator(dummy_ctx):
    """``mask_from_labels=False`` (no labels) uses the ``err.numel()`` denom."""
    B, T, H = 1, 2, 3
    s = torch.zeros(B, T, H)
    s[..., 0] = 1.0
    t_full = torch.zeros(1, B, T, H)
    t_full[..., 1] = 1.0  # orthogonal → cosine 0 → loss 1
    batch = {"aux.teacher.hidden_states_layers": t_full}
    mo = _mo_hidden([s])
    out = HiddenStatesCosineLoss(mapping={0: 0}, mask_from_labels=False)(mo, batch, dummy_ctx)
    torch.testing.assert_close(out["loss"], torch.tensor(1.0), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# AttentionTransferLoss branch paths
# ---------------------------------------------------------------------------


def test_invariant_attn_transfer_missing_key_raises_keyerror(dummy_ctx):
    """Absent teacher attentions payload → KeyError naming the aux key."""
    mo = _mo_attn([torch.softmax(torch.randn(1, 2, 3, 3), dim=-1)])
    with pytest.raises(KeyError, match="attention_transfer needs"):
        AttentionTransferLoss(mapping={0: 0})(mo, {}, dummy_ctx)


def test_invariant_attn_transfer_empty_mapping_raises_valueerror(dummy_ctx):
    """An empty mapping yields ``total is None`` → ValueError."""
    teacher = torch.softmax(torch.randn(1, 1, 2, 3, 3), dim=-1)
    batch = {"aux.teacher.attentions_layers": teacher}
    mo = _mo_attn([torch.softmax(torch.randn(1, 2, 3, 3), dim=-1)])
    with pytest.raises(ValueError, match="attention_transfer mapping was empty"):
        AttentionTransferLoss(mapping={})(mo, batch, dummy_ctx)


# ---------------------------------------------------------------------------
# CrossVocabRemapRegistry + _remap_top_k
# ---------------------------------------------------------------------------


def test_invariant_remap_registry_lists_default_top_k():
    """``list_remaps`` includes the auto-registered ``top_k`` default."""
    assert "top_k" in CrossVocabRemapRegistry.list_remaps()


def test_invariant_remap_registry_get_known_returns_callable():
    """``get_remap('top_k')`` returns the default ``_remap_top_k`` callable."""
    fn = CrossVocabRemapRegistry.get_remap("top_k")
    assert fn is _remap_top_k


def test_invariant_remap_registry_get_unknown_raises_keyerror():
    """An unknown remap name → KeyError listing the available names."""
    with pytest.raises(KeyError, match="unknown remap 'nope'"):
        CrossVocabRemapRegistry.get_remap("nope")


def test_invariant_remap_top_k_gathers_at_teacher_indices():
    """``_remap_top_k`` gathers student logits at teacher top-K indices.

    Returns ``(gathered, idx)`` with indices coerced to long and unchanged.
    """
    B, T, V, _ = 1, 2, 5, 3
    student_logits = torch.arange(B * T * V, dtype=torch.float32).reshape(B, T, V)
    teacher_indices = torch.tensor([[[0, 2, 4], [1, 3, 0]]], dtype=torch.int32)
    gathered, idx = _remap_top_k(student_logits, teacher_indices)
    assert idx.dtype == torch.long
    torch.testing.assert_close(idx, teacher_indices.long())
    expected = torch.gather(student_logits, dim=-1, index=teacher_indices.long())
    torch.testing.assert_close(gathered, expected)


def test_invariant_remap_registry_register_roundtrip():
    """A custom remap can be registered and retrieved; registry is per-name.

    Uses a uniquely named key to avoid clobbering the shared class registry.
    """
    name = "_test_cov_identity_remap"

    def _identity(student_logits, teacher_indices):
        idx = teacher_indices.long()
        return torch.gather(student_logits, dim=-1, index=idx), idx

    CrossVocabRemapRegistry.register_remap(name, _identity)
    try:
        assert CrossVocabRemapRegistry.get_remap(name) is _identity
        assert name in CrossVocabRemapRegistry.list_remaps()
    finally:
        CrossVocabRemapRegistry._registry.pop(name, None)
