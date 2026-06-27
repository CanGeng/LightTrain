"""Coverage-completion tests for lighttrain.builtin_plugins.rl.ref_policy.

Pins branches left uncovered by tests/rl/test_ref_policy.py (77% baseline):

* Lines 41-48   : ``ReferencePolicy.device`` property — all four branches:
                  (a) _device explicitly set, (b) model present with params,
                  (c) model present but parameter-less (StopIteration path),
                  (d) model is None.
* Line  89      : ``log_probs`` LoRA-base path dispatches to
                  ``_lora_base_log_probs`` when lora_base_as_ref=True
                  and per_token=False.
* Line  91      : ``log_probs`` raises when model=None and
                  lora_base_as_ref=False.
* Line  105     : ``_frozen_log_probs`` attention_mask kwarg branch.
* Lines 130-131 : ``_lora_base_log_probs`` raises when live_model is None.
* Lines 135-140 : ``_lora_base_log_probs`` happy-path — disable then
                  re-enables adapters via finally block; also verifies that
                  re-enabling is called even when _frozen_log_probs raises.
* Line  213     : ``freeze_as_ref(lora_base_as_ref=True)`` — returns
                  ReferencePolicy with model=None.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.rl.ref_policy import (
    ReferencePolicy,
    _per_token_log_probs,
    _sequence_log_probs,
    freeze_as_ref,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Minimal LM: emb + linear head; returns ModelOutput with 'logits'."""

    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self._vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        logits = self.head(h)
        return ModelOutput(outputs={"logits": logits})


class _TinyLMDict(nn.Module):
    """Same but returns a plain dict (not ModelOutput) — exercises the other logits branch."""

    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        logits = self.head(h)
        return {"logits": logits}


class _NoParamModel(nn.Module):
    """Module with no parameters at all — triggers StopIteration in device lookup."""

    def forward(self, input_ids, attention_mask=None):  # noqa: ARG002
        B, T = input_ids.shape
        return {"logits": torch.zeros(B, T, 4)}


class _LoRAWrapper(nn.Module):
    """Simulates a PEFT LoRA-wrapped model with disable/enable_adapter_layers."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self._base = base
        self.disable_calls: int = 0
        self.enable_calls: int = 0
        self._adapters_enabled: bool = True

    def disable_adapter_layers(self) -> None:
        self._adapters_enabled = False
        self.disable_calls += 1

    def enable_adapter_layers(self) -> None:
        self._adapters_enabled = True
        self.enable_calls += 1

    def parameters(self):  # type: ignore[override]
        return self._base.parameters()

    def forward(self, input_ids, attention_mask=None):
        return self._base(input_ids, attention_mask)


class _ExplodingLoRA(_LoRAWrapper):
    """Like _LoRAWrapper but the forward call raises after adapters are disabled."""

    def forward(self, input_ids, attention_mask=None):
        raise RuntimeError("model exploded mid-forward")


# ---------------------------------------------------------------------------
# device property  (lines 41-48)
# ---------------------------------------------------------------------------

class TestDeviceProperty:
    def test_invariant_device_returns_explicit_device_when_set(self):
        """(Line 41-42) If _device is not None it is returned immediately."""
        dev = torch.device("cpu")
        ref = ReferencePolicy(model=None, _device=dev)
        assert ref.device == dev

    def test_invariant_device_returns_model_device_when_model_has_params(self):
        """(Lines 43-45) When model has parameters, device is inferred from first param."""
        model = _TinyLM()
        ref = ReferencePolicy(model=model)
        assert ref.device == torch.device("cpu")

    def test_invariant_device_returns_none_for_parameterless_model(self):
        """(Lines 43-47) StopIteration branch: no parameters → returns None."""
        ref = ReferencePolicy(model=_NoParamModel())
        # _NoParamModel has no nn.Parameters registered, so next(...) raises StopIteration.
        assert ref.device is None

    def test_invariant_device_returns_none_when_model_is_none(self):
        """(Line 48) Model is None with no _device set → returns None."""
        ref = ReferencePolicy(model=None)
        assert ref.device is None

    def test_invariant_explicit_device_takes_precedence_over_model(self):
        """(Line 41-42) Explicit _device is returned without inspecting model."""
        forced = torch.device("cpu")
        model = _TinyLM()
        ref = ReferencePolicy(model=model, _device=forced)
        # Even though model has params, _device wins.
        assert ref.device == forced


# ---------------------------------------------------------------------------
# log_probs — model=None guard  (line 91)
# ---------------------------------------------------------------------------

class TestLogProbsModelNoneGuard:
    def test_invariant_raises_when_model_none_and_not_lora_base(self):
        """(Line 91) RuntimeError when model is None and lora_base_as_ref=False."""
        ref = ReferencePolicy(model=None, lora_base_as_ref=False)
        input_ids = torch.randint(0, 8, (2, 4))
        labels = input_ids.clone()
        with pytest.raises(RuntimeError, match="model is None"):
            ref.log_probs(input_ids, None, labels)


# ---------------------------------------------------------------------------
# log_probs — lora_base path dispatch  (line 89)
# ---------------------------------------------------------------------------

class TestLogProbsLoraBaseDispatch:
    def test_invariant_lora_base_dispatch_calls_lora_base_log_probs(self):
        """(Line 89) lora_base_as_ref=True dispatches to _lora_base_log_probs.

        The returned tensor must be (B,) shape matching a sequence-level signal.
        """
        torch.manual_seed(10)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        input_ids = torch.randint(0, 8, (2, 5))
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, None, labels, live_model=lora)

        assert out.shape == (2,)
        assert torch.isfinite(out).all()
        # Adapters must be left in enabled state after the call.
        assert lora._adapters_enabled is True
        assert lora.disable_calls == 1
        assert lora.enable_calls == 1


# ---------------------------------------------------------------------------
# _frozen_log_probs — attention_mask branch  (line 105)
# ---------------------------------------------------------------------------

class TestFrozenLogProbsAttentionMask:
    def test_invariant_attention_mask_passed_to_model_when_not_none(self):
        """(Line 105) When attention_mask is not None it is forwarded as kwarg.

        The result shape and finiteness must still hold.
        """
        torch.manual_seed(20)
        model = _TinyLM(vocab=16, hidden=8)
        ref = freeze_as_ref(model)
        B, T = 2, 6
        input_ids = torch.randint(0, 16, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, attention_mask, labels)

        assert out.shape == (B,)
        assert torch.isfinite(out).all()

    def test_invariant_attention_mask_none_omits_kwarg(self):
        """Contrast: passing attention_mask=None must also work (existing path)."""
        torch.manual_seed(21)
        model = _TinyLM(vocab=16, hidden=8)
        ref = freeze_as_ref(model)
        input_ids = torch.randint(0, 16, (2, 6))
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, None, labels)

        assert out.shape == (2,)

    def test_invariant_attention_mask_result_consistent_with_plain_dict_model(self):
        """Dict-returning model also exercises the attention_mask branch cleanly."""
        torch.manual_seed(22)
        model = _TinyLMDict(vocab=8, hidden=4)
        ref = ReferencePolicy(model=model)
        input_ids = torch.randint(0, 8, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.long)
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, mask, labels)

        assert out.shape == (2,)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _lora_base_log_probs — live_model=None guard  (lines 130-131)
# ---------------------------------------------------------------------------

class TestLoraBaseLogProbsGuards:
    def test_invariant_raises_when_live_model_is_none(self):
        """(Lines 130-131) live_model=None with lora_base_as_ref=True raises."""
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        input_ids = torch.randint(0, 8, (2, 4))
        labels = input_ids.clone()
        with pytest.raises(RuntimeError, match="needs live_model"):
            ref.log_probs(input_ids, None, labels, live_model=None)


# ---------------------------------------------------------------------------
# _lora_base_log_probs — adapter toggle happy path + finally guard  (lines 135-140)
# ---------------------------------------------------------------------------

class TestLoraAdapterToggle:
    def test_invariant_adapters_disabled_then_reenabled_on_success(self):
        """(Lines 135-140) disable_adapter_layers → forward → enable_adapter_layers.

        Both calls must happen exactly once in the success path.
        """
        torch.manual_seed(30)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        input_ids = torch.randint(0, 8, (2, 4))
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, None, labels, live_model=lora)

        assert lora.disable_calls == 1
        assert lora.enable_calls == 1
        assert lora._adapters_enabled is True
        assert out.shape == (2,)

    def test_invariant_adapters_reenabled_even_when_forward_raises(self):
        """(Lines 138-139) The finally block re-enables adapters on exception.

        Even if the model's forward method raises, enable_adapter_layers() must
        still be called once — preventing a permanently-broken training model.
        """
        base = _TinyLM(vocab=8, hidden=4)
        lora = _ExplodingLoRA(base)
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        input_ids = torch.randint(0, 8, (2, 4))
        labels = input_ids.clone()

        with pytest.raises(RuntimeError, match="exploded"):
            ref.log_probs(input_ids, None, labels, live_model=lora)

        # finally must have fired: adapters re-enabled despite the exception.
        assert lora.disable_calls == 1
        assert lora.enable_calls == 1
        assert lora._adapters_enabled is True

    def test_invariant_lora_base_result_matches_frozen_copy_result(self):
        """LoRA-base path with adapters disabled is equivalent to the frozen copy.

        When lora=_LoRAWrapper(base) passes the forward unchanged (adapters are
        a no-op for _LoRAWrapper), result must match the frozen-copy path.
        """
        torch.manual_seed(31)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref_lora = ReferencePolicy(model=None, lora_base_as_ref=True)
        ref_copy = freeze_as_ref(base)

        input_ids = torch.randint(0, 8, (2, 5))
        labels = input_ids.clone()

        out_lora = ref_lora.log_probs(input_ids, None, labels, live_model=lora)
        out_copy = ref_copy.log_probs(input_ids, None, labels)

        torch.testing.assert_close(out_lora, out_copy, atol=1e-5, rtol=1e-4)

    def test_invariant_lora_base_per_token_matches_frozen_copy(self):
        """(A2) per_token=True on the LoRA-base path: adapters toggle once each and
        the (B, T) result equals the frozen-copy per-token log-probs."""
        torch.manual_seed(33)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref_lora = ReferencePolicy(model=None, lora_base_as_ref=True)
        ref_copy = freeze_as_ref(base)

        input_ids = torch.randint(0, 8, (2, 5))
        labels = input_ids.clone()

        out_lora = ref_lora.log_probs(
            input_ids, None, labels, live_model=lora, per_token=True
        )
        out_copy = ref_copy.log_probs(input_ids, None, labels, per_token=True)

        assert out_lora.shape == (2, 5)
        assert lora.disable_calls == 1
        assert lora.enable_calls == 1
        assert lora._adapters_enabled is True
        torch.testing.assert_close(out_lora, out_copy, atol=1e-5, rtol=1e-4)

    def test_invariant_lora_base_per_token_reenabled_when_forward_raises(self):
        """(A2) finally guard also holds on the per-token path: adapters re-enabled
        even if the forward raises."""
        base = _TinyLM(vocab=8, hidden=4)
        lora = _ExplodingLoRA(base)
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        input_ids = torch.randint(0, 8, (2, 4))

        with pytest.raises(RuntimeError, match="exploded"):
            ref.log_probs(input_ids, None, input_ids.clone(),
                          live_model=lora, per_token=True)

        assert lora.disable_calls == 1
        assert lora.enable_calls == 1
        assert lora._adapters_enabled is True

    def test_invariant_lora_base_with_attention_mask(self):
        """(Line 105 via _lora_base_log_probs) Attention mask forwarded correctly."""
        torch.manual_seed(32)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref = ReferencePolicy(model=None, lora_base_as_ref=True)
        B, T = 2, 5
        input_ids = torch.randint(0, 8, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels = input_ids.clone()

        out = ref.log_probs(input_ids, attention_mask, labels, live_model=lora)

        assert out.shape == (B,)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# freeze_as_ref lora_base_as_ref=True path  (line 213)
# ---------------------------------------------------------------------------

class TestFreezeAsRefLoraBase:
    def test_invariant_lora_base_returns_ref_policy_with_model_none(self):
        """(Line 213) freeze_as_ref(lora_base_as_ref=True) skips deepcopy.

        Result must have model=None and lora_base_as_ref=True.
        """
        model = _TinyLM(vocab=8, hidden=4)
        ref = freeze_as_ref(model, lora_base_as_ref=True)
        assert ref.model is None
        assert ref.lora_base_as_ref is True

    def test_invariant_lora_base_preserves_custom_ignore_index(self):
        """(Line 213) Custom ignore_index is preserved on the returned policy."""
        model = _TinyLM(vocab=8, hidden=4)
        ref = freeze_as_ref(model, lora_base_as_ref=True, ignore_index=-1)
        assert ref.ignore_index == -1

    def test_invariant_lora_base_does_not_copy_model(self):
        """(Line 213) No deepcopy means no extra memory: ref.model is None."""
        model = _TinyLM(vocab=8, hidden=4)
        ref = freeze_as_ref(model, lora_base_as_ref=True)
        # The original model is NOT stored inside the ReferencePolicy.
        assert ref.model is not model
        assert ref.model is None

    def test_invariant_lora_base_can_call_log_probs_end_to_end(self):
        """Smoke-test the full pipeline: freeze_as_ref(..., lora_base_as_ref=True)
        followed by log_probs(..., live_model=...) runs without error.
        """
        torch.manual_seed(40)
        base = _TinyLM(vocab=8, hidden=4)
        lora = _LoRAWrapper(base)
        ref = freeze_as_ref(base, lora_base_as_ref=True)

        input_ids = torch.randint(0, 8, (2, 5))
        labels = input_ids.clone()
        out = ref.log_probs(input_ids, None, labels, live_model=lora)

        assert out.shape == (2,)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _per_token_log_probs standalone  (already partially tested; extend for shapes)
# ---------------------------------------------------------------------------

class TestPerTokenLogProbsStandalone:
    def test_invariant_per_token_first_col_zero(self):
        """Leading zero column is always zero regardless of logits."""
        torch.manual_seed(50)
        B, T, V = 3, 7, 12
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        out = _per_token_log_probs(logits, input_ids)
        assert out.shape == (B, T)
        assert torch.all(out[:, 0] == 0.0)

    def test_invariant_per_token_values_are_nonpositive(self):
        """Log-probs (from log_softmax) must all be <= 0."""
        torch.manual_seed(51)
        B, T, V = 2, 6, 10
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        out = _per_token_log_probs(logits, input_ids)
        # First col is 0 exactly; rest are <= 0.
        assert torch.all(out <= 0.0)


# ---------------------------------------------------------------------------
# Additional edge cases for _sequence_log_probs
# ---------------------------------------------------------------------------

class TestSequenceLogProbsEdgeCases:
    def test_invariant_all_positions_masked_returns_zero_not_nan(self):
        """When all labels are ignore_index, clamp_min(1.0) prevents division-by-zero.

        Result should be 0.0 for each sample rather than NaN.
        """
        B, T, V = 2, 4, 5
        logits = torch.zeros(B, T, V)
        # All labels = -100 means all shifted labels are -100 (fully masked).
        labels = torch.full((B, T), -100, dtype=torch.long)
        out = _sequence_log_probs(logits, labels, ignore_index=-100)
        assert out.shape == (B,)
        # numerator = 0, denominator = clamp_min(1.0) = 1 → 0.0
        assert torch.all(out == 0.0)

    def test_invariant_custom_ignore_index_honored(self):
        """Non-default ignore_index (-1) is respected for masking."""
        B, T, V = 1, 4, 5
        logits = torch.zeros(B, T, V)
        # Only label at shift-pos 0 is valid (value=2); rest = -1 (masked).
        labels = torch.tensor([[-1, 2, -1, -1]], dtype=torch.long)
        out = _sequence_log_probs(logits, labels, ignore_index=-1)
        # uniform logits → log_softmax = -log(5) at pos 0; denominator = 1.
        import math
        expected = torch.tensor([-math.log(5)])
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("B,T,V", [(1, 2, 2), (4, 8, 32), (1, 3, 100)])
    def test_invariant_output_shape_is_batch_size(self, B, T, V):
        """Output shape is always (B,)."""
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = _sequence_log_probs(logits, labels)
        assert out.shape == (B,)
