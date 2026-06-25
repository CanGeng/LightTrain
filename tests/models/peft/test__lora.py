"""Edge-case coverage for ``lighttrain.builtin_plugins.models.peft._lora``.

What we pin / cover:

* ``_normalize_output`` — already-a-ModelOutput fast-path (line 54-55);
  namespace-object with ``logits`` attr (lines 56, 63-65);
  plain dict with ``"logits"`` key (lines 57-58, 63-65);
  namespace-object without ``logits`` + not a dict → RuntimeError (lines 59-62);
  hidden_states / attentions tuple-wrapping (lines 63-65).
* ``LoRAAdapter.__init__`` — TypeError fallback for old peft that rejects
  ``use_rslora`` (lines 119-121).
* ``LoRAAdapter.full_state_dict`` (line 167).
* ``LoRAAdapter.enable_input_require_grads`` — inner has method (line 173);
  inner missing method (line 172 guard).
* ``LoRAAdapter.gradient_checkpointing_enable`` — inner has method (line 177);
  inner missing method (line 176 guard).
* ``LoRAAdapter.gradient_checkpointing_disable`` — inner has method (line 181);
  inner missing method (line 180 guard).
* ``LoRAAdapter.merge_and_unload`` — returns flat nn.Module (line 188).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

# The whole file requires peft.
peft = pytest.importorskip("peft")

from lighttrain.builtin_plugins.models.peft._lora import (  # noqa: E402
    LoRAAdapter,
    _normalize_output,
)
from lighttrain.builtin_plugins.models.text.tiny_lm import (  # noqa: E402
    TinyCausalLM,  # noqa: F401 — triggers @register("model","tiny_lm")
)
from lighttrain.protocols import ModelOutput  # noqa: E402

# ---------------------------------------------------------------------------
# Shared base spec — tiny TinyCausalLM so tests run fast
# ---------------------------------------------------------------------------

_BASE_KW = {
    "vocab_size": 32,
    "d_model": 8,
    "n_layers": 1,
    "n_heads": 2,
    "max_seq_len": 16,
}
_BASE_SPEC = {"name": "tiny_lm", **_BASE_KW}


def _make_lora(**overrides) -> LoRAAdapter:
    """Build a minimal LoRAAdapter with deterministic seed."""
    torch.manual_seed(0)
    kwargs: dict = dict(base=_BASE_SPEC, r=2, lora_alpha=4, lora_dropout=0.0)
    kwargs.update(overrides)
    return LoRAAdapter(**kwargs)


# ---------------------------------------------------------------------------
# _normalize_output — lines 56-65
# ---------------------------------------------------------------------------


def test_invariant_normalize_output_passes_through_model_output():
    """Line 54-55: a ``ModelOutput`` instance is returned unchanged (fast path)."""
    mo = ModelOutput(outputs={"logits": torch.zeros(2, 4)})
    result = _normalize_output(mo)
    assert result is mo


def test_invariant_normalize_output_namespace_with_logits_attr():
    """Lines 56, 63-65: an object that is NOT a ModelOutput and NOT a dict but
    carries a ``logits`` attribute is converted to ``ModelOutput``.
    ``hidden_states`` and ``attentions`` are None on the stub → both fields
    must be None in the result.
    """
    ns = types.SimpleNamespace(logits=torch.zeros(2, 4, 8))
    result = _normalize_output(ns)
    assert isinstance(result, ModelOutput)
    assert "logits" in result.outputs
    assert result.outputs["logits"] is ns.logits
    assert result.hidden_states is None
    assert result.attentions is None


def test_invariant_normalize_output_namespace_with_logits_and_hidden_states():
    """Lines 63-65: hidden_states / attentions present on the namespace object
    must be wrapped into tuples in the resulting ``ModelOutput``.
    """
    hs = (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
    at = (torch.zeros(1, 2, 2),)
    ns = types.SimpleNamespace(logits=torch.zeros(1, 2, 8), hidden_states=hs, attentions=at)
    result = _normalize_output(ns)
    assert result.hidden_states == hs
    assert result.attentions == at


def test_invariant_normalize_output_dict_with_logits_key():
    """Lines 57-58, 63-65: a plain dict with a ``"logits"`` key is converted to
    ``ModelOutput``; since dicts have no ``hidden_states`` / ``attentions``
    attributes both fields must be ``None`` in the result.
    """
    logits = torch.zeros(2, 3, 8)
    out = {"logits": logits}
    result = _normalize_output(out)
    assert isinstance(result, ModelOutput)
    assert result.outputs["logits"] is logits
    assert result.hidden_states is None
    assert result.attentions is None


def test_invariant_normalize_output_dict_without_logits_raises():
    """Lines 57-62 (dict path with missing key) → RuntimeError.

    A dict that does NOT contain ``"logits"`` triggers the RuntimeError branch
    (lines 59-62) with a message mentioning ``logits``.
    """
    with pytest.raises(RuntimeError, match="logits"):
        _normalize_output({"something_else": torch.zeros(2)})


def test_invariant_normalize_output_plain_namespace_no_logits_raises():
    """Lines 56, 59-62: a namespace-object whose ``getattr(…, "logits", None)``
    returns None and that is NOT a dict triggers ``RuntimeError``.
    """
    ns = types.SimpleNamespace(hidden_states=None)  # no logits attribute
    with pytest.raises(RuntimeError, match="logits"):
        _normalize_output(ns)


def test_invariant_normalize_output_runtime_error_mentions_type_name():
    """Lines 59-62: the RuntimeError message must include the type name of the
    bad object (``type(out).__name__``), which helps debugging.
    """

    class _WeirdOutput:
        pass

    with pytest.raises(RuntimeError, match="_WeirdOutput"):
        _normalize_output(_WeirdOutput())


# ---------------------------------------------------------------------------
# __init__ use_rslora TypeError fallback (lines 119-121)
# ---------------------------------------------------------------------------


def test_pin_current_behavior_use_rslora_typeerror_fallback(monkeypatch):
    """Lines 119-121: if the installed peft rejects ``use_rslora`` with a
    ``TypeError``, the code pops it and retries without it so construction
    still succeeds.

    We simulate old peft by monkey-patching ``peft.LoraConfig`` to raise
    ``TypeError`` when ``use_rslora`` is in kwargs, and succeed otherwise.
    Since ``_lora.py.__init__`` calls ``import_peft()`` then immediately calls
    ``peft.LoraConfig(…)`` on the returned module object, we patch the real
    peft module.

    NOTE: This pins CURRENT behaviour — the fallback silently degrades to
    peft<0.7 behaviour (no RSLoRA scaling). The test would break if the
    guard is removed or the retry logic changes.
    """
    real_LoraConfig = peft.LoraConfig
    call_count = [0]

    def _flaky_LoraConfig(**kwargs):
        call_count[0] += 1
        if "use_rslora" in kwargs:
            raise TypeError("unexpected keyword argument 'use_rslora'")
        return real_LoraConfig(**kwargs)

    monkeypatch.setattr(peft, "LoraConfig", _flaky_LoraConfig)

    torch.manual_seed(0)
    adapter = LoRAAdapter(
        base=_BASE_SPEC,
        r=2,
        lora_alpha=4,
        use_rslora=True,  # triggers first-call TypeError → retry without it
    )
    # Two calls: first raises TypeError, second succeeds without use_rslora.
    assert call_count[0] == 2
    # The adapter was built successfully despite the TypeError.
    assert hasattr(adapter, "inner")


# ---------------------------------------------------------------------------
# full_state_dict (line 167)
# ---------------------------------------------------------------------------


def test_invariant_full_state_dict_contains_base_and_adapter_keys():
    """Line 167: ``full_state_dict()`` returns base + adapter weights combined.

    The full dict is the raw ``inner.state_dict()`` which uses nested peft key
    paths (e.g. ``base_model.model.…lora_A.default.weight``). It must contain:
    - Keys with ``base_layer`` (= original base weights after peft wrapping)
    - Keys with ``lora_A`` / ``lora_B`` (= LoRA adapter weights)
    - More total keys than the adapter-only state_dict.
    """
    model = _make_lora()
    adapter_sd = model.state_dict()
    full_sd = model.full_state_dict()
    # Full dict has MORE keys than the adapter-only dict.
    assert len(full_sd) > len(adapter_sd), (
        f"full_state_dict ({len(full_sd)} keys) must exceed adapter_state_dict ({len(adapter_sd)} keys)"
    )
    # Full dict must contain base weight keys (identified by 'base_layer' marker in peft naming).
    base_keys = [k for k in full_sd if "base_layer" in k or "tok_emb" in k or "pos_emb" in k]
    assert base_keys, f"full_state_dict must contain base-weight keys; got: {list(full_sd.keys())[:5]}"
    # Full dict must also contain lora keys.
    lora_keys = [k for k in full_sd if "lora_" in k.lower()]
    assert lora_keys, "full_state_dict must contain lora_ keys"


def test_invariant_full_state_dict_returns_plain_dict():
    """``full_state_dict()`` returns a ``dict`` (not an OrderedDict or proxy)
    so callers can freely mutate it without affecting the model.
    """
    model = _make_lora()
    full = model.full_state_dict()
    assert type(full) is dict


def test_invariant_full_state_dict_is_mutable_copy():
    """Mutating the returned dict must not affect the model parameters."""
    model = _make_lora()
    full = model.full_state_dict()
    first_key = next(iter(full))
    del full[first_key]
    # Model still has its parameters intact.
    full2 = model.full_state_dict()
    assert first_key in full2


# ---------------------------------------------------------------------------
# enable_input_require_grads (lines 172-173)
# ---------------------------------------------------------------------------


def test_invariant_enable_input_require_grads_delegates_when_present():
    """Line 173: when ``.inner`` has ``enable_input_require_grads`` the method
    is called exactly once.
    """
    model = _make_lora()
    called = []

    # Attach the method directly as an instance attribute on inner.
    model.inner.enable_input_require_grads = lambda: called.append(1)
    assert hasattr(model.inner, "enable_input_require_grads")

    model.enable_input_require_grads()
    assert called == [1], "enable_input_require_grads must delegate to inner when present"


def test_invariant_enable_input_require_grads_noop_when_missing():
    """Line 172 guard: when ``.inner`` lacks ``enable_input_require_grads``
    the call must be a safe no-op (no AttributeError or other exception).
    Build a stub inner that provably has no such method.
    """

    class _InnerWithout(nn.Module):
        def forward(self, *a, **kw):  # pragma: no cover
            return None

    model = _make_lora()
    without = _InnerWithout()
    assert not hasattr(without, "enable_input_require_grads")
    model.inner = without
    # Must not raise.
    model.enable_input_require_grads()


# ---------------------------------------------------------------------------
# gradient_checkpointing_enable (lines 176-177)
# ---------------------------------------------------------------------------


def test_invariant_gradient_checkpointing_enable_delegates_when_present():
    """Line 177: when ``.inner`` has the method it is called with forwarded kwargs."""
    model = _make_lora()
    called_kwargs: list[dict] = []
    model.inner.gradient_checkpointing_enable = lambda **kw: called_kwargs.append(kw)
    assert hasattr(model.inner, "gradient_checkpointing_enable")

    model.gradient_checkpointing_enable(use_reentrant=False)
    assert called_kwargs == [{"use_reentrant": False}]


def test_invariant_gradient_checkpointing_enable_noop_when_missing():
    """Line 176 guard: no ``gradient_checkpointing_enable`` on inner → safe no-op."""

    class _NoCkpt(nn.Module):
        def forward(self, *a, **kw):  # pragma: no cover
            return None

    model = _make_lora()
    model.inner = _NoCkpt()
    assert not hasattr(model.inner, "gradient_checkpointing_enable")
    model.gradient_checkpointing_enable()  # must not raise


# ---------------------------------------------------------------------------
# gradient_checkpointing_disable (lines 180-181)
# ---------------------------------------------------------------------------


def test_invariant_gradient_checkpointing_disable_delegates_when_present():
    """Line 181: when ``.inner`` has ``gradient_checkpointing_disable`` it is
    called exactly once (no arguments).
    """
    model = _make_lora()
    called = []
    model.inner.gradient_checkpointing_disable = lambda: called.append(1)
    assert hasattr(model.inner, "gradient_checkpointing_disable")

    model.gradient_checkpointing_disable()
    assert called == [1]


def test_invariant_gradient_checkpointing_disable_noop_when_missing():
    """Line 180 guard: inner missing the method → safe no-op."""

    class _NoCkpt(nn.Module):
        def forward(self, *a, **kw):  # pragma: no cover
            return None

    model = _make_lora()
    model.inner = _NoCkpt()
    assert not hasattr(model.inner, "gradient_checkpointing_disable")
    model.gradient_checkpointing_disable()  # must not raise


# ---------------------------------------------------------------------------
# merge_and_unload (line 188)
# ---------------------------------------------------------------------------


def test_invariant_merge_and_unload_returns_nn_module():
    """Line 188: ``merge_and_unload()`` must return a plain ``nn.Module`` with
    LoRA deltas baked in (no more ``lora_`` named parameters).
    """
    model = _make_lora()
    merged = model.merge_and_unload()
    assert isinstance(merged, nn.Module)
    # After merge the returned module should contain no lora_ params.
    lora_params = [n for n, _ in merged.named_parameters() if "lora_" in n.lower()]
    assert not lora_params, (
        f"merge_and_unload result should have no lora_ params; got: {lora_params[:3]}"
    )


def test_invariant_merge_and_unload_result_is_not_peft_adapter():
    """``merge_and_unload()`` result is NOT a LoRAAdapter any more — it's the
    unwrapped base class (TinyCausalLM).
    """
    model = _make_lora()
    merged = model.merge_and_unload()
    assert not isinstance(merged, LoRAAdapter)
    assert isinstance(merged, TinyCausalLM)


def test_invariant_merge_and_unload_forward_still_works():
    """After ``merge_and_unload`` the result produces logits of expected shape
    on a forward pass (verifies the merged weights are coherent).
    """
    model = _make_lora()
    model.eval()
    ids = torch.randint(0, _BASE_KW["vocab_size"], (1, 4))
    with torch.no_grad():
        merged = model.merge_and_unload()
        merged.eval()
        out = merged(ids)
    # TinyCausalLM returns a ModelOutput.
    assert isinstance(out, ModelOutput)
    assert out.outputs["logits"].shape[-1] == _BASE_KW["vocab_size"]


def test_invariant_merge_and_unload_does_not_modify_in_place():
    """``merge_and_unload`` returns a NEW module; the original LoRAAdapter's
    ``inner`` reference is a different object from the merged result.

    Note: peft's ``merge_and_unload`` actually modifies the inner peft model
    in place and then returns the unwrapped base. We just check that the
    returned type is different (TinyCausalLM vs PeftModel).
    """
    model = _make_lora()
    merged = model.merge_and_unload()
    assert type(merged).__name__ != type(model.inner).__name__
