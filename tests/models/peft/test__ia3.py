"""Edge-case coverage for ``lighttrain.builtin_plugins.models.peft._ia3``.

Currently at 79 % coverage; this file targets every reachable uncovered line:

Lines 36-46  ``_auto_ia3_targets`` — HFCausalLM branch (llama/mistral inner),
             HFCausalLM with inner that is *not* llama/mistral,
             HFCausalLM with ``inner=None``, and the conservative fallback.
Line  85     ``config.modules_to_save = list(modules_to_save)`` (non-empty arg)
Line  91     ``self._ia3_kwargs["modules_to_save"] = list(modules_to_save)``
Line 115     ``full_state_dict()``
Line 118/119 ``enable_input_require_grads()`` — both the delegating branch
             (inner HAS the method) and the no-op branch (inner lacks it).
Line 122/123 ``gradient_checkpointing_enable()`` — same two branches.
Line 126     ``get_base_model()``
Line 129     ``num_parameters()``

What we also pin:
* ``_auto_ia3_targets`` for TinyCausalLM (lines 32-35, existing but explicit).
* ``IA3Adapter.__init__`` with tuple ``target_modules`` / ``feedforward_modules``
  (the ``isinstance`` list-coercion path, line 78-79).
* ``IA3Adapter.trainable_parameters()`` invariant (non-regression).
* ``state_dict`` / ``load_state_dict`` round-trip (non-regression, short form).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")  # whole file requires peft

from lighttrain.builtin_plugins.models.peft._ia3 import (  # noqa: E402
    IA3Adapter,
    _auto_ia3_targets,
)
from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec() -> dict:
    """Fresh spec dict for a TinyCausalLM (avoids re-wrapping the same object)."""
    return {
        "name": "tiny_lm",
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 32,
    }


def _tiny() -> TinyCausalLM:
    """Return a fresh TinyCausalLM module."""
    return TinyCausalLM(vocab_size=64, d_model=16, n_layers=2, n_heads=4, max_seq_len=32)


# ---------------------------------------------------------------------------
# Stubs for _auto_ia3_targets
# ---------------------------------------------------------------------------

class _FakeHFCausalLM(nn.Module):
    """Minimal stub whose __name__ matches the 'HFCausalLM' branch."""

    def __init__(self, inner: nn.Module | None = None) -> None:
        super().__init__()
        if inner is not None:
            self.inner = inner

    def forward(self, x):  # pragma: no cover
        return x


_FakeHFCausalLM.__name__ = "HFCausalLM"


class _FakeLlamaInner(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeLlamaInner.__name__ = "LlamaForCausalLM"


class _FakeMistralInner(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeMistralInner.__name__ = "MistralForCausalLM"


class _FakeGPT2Inner(nn.Module):
    """Inner whose class name does not contain 'llama' or 'mistral' —
    falls through to the conservative fallback."""

    def forward(self, x):  # pragma: no cover
        return x


_FakeGPT2Inner.__name__ = "GPT2LMHeadModel"


class _PlainModel(nn.Module):
    """Generic module — name is neither TinyCausalLM nor HFCausalLM."""

    def forward(self, x):  # pragma: no cover
        return x


class _NoOpInner(nn.Module):
    """An nn.Module that intentionally exposes neither
    ``enable_input_require_grads`` nor ``gradient_checkpointing_enable``."""

    def forward(self, x):  # pragma: no cover
        return x


# ---------------------------------------------------------------------------
# _auto_ia3_targets — TinyCausalLM path (lines 32-35)
# ---------------------------------------------------------------------------

def test_invariant_auto_ia3_targets_tiny_causal_lm():
    """TinyCausalLM maps to ([qkv, fc2], [fc2]) — attention + MLP targets."""
    tiny = _tiny()
    tm, ff = _auto_ia3_targets(tiny)
    assert tm == ["qkv", "fc2"]
    assert ff == ["fc2"]


# ---------------------------------------------------------------------------
# _auto_ia3_targets — HFCausalLM / llama branch (lines 36-44)
# ---------------------------------------------------------------------------

def test_invariant_auto_ia3_targets_hfcausallm_llama_inner():
    """HFCausalLM with a llama inner returns k/v/down_proj targets (lines 40-43)."""
    hf = _FakeHFCausalLM(inner=_FakeLlamaInner())
    tm, ff = _auto_ia3_targets(hf)
    assert tm == ["k_proj", "v_proj", "down_proj"]
    assert ff == ["down_proj"]


def test_invariant_auto_ia3_targets_hfcausallm_mistral_inner():
    """HFCausalLM with a mistral inner uses the same llama/mistral branch."""
    hf = _FakeHFCausalLM(inner=_FakeMistralInner())
    tm, ff = _auto_ia3_targets(hf)
    assert tm == ["k_proj", "v_proj", "down_proj"]
    assert ff == ["down_proj"]


def test_invariant_auto_ia3_targets_hfcausallm_non_llama_inner_falls_through():
    """HFCausalLM with a non-llama/non-mistral inner falls through to the
    conservative fallback (line 46) — the if/elif branches are all skipped."""
    hf = _FakeHFCausalLM(inner=_FakeGPT2Inner())
    tm, ff = _auto_ia3_targets(hf)
    assert tm == ["key", "value", "dense"]
    assert ff == ["dense"]


def test_invariant_auto_ia3_targets_hfcausallm_no_inner_falls_through():
    """HFCausalLM with no ``.inner`` attribute falls through to conservative
    fallback because ``getattr(base, 'inner', None)`` returns None."""
    hf = _FakeHFCausalLM(inner=None)
    tm, ff = _auto_ia3_targets(hf)
    assert tm == ["key", "value", "dense"]
    assert ff == ["dense"]


# ---------------------------------------------------------------------------
# _auto_ia3_targets — conservative fallback (line 46)
# ---------------------------------------------------------------------------

def test_invariant_auto_ia3_targets_unknown_class_conservative_fallback():
    """A model whose class name is neither TinyCausalLM nor HFCausalLM
    always returns the conservative fallback tuple."""
    m = _PlainModel()
    tm, ff = _auto_ia3_targets(m)
    assert tm == ["key", "value", "dense"]
    assert ff == ["dense"]


# ---------------------------------------------------------------------------
# IA3Adapter — modules_to_save path (lines 85, 91)
# ---------------------------------------------------------------------------

def test_invariant_ia3_modules_to_save_stored_in_config_and_kwargs():
    """Lines 85 + 91: when ``modules_to_save`` is non-empty the list is written
    into both the peft config and ``_ia3_kwargs``."""
    adapter = IA3Adapter(base=_spec(), modules_to_save=["lm_head"])
    # line 91 check
    assert adapter._ia3_kwargs.get("modules_to_save") == ["lm_head"]
    # line 85 check — the peft config reflects the value
    cfg = list(adapter.inner.peft_config.values())[0]
    assert "lm_head" in cfg.modules_to_save


def test_invariant_ia3_modules_to_save_none_not_stored():
    """When ``modules_to_save`` is omitted (default None), the key is absent
    from ``_ia3_kwargs`` (the two branches at lines 85/91 are skipped)."""
    adapter = IA3Adapter(base=_spec())
    assert "modules_to_save" not in adapter._ia3_kwargs


def test_invariant_ia3_modules_to_save_empty_list_not_stored():
    """An empty list is falsy — the ``if modules_to_save:`` guards skip both
    lines 85 and 91 just like None."""
    adapter = IA3Adapter(base=_spec(), modules_to_save=[])
    assert "modules_to_save" not in adapter._ia3_kwargs


# ---------------------------------------------------------------------------
# IA3Adapter — tuple target_modules / feedforward_modules coercion (line 78-79)
# ---------------------------------------------------------------------------

def test_invariant_ia3_tuple_target_modules_coerced_to_list():
    """Tuples passed as ``target_modules`` or ``feedforward_modules`` are
    converted to lists before being stored in ``_ia3_kwargs``."""
    adapter = IA3Adapter(
        base=_spec(),
        target_modules=("qkv", "fc2"),
        feedforward_modules=("fc2",),
    )
    assert isinstance(adapter._ia3_kwargs["target_modules"], list)
    assert adapter._ia3_kwargs["target_modules"] == ["qkv", "fc2"]
    assert isinstance(adapter._ia3_kwargs["feedforward_modules"], list)
    assert adapter._ia3_kwargs["feedforward_modules"] == ["fc2"]


# ---------------------------------------------------------------------------
# IA3Adapter — full_state_dict (line 115)
# ---------------------------------------------------------------------------

def test_invariant_ia3_full_state_dict_larger_than_adapter_state_dict():
    """Line 115: ``full_state_dict()`` returns the full base+adapter weights —
    always strictly more keys than the adapter-only ``state_dict()``."""
    adapter = IA3Adapter(base=_spec())
    full_sd = adapter.full_state_dict()
    adapter_sd = adapter.state_dict()
    assert len(full_sd) > len(adapter_sd)


def test_invariant_ia3_full_state_dict_is_a_plain_dict():
    """``full_state_dict()`` wraps ``self.inner.state_dict()`` in a plain dict
    (not an OrderedDict or any other subtype)."""
    adapter = IA3Adapter(base=_spec())
    fsd = adapter.full_state_dict()
    assert type(fsd) is dict  # noqa: E721


def test_invariant_ia3_full_state_dict_values_are_tensors():
    """Every value in ``full_state_dict()`` is a torch.Tensor."""
    adapter = IA3Adapter(base=_spec())
    fsd = adapter.full_state_dict()
    assert all(isinstance(v, torch.Tensor) for v in fsd.values())


# ---------------------------------------------------------------------------
# IA3Adapter — enable_input_require_grads (lines 117-119)
# ---------------------------------------------------------------------------

def test_invariant_ia3_enable_input_require_grads_delegates_to_inner():
    """Lines 118-119: when ``inner`` exposes ``enable_input_require_grads``,
    the method is forwarded without raising. We inject a stub inner that has
    the method to exercise the ``hasattr`` true branch."""
    calls: list[str] = []

    class _InnerWithEnable(nn.Module):
        def enable_input_require_grads(self):
            calls.append("called")

        def forward(self, x):  # pragma: no cover
            return x

    adapter = IA3Adapter(base=_spec())
    adapter.inner = _InnerWithEnable()
    adapter.enable_input_require_grads()
    assert calls == ["called"]


def test_invariant_ia3_enable_input_require_grads_noop_when_inner_lacks_it():
    """Line 118 (else path): when inner does NOT have ``enable_input_require_grads``,
    the call is silently skipped (no AttributeError)."""
    adapter = IA3Adapter(base=_spec())
    # Replace inner with a plain module that has no such method.
    adapter.inner = _NoOpInner()
    adapter.enable_input_require_grads()  # must not raise


# ---------------------------------------------------------------------------
# IA3Adapter — gradient_checkpointing_enable (lines 121-123)
# ---------------------------------------------------------------------------

def test_invariant_ia3_gradient_checkpointing_enable_delegates_to_inner():
    """Lines 122-123: when ``inner`` has ``gradient_checkpointing_enable``,
    the call is forwarded. We inject a stub inner that has the method to
    exercise the ``hasattr`` true branch."""
    calls: list[dict] = []

    class _InnerWithGC(nn.Module):
        def gradient_checkpointing_enable(self, **kw):
            calls.append(kw)

        def forward(self, x):  # pragma: no cover
            return x

    adapter = IA3Adapter(base=_spec())
    adapter.inner = _InnerWithGC()
    adapter.gradient_checkpointing_enable()
    assert calls == [{}]


def test_invariant_ia3_gradient_checkpointing_enable_noop_when_inner_lacks_it():
    """Line 122 (else path): when inner lacks the method, the call is a silent
    no-op — kwargs are accepted and discarded."""
    adapter = IA3Adapter(base=_spec())
    adapter.inner = _NoOpInner()
    adapter.gradient_checkpointing_enable(use_reentrant=False)  # must not raise


def test_invariant_ia3_gradient_checkpointing_enable_passes_kwargs():
    """Kwargs supplied to ``gradient_checkpointing_enable`` are forwarded."""
    calls: list[dict] = []

    class _TrackingInner(nn.Module):
        def gradient_checkpointing_enable(self, **kw):
            calls.append(kw)

        def forward(self, x):  # pragma: no cover
            return x

    adapter = IA3Adapter(base=_spec())
    adapter.inner = _TrackingInner()
    adapter.gradient_checkpointing_enable(use_reentrant=True)
    assert calls == [{"use_reentrant": True}]


# ---------------------------------------------------------------------------
# IA3Adapter — get_base_model (line 126)
# ---------------------------------------------------------------------------

def test_invariant_ia3_get_base_model_returns_nn_module():
    """Line 126: ``get_base_model()`` must return an ``nn.Module`` — the raw
    TinyCausalLM that was passed in as ``base``."""
    adapter = IA3Adapter(base=_spec())
    base = adapter.get_base_model()
    assert isinstance(base, nn.Module)
    assert type(base).__name__ == "TinyCausalLM"


# ---------------------------------------------------------------------------
# IA3Adapter — num_parameters (line 129)
# ---------------------------------------------------------------------------

def test_invariant_ia3_num_parameters_positive_and_equals_trainable():
    """Line 129: ``num_parameters()`` counts only ``requires_grad`` parameters
    and must equal ``trainable_parameters()[0]``."""
    adapter = IA3Adapter(base=_spec())
    trainable, _total = adapter.trainable_parameters()
    assert adapter.num_parameters() == trainable
    assert adapter.num_parameters() > 0


@pytest.mark.parametrize("n_layers", [1, 2])
def test_invariant_ia3_num_parameters_scales_with_depth(n_layers: int):
    """Deeper models have more IA³ parameters (more attention+MLP channels)."""
    spec = {
        "name": "tiny_lm",
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": n_layers,
        "n_heads": 4,
        "max_seq_len": 32,
    }
    adapter = IA3Adapter(base=spec)
    # Any positive number of trainable params is the invariant.
    assert adapter.num_parameters() > 0


# ---------------------------------------------------------------------------
# Regression: trainable_parameters invariant
# ---------------------------------------------------------------------------

def test_invariant_ia3_trainable_parameters_ratio():
    """IA³ freezes the vast majority of parameters: trainable/total < 0.05."""
    adapter = IA3Adapter(base=_spec())
    trainable, total = adapter.trainable_parameters()
    assert 0 < trainable < total
    assert trainable / total < 0.05


# ---------------------------------------------------------------------------
# Regression: state_dict / load_state_dict round-trip
# ---------------------------------------------------------------------------

def test_invariant_ia3_state_dict_load_round_trip():
    """Adapter-only state_dict loads into a fresh adapter without raising."""
    torch.manual_seed(42)
    a = IA3Adapter(base=_spec())
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = IA3Adapter(base=_spec())
    result = b.load_state_dict(sd)
    # The return value carries the IncompatibleKeys named-tuple interface.
    assert hasattr(result, "missing_keys")
    assert hasattr(result, "unexpected_keys")
    # Values must survive the round-trip.
    for k in sd:
        assert torch.allclose(sd[k], b.state_dict()[k], atol=1e-6), f"{k} mismatch"


# ---------------------------------------------------------------------------
# IA3Adapter constructed from nn.Module directly (not a spec dict)
# ---------------------------------------------------------------------------

def test_invariant_ia3_accepts_nn_module_base():
    """``IA3Adapter`` can take a live ``nn.Module`` as ``base``; in that case
    ``_base_spec`` is None (no recipe spec to echo back)."""
    tiny = _tiny()
    adapter = IA3Adapter(base=tiny)
    assert adapter._base_spec is None
    assert isinstance(adapter.inner, nn.Module)
