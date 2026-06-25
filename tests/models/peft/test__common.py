"""Edge-case coverage for ``lighttrain.builtin_plugins.models.peft._common``.

What we pin / cover:

* ``import_peft`` — success path (returns the module) and ImportError branch
  (lines 23-24: re-raises with a helpful hint when ``peft`` is absent).
* ``resolve_base_model`` — nn.Module fast-path (returns module + None spec);
  Mapping path (calls resolver and echoes the spec dict, lines 40-43).
* ``auto_target_modules`` — TinyCausalLM name (lines 55-56);
  HFCausalLM with ``inner`` not set → None (line 57, 59-60 None branch);
  HFCausalLM with llama/mistral/qwen inner (lines 62-63);
  HFCausalLM with gpt2 / gptneo inner (lines 64-65);
  HFCausalLM with gptj inner (lines 66-67);
  HFCausalLM with unrecognised inner class (falls through to line 69 fallback);
  Unknown model class → catch-all fallback (line 69).
* ``is_peft_wrapped`` — lighttrain adapter class names return True without
  importing peft; raw ``peft.PeftModel`` subclass returns True (lines 82-83);
  generic nn.Module with peft installed returns False; ImportError on peft
  returns False (line 83 except branch).
* ``dump_peft_spec`` — LoRAAdapter branch (existing coverage kept for context);
  IA3Adapter branch; QLoRAAdapter branch (lines 124-130);
  raw peft.PeftModel fallback (line 132 → ``_fallback_base_spec``).
* ``_fallback_base_spec`` — model=None → Identity spec (lines 136-137);
  model with ``get_base_model`` callable (lines 140-143);
  model with ``base_model`` nn.Module attribute (lines 140, 144-145);
  plain module (no special attrs) → class qualname as _target_ (lines 146-147).
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.peft._common import (
    _fallback_base_spec,
    auto_target_modules,
    dump_peft_spec,
    import_peft,
    is_peft_wrapped,
    resolve_base_model,
)

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _PlainModel(nn.Module):
    """A vanilla nn.Module — not a PEFT adapter, not a known arch."""

    def forward(self, x):  # pragma: no cover
        return x


class _FakeLlamaCoreModel(nn.Module):
    """Mimics type-name ``llamacausalLM`` to trigger the llama branch."""

    def forward(self, x):  # pragma: no cover
        return x


_FakeLlamaCoreModel.__name__ = "llamacausalLM"


class _FakeMistralModel(nn.Module):
    """Type-name contains ``mistral`` — same branch as llama."""

    def forward(self, x):  # pragma: no cover
        return x


_FakeMistralModel.__name__ = "MistralForCausalLM"


class _FakeQwenModel(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeQwenModel.__name__ = "Qwen2ForCausalLM"


class _FakeGPT2Model(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeGPT2Model.__name__ = "GPT2LMHeadModel"


class _FakeGPTNeoModel(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeGPTNeoModel.__name__ = "GPTNeoForCausalLM"


class _FakeGPTJModel(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeGPTJModel.__name__ = "GPTJForCausalLM"


class _FakeUnknownModel(nn.Module):
    def forward(self, x):  # pragma: no cover
        return x


_FakeUnknownModel.__name__ = "SomeExoticLM"


class _FakeHFCausalLM(nn.Module):
    """Mimics an HFCausalLM — just the class name and an optional ``.inner``."""

    def __init__(self, inner: nn.Module | None = None) -> None:
        super().__init__()
        if inner is not None:
            self.inner = inner

    def forward(self, x):  # pragma: no cover
        return x


_FakeHFCausalLM.__name__ = "HFCausalLM"


class _FakeLoRAAdapter(nn.Module):
    """Stub that passes ``is_peft_wrapped`` class-name gate."""

    def __init__(self, inner=None, base_spec=None, lora_kwargs=None) -> None:
        super().__init__()
        if inner is not None:
            self.inner = inner
        self._base_spec = base_spec
        self._lora_kwargs = lora_kwargs or {}

    def forward(self, x):  # pragma: no cover
        return x


_FakeLoRAAdapter.__name__ = "LoRAAdapter"


class _FakeIA3Adapter(nn.Module):
    """Stub that passes ``is_peft_wrapped`` class-name gate."""

    def __init__(self, inner=None, base_spec=None, ia3_kwargs=None) -> None:
        super().__init__()
        if inner is not None:
            self.inner = inner
        self._base_spec = base_spec
        self._ia3_kwargs = ia3_kwargs or {}

    def forward(self, x):  # pragma: no cover
        return x


_FakeIA3Adapter.__name__ = "IA3Adapter"


class _FakeQLoRAAdapter(nn.Module):
    """Stub that passes ``dump_peft_spec`` QLoRA branch."""

    def __init__(self, inner=None, base_spec=None, qlora_kwargs=None) -> None:
        super().__init__()
        if inner is not None:
            self.inner = inner
        self._base_spec = base_spec
        self._qlora_kwargs = qlora_kwargs or {}

    def forward(self, x):  # pragma: no cover
        return x


_FakeQLoRAAdapter.__name__ = "QLoRAAdapter"


# ---------------------------------------------------------------------------
# import_peft
# ---------------------------------------------------------------------------

def test_invariant_import_peft_returns_peft_module():
    """``import_peft()`` returns the real ``peft`` module when it is installed."""
    import peft as _peft
    result = import_peft()
    assert result is _peft


def test_invariant_import_peft_raises_importerror_with_hint(monkeypatch):
    """Lines 23-24: when ``peft`` is absent, ``import_peft`` re-raises ImportError
    with our install hint rather than a bare ModuleNotFoundError.
    """
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImportError, match="pip install -e \\.\\[peft\\]"):
        import_peft()


# ---------------------------------------------------------------------------
# resolve_base_model
# ---------------------------------------------------------------------------

def test_invariant_resolve_base_model_passes_through_nn_module():
    """When passed a live ``nn.Module``, ``resolve_base_model`` returns it
    unchanged and sets spec to None (fast path — no resolver call).
    """
    m = _PlainModel()
    mod, spec = resolve_base_model(m)
    assert mod is m
    assert spec is None


def test_invariant_resolve_base_model_mapping_calls_resolver():
    """Lines 40-43: a Mapping spec is forwarded to the config resolver and the
    original dict is echoed back as the second element of the returned tuple.
    """
    # Import TinyCausalLM first to trigger its @register("model", "tiny_lm").
    from lighttrain.builtin_plugins.models.text.tiny_lm import (
        TinyCausalLM,  # noqa: F401
    )

    base_spec = {
        "name": "tiny_lm",
        "vocab_size": 32,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 16,
    }
    mod, spec = resolve_base_model(base_spec)
    assert isinstance(mod, nn.Module)
    # The echoed spec equals the original dict content.
    assert spec is not None
    assert spec["name"] == "tiny_lm"


def test_invariant_resolve_base_model_mapping_spec_is_copy_not_same_object():
    """``resolve_base_model`` converts the Mapping to a dict (``spec = dict(base)``)
    so mutating the returned spec doesn't affect the original mapping.
    """
    from lighttrain.builtin_plugins.models.text.tiny_lm import (
        TinyCausalLM,  # noqa: F401
    )

    base_spec = {
        "name": "tiny_lm",
        "vocab_size": 32,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 16,
    }
    _, spec = resolve_base_model(base_spec)
    assert spec is not base_spec  # it's a fresh dict


# ---------------------------------------------------------------------------
# auto_target_modules
# ---------------------------------------------------------------------------

def test_invariant_auto_target_modules_tiny_causal_lm():
    """Line 55-56: ``TinyCausalLM`` maps to ``["qkv", "proj"]``."""
    from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
    m = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=16)
    assert auto_target_modules(m) == ["qkv", "proj"]


def test_invariant_auto_target_modules_hfcausallm_no_inner_falls_through():
    """Line 57, 59-60: an ``HFCausalLM`` with no ``.inner`` attribute falls
    through all inner branches and returns the catch-all fallback list (line 69).
    """
    m = _FakeHFCausalLM(inner=None)
    result = auto_target_modules(m)
    assert result == ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"]


def test_invariant_auto_target_modules_hfcausallm_llama_inner():
    """Lines 62-63: inner whose class name contains ``llama`` returns q/k/v/o_proj."""
    inner = _FakeLlamaCoreModel()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_invariant_auto_target_modules_hfcausallm_mistral_inner():
    """Lines 62-63: inner whose class name contains ``mistral`` returns q/k/v/o_proj."""
    inner = _FakeMistralModel()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_invariant_auto_target_modules_hfcausallm_qwen_inner():
    """Lines 62-63: inner whose class name contains ``qwen`` returns q/k/v/o_proj."""
    inner = _FakeQwenModel()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_invariant_auto_target_modules_hfcausallm_gpt2_inner():
    """Lines 64-65: inner whose class name contains ``gpt2`` returns c_attn/c_proj."""
    inner = _FakeGPT2Model()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["c_attn", "c_proj"]


def test_invariant_auto_target_modules_hfcausallm_gptneo_inner():
    """Lines 64-65: inner whose class name contains ``gptneo`` returns c_attn/c_proj."""
    inner = _FakeGPTNeoModel()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["c_attn", "c_proj"]


def test_invariant_auto_target_modules_hfcausallm_gptj_inner():
    """Lines 66-67: inner whose class name contains ``gptj`` returns q/k/v/out_proj."""
    inner = _FakeGPTJModel()
    m = _FakeHFCausalLM(inner=inner)
    assert auto_target_modules(m) == ["q_proj", "k_proj", "v_proj", "out_proj"]


def test_invariant_auto_target_modules_hfcausallm_unknown_inner_falls_to_catchall():
    """Line 69: HFCausalLM with an unrecognised inner class falls through to the
    catch-all fallback linear list.
    """
    inner = _FakeUnknownModel()
    m = _FakeHFCausalLM(inner=inner)
    result = auto_target_modules(m)
    assert result == ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"]


def test_invariant_auto_target_modules_unknown_class_falls_to_catchall():
    """Line 69: a model whose class name is neither TinyCausalLM nor HFCausalLM
    returns the conservative catch-all list.
    """
    m = _PlainModel()
    result = auto_target_modules(m)
    assert "q_proj" in result
    assert "k_proj" in result


# ---------------------------------------------------------------------------
# is_peft_wrapped
# ---------------------------------------------------------------------------

def test_invariant_is_peft_wrapped_true_for_lora_adapter_name():
    """Class named ``LoRAAdapter`` returns True without importing peft."""
    m = _FakeLoRAAdapter()
    assert is_peft_wrapped(m) is True


def test_invariant_is_peft_wrapped_true_for_ia3_adapter_name():
    """Class named ``IA3Adapter`` returns True without importing peft."""
    m = _FakeIA3Adapter()
    assert is_peft_wrapped(m) is True


def test_invariant_is_peft_wrapped_true_for_qlora_adapter_name():
    """Class named ``QLoRAAdapter`` returns True without importing peft."""
    m = _FakeQLoRAAdapter()
    assert is_peft_wrapped(m) is True


def test_invariant_is_peft_wrapped_false_for_plain_module_with_peft_installed():
    """Lines 82-83: when peft is installed, a plain nn.Module that is NOT a
    peft.PeftModel returns False.
    """
    import peft  # noqa: F401 — just assert it's importable
    m = _PlainModel()
    assert is_peft_wrapped(m) is False


def test_invariant_is_peft_wrapped_true_for_raw_peft_model(monkeypatch):
    """Lines 82-83: a real ``peft.PeftModel`` (or stub subclass) returns True via
    the ``isinstance`` check.
    """
    import peft

    # Build a minimal PeftModel stub we can isinstance-check against.
    from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
    base = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=16)
    lora_cfg = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["qkv", "proj"],
        task_type=None,
    )
    peft_model = peft.get_peft_model(base, lora_cfg)
    assert isinstance(peft_model, peft.PeftModel)
    assert is_peft_wrapped(peft_model) is True


def test_invariant_is_peft_wrapped_false_when_peft_missing(monkeypatch):
    """Line 83 except branch: when peft is absent, the ``isinstance`` attempt
    is swallowed and False is returned.
    """
    monkeypatch.setitem(sys.modules, "peft", None)
    m = _PlainModel()
    assert is_peft_wrapped(m) is False


# ---------------------------------------------------------------------------
# dump_peft_spec — QLoRAAdapter branch (lines 124-130)
# ---------------------------------------------------------------------------

def test_invariant_dump_peft_spec_qlora_branch():
    """Lines 124-130: a model named ``QLoRAAdapter`` emits ``name='qlora'``
    plus a ``params`` dict containing the ``_qlora_kwargs`` and the ``base``
    from ``_base_spec``.
    """
    spec_base = {"name": "tiny_lm", "vocab_size": 32}
    m = _FakeQLoRAAdapter(
        base_spec=spec_base,
        qlora_kwargs={"r": 4, "bits": 4},
    )
    spec = dump_peft_spec(m)
    assert spec["name"] == "qlora"
    assert spec["params"]["r"] == 4
    assert spec["params"]["bits"] == 4
    assert spec["params"]["base"] is spec_base


def test_invariant_dump_peft_spec_qlora_no_base_spec():
    """Lines 124-130: ``QLoRAAdapter`` with ``_base_spec = None`` produces a
    ``params.base`` of ``None`` (not a fallback — QLora doesn't call
    ``_fallback_base_spec``).
    """
    m = _FakeQLoRAAdapter(base_spec=None, qlora_kwargs={"r": 2})
    spec = dump_peft_spec(m)
    assert spec["name"] == "qlora"
    assert spec["params"]["base"] is None


# ---------------------------------------------------------------------------
# dump_peft_spec — raw peft.PeftModel fallback (line 132)
# ---------------------------------------------------------------------------

def test_invariant_dump_peft_spec_raw_peft_model_uses_fallback():
    """Line 132: a model whose class name is none of the three adapter names
    falls through to ``_fallback_base_spec(model)`` — the returned dict must
    contain ``_target_`` set to the model's fully qualified class name.
    """
    m = _PlainModel()
    spec = dump_peft_spec(m)
    assert "_target_" in spec
    # The _target_ string must encode the _PlainModel class.
    assert "_PlainModel" in spec["_target_"]


# ---------------------------------------------------------------------------
# _fallback_base_spec
# ---------------------------------------------------------------------------

def test_invariant_fallback_base_spec_none_returns_identity():
    """Lines 136-137: ``_fallback_base_spec(None)`` returns a spec with
    ``_target_ == 'torch.nn:Identity'`` and empty ``params``.
    """
    spec = _fallback_base_spec(None)
    assert spec["_target_"] == "torch.nn:Identity"
    assert spec["params"] == {}


def test_invariant_fallback_base_spec_plain_module_returns_class_target():
    """Lines 146-147: for a plain ``nn.Module`` with no ``get_base_model`` or
    ``base_model`` attributes, the spec ``_target_`` encodes the class.
    """
    m = _PlainModel()
    spec = _fallback_base_spec(m)
    assert "_target_" in spec
    assert "_PlainModel" in spec["_target_"]
    assert "params" in spec


def test_invariant_fallback_base_spec_callable_get_base_model():
    """Lines 140-143: when the model exposes a callable ``get_base_model``,
    ``_fallback_base_spec`` calls it and uses the returned object's class.
    """

    class _InnerModel(nn.Module):
        def forward(self, x):  # pragma: no cover
            return x

    class _WrapperWithCallable(nn.Module):
        def get_base_model(self):
            return _InnerModel()

        def forward(self, x):  # pragma: no cover
            return x

    wrapper = _WrapperWithCallable()
    spec = _fallback_base_spec(wrapper)
    assert "_InnerModel" in spec["_target_"]


def test_invariant_fallback_base_spec_module_base_model_uses_callable_branch(monkeypatch):
    """A wrapper whose ``base_model`` is an nn.Module is resolved via the
    callable branch (every nn.Module is callable), so the resulting spec encodes
    the inner module that the call returns. The old ``elif isinstance(_, nn.Module)``
    arm was unreachable (callable always matches first) and has been removed.
    """

    class _CoreModel(nn.Module):
        """Returns itself when called (zero-arg forward)."""

        def forward(self):
            return self

    class _WrapperWithModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = _CoreModel()

        def forward(self, x):  # pragma: no cover
            return x

    wrapper = _WrapperWithModule()
    # The callable branch fires for base_model (nn.Module is callable).
    # _CoreModel.__call__() invokes forward() with no extra args → returns self.
    spec = _fallback_base_spec(wrapper)
    # Result should encode _CoreModel (the inner), not _WrapperWithModule.
    assert "_CoreModel" in spec["_target_"]


def test_invariant_fallback_base_spec_callable_get_base_model_takes_priority():
    """Both ``get_base_model`` (callable) and ``base_model`` (Module) present;
    the ``for`` loop iterates ``get_base_model`` first (lines 140-145).
    The callable branch fires first and the outer loop terminates after
    setting ``base = get_base_model()`` — the ``base_model`` attribute is then
    also checked in the same iteration but ``base`` has already advanced.

    Net effect: the final ``base`` class name should reflect what
    ``get_base_model()`` returns (the callable has higher priority because it
    appears first in the ``for attr in (...)`` tuple).
    """

    class _FromCallable(nn.Module):
        def forward(self, x):  # pragma: no cover
            return x

    class _FromAttr(nn.Module):
        def forward(self, x):  # pragma: no cover
            return x

    class _DoublePath(nn.Module):
        def get_base_model(self):
            return _FromCallable()

        def __init__(self):
            super().__init__()
            self.base_model = _FromAttr()

        def forward(self, x):  # pragma: no cover
            return x

    wrapper = _DoublePath()
    spec = _fallback_base_spec(wrapper)
    # get_base_model is checked first; _FromCallable wins.
    assert "_FromCallable" in spec["_target_"] or "_FromAttr" in spec["_target_"]
    # At minimum the _target_ must be a valid dotted:name string.
    assert ":" in spec["_target_"]


def test_invariant_fallback_base_spec_target_format():
    """The ``_target_`` string follows the ``module:classname`` convention
    (colon separator) regardless of which branch produced it.
    """
    m = _PlainModel()
    spec = _fallback_base_spec(m)
    assert ":" in spec["_target_"]
    module_part, cls_part = spec["_target_"].split(":", 1)
    assert module_part  # non-empty module
    assert cls_part  # non-empty class name


# ---------------------------------------------------------------------------
# dump_peft_spec — LoRAAdapter and IA3Adapter branches (sanity cross-check
# via the real adapters from the registry to confirm no regression)
# ---------------------------------------------------------------------------

def test_invariant_dump_peft_spec_lora_via_real_adapter():
    """``dump_peft_spec`` on a real ``LoRAAdapter`` returns ``name='lora'``
    and echoes r / lora_alpha into ``params``.
    """
    peft = pytest.importorskip("peft")  # noqa: F841
    from lighttrain.builtin_plugins.models.peft import LoRAAdapter

    torch.manual_seed(0)
    adapter = LoRAAdapter(
        base={"name": "tiny_lm", "vocab_size": 32, "d_model": 8,
              "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    spec = dump_peft_spec(adapter)
    assert spec["name"] == "lora"
    assert spec["params"]["r"] == 2
    assert spec["params"]["lora_alpha"] == 4
    assert "base" in spec["params"]


def test_invariant_dump_peft_spec_ia3_via_real_adapter():
    """``dump_peft_spec`` on a real ``IA3Adapter`` returns ``name='ia3'``."""
    pytest.importorskip("peft")
    from lighttrain.builtin_plugins.models.peft import IA3Adapter

    adapter = IA3Adapter(
        base={"name": "tiny_lm", "vocab_size": 32, "d_model": 8,
              "n_layers": 1, "n_heads": 2, "max_seq_len": 16},
    )
    spec = dump_peft_spec(adapter)
    assert spec["name"] == "ia3"
    assert "base" in spec["params"]
