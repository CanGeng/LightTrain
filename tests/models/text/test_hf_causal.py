"""Tests for ``lighttrain.builtin_plugins.models.text.hf_causal.HFCausalLM``.

Coverage pins — uncovered lines targeted:
* Line 62  : ``raise ValueError`` when dtype string is not in ``_DTYPE_MAP``.
* Line 67  : ``trust_remote_code=True`` branch — sets key in kwargs.
* Lines 89-92 : ``forward()`` — None-default fallback for ``output_hidden_states``
                 and ``output_attentions`` from constructor defaults.
* Lines 97-103: ``inner(...)`` call in forward, passing correct tensor args.
* Lines 104-106: logits extraction from dict-like output (``out.get("logits")``).
* Lines 107-110: ``RuntimeError`` when no logits found.
* Lines 111-113: ``hidden_states`` / ``attentions`` packaging; both None and not-None paths.

General edges also covered:
* ``dtype=None`` skips torch_dtype injection.
* ``use_auth_token=True`` (bool True path) with and without env tokens.
* ``use_auth_token=None`` with and without env tokens (lines 74-75).
* ``from_pretrained_kwargs`` passthrough.
* Registry: class registered under "model" / "hf_causal".
* ``pretrained`` attribute stored.
* ``_default_output_hidden_states`` / ``_default_output_attentions`` defaults.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch

from lighttrain.builtin_plugins.models.text.hf_causal import _DTYPE_MAP, HFCausalLM
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _fake_hf_out(
    logits: torch.Tensor | None = None,
    hidden_states=None,
    attentions=None,
) -> SimpleNamespace:
    """Return a HF-output-like namespace."""
    return SimpleNamespace(
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
    )


class _FakeInner(torch.nn.Module):
    """Configurable stub for ``AutoModelForCausalLM`` inner model."""

    def __init__(
        self,
        *,
        logits: torch.Tensor | None = None,
        hidden_states=None,
        attentions=None,
        as_dict: bool = False,
    ) -> None:
        super().__init__()
        self._logits = logits
        self._hidden_states = hidden_states
        self._attentions = attentions
        self._as_dict = as_dict

    def __call__(self, **kwargs: Any):  # noqa: ARG002
        if self._as_dict:
            return {"logits": self._logits}
        return _fake_hf_out(
            logits=self._logits,
            hidden_states=self._hidden_states,
            attentions=self._attentions,
        )


class _FakeAutoModelCls:
    """Mimics ``AutoModelForCausalLM`` class (not instance)."""

    def __init__(self, inner: _FakeInner) -> None:
        self._inner = inner
        self.from_pretrained_calls: list[tuple] = []
        self.from_pretrained_kwargs: list[dict] = []

    def from_pretrained(self, pretrained: str, **kwargs: Any) -> _FakeInner:
        self.from_pretrained_calls.append((pretrained,))
        self.from_pretrained_kwargs.append(kwargs)
        return self._inner


def _stub_transformers(fake_cls: _FakeAutoModelCls):
    """Patch ``sys.modules`` so ``from transformers import AutoModelForCausalLM``
    returns *fake_cls*.
    """
    fake_module = mock.MagicMock()
    fake_module.AutoModelForCausalLM = fake_cls
    return mock.patch.dict("sys.modules", {"transformers": fake_module})


def _default_logits(B: int = 2, T: int = 4, V: int = 32) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, V)


def _make_model(
    pretrained: str = "fake/model",
    *,
    dtype: str | None = "bfloat16",
    trust_remote_code: bool = False,
    revision: str | None = None,
    use_auth_token: bool | str | None = None,
    from_pretrained_kwargs: dict | None = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    inner: _FakeInner | None = None,
    monkeypatch_env: dict | None = None,
    env_patches: dict | None = None,
) -> tuple[HFCausalLM, _FakeAutoModelCls]:
    """Build an HFCausalLM with a stubbed transformers and return both the
    model and the fake class (so callers can inspect call args).
    """
    if inner is None:
        inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(
            pretrained=pretrained,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_auth_token=use_auth_token,
            from_pretrained_kwargs=from_pretrained_kwargs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
    return model, fake_cls


# ---------------------------------------------------------------------------
# __init__ — dtype validation (line 62)
# ---------------------------------------------------------------------------


def test_invariant_unknown_dtype_raises_value_error():
    """Line 62: an unrecognised dtype string must raise ValueError immediately."""
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        with pytest.raises(ValueError, match="Unknown dtype"):
            HFCausalLM(pretrained="x/y", dtype="int8_bad")


@pytest.mark.parametrize("dtype_str", list(_DTYPE_MAP.keys()))
def test_invariant_all_known_dtypes_accepted(dtype_str):
    """Every key in ``_DTYPE_MAP`` must be accepted without error."""
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y", dtype=dtype_str)
    assert model is not None


def test_invariant_dtype_none_skips_torch_dtype_injection():
    """When dtype=None, ``torch_dtype`` must NOT appear in from_pretrained kwargs."""
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", dtype=None)
    assert "torch_dtype" not in fake_cls.from_pretrained_kwargs[0]


def test_invariant_dtype_sets_correct_torch_dtype():
    """dtype='float16' → torch_dtype=torch.float16 forwarded."""
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", dtype="float16")
    assert fake_cls.from_pretrained_kwargs[0]["torch_dtype"] == torch.float16


# ---------------------------------------------------------------------------
# __init__ — trust_remote_code branch (line 67)
# ---------------------------------------------------------------------------


def test_invariant_trust_remote_code_true_forwarded(monkeypatch):
    """Line 67: trust_remote_code=True must appear in from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", trust_remote_code=True)
    assert fake_cls.from_pretrained_kwargs[0].get("trust_remote_code") is True


def test_invariant_trust_remote_code_false_not_forwarded(monkeypatch):
    """trust_remote_code=False (default): key must NOT be in from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", trust_remote_code=False)
    assert "trust_remote_code" not in fake_cls.from_pretrained_kwargs[0]


# ---------------------------------------------------------------------------
# __init__ — revision forwarding
# ---------------------------------------------------------------------------


def test_invariant_revision_forwarded_when_set(monkeypatch):
    """revision='main' must appear in from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", revision="main")
    assert fake_cls.from_pretrained_kwargs[0]["revision"] == "main"


def test_invariant_revision_none_not_forwarded(monkeypatch):
    """revision=None: 'revision' key absent from from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", revision=None)
    assert "revision" not in fake_cls.from_pretrained_kwargs[0]


# ---------------------------------------------------------------------------
# __init__ — token plumbing (lines 69-75)
# ---------------------------------------------------------------------------


def test_invariant_use_auth_token_str_forwarded_as_token(monkeypatch):
    """use_auth_token='mytoken' (str): 'token'='mytoken' in from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token="mytoken")
    assert fake_cls.from_pretrained_kwargs[0].get("token") == "mytoken"


def test_invariant_use_auth_token_true_with_env_token(monkeypatch):
    """use_auth_token=True + HF_TOKEN set → env token forwarded as 'token'."""
    monkeypatch.setenv("HF_TOKEN", "env-secret")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token=True)
    token_val = fake_cls.from_pretrained_kwargs[0].get("token")
    assert token_val == "env-secret"


def test_invariant_use_auth_token_true_without_env_token(monkeypatch):
    """use_auth_token=True without env tokens → token=True forwarded."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token=True)
    token_val = fake_cls.from_pretrained_kwargs[0].get("token")
    assert token_val is True


def test_invariant_use_auth_token_none_with_env_token(monkeypatch):
    """Lines 74-75: use_auth_token=None + HF_TOKEN set → env token forwarded."""
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token=None)
    token_val = fake_cls.from_pretrained_kwargs[0].get("token")
    assert token_val == "from-env"


def test_invariant_use_auth_token_none_without_env_token_no_token_kwarg(monkeypatch):
    """use_auth_token=None + no env token → 'token' NOT in from_pretrained kwargs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token=None)
    assert "token" not in fake_cls.from_pretrained_kwargs[0]


def test_invariant_hugging_face_hub_token_env_fallback(monkeypatch):
    """HUGGING_FACE_HUB_TOKEN is picked up when HF_TOKEN is absent."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub-token")
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token=None)
    token_val = fake_cls.from_pretrained_kwargs[0].get("token")
    assert token_val == "hub-token"


# ---------------------------------------------------------------------------
# __init__ — from_pretrained_kwargs passthrough
# ---------------------------------------------------------------------------


def test_invariant_from_pretrained_kwargs_merged(monkeypatch):
    """Extra kwargs in from_pretrained_kwargs appear in the from_pretrained call."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        HFCausalLM(
            pretrained="x/y",
            from_pretrained_kwargs={"low_cpu_mem_usage": True},
        )
    assert fake_cls.from_pretrained_kwargs[0].get("low_cpu_mem_usage") is True


# ---------------------------------------------------------------------------
# __init__ — stored attributes
# ---------------------------------------------------------------------------


def test_invariant_pretrained_stored_as_attribute(monkeypatch):
    """``model.pretrained`` holds the pretrained name passed to constructor."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="myorg/mymodel")
    assert model.pretrained == "myorg/mymodel"


def test_invariant_default_output_flags_are_false_by_default(monkeypatch):
    """Constructor defaults: ``_default_output_hidden_states`` and
    ``_default_output_attentions`` are both False when not specified.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    assert model._default_output_hidden_states is False
    assert model._default_output_attentions is False


def test_invariant_constructor_output_flags_stored_correctly(monkeypatch):
    """output_hidden_states=True / output_attentions=True are stored on the model."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    inner = _FakeInner(logits=_default_logits())
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(
            pretrained="x/y",
            output_hidden_states=True,
            output_attentions=True,
        )
    assert model._default_output_hidden_states is True
    assert model._default_output_attentions is True


# ---------------------------------------------------------------------------
# forward — None-default fallback (lines 89-92)
# ---------------------------------------------------------------------------


def test_invariant_forward_uses_constructor_default_output_hidden_states(monkeypatch):
    """Lines 89-90: when forward() kwarg output_hidden_states is None (not passed),
    it falls back to _default_output_hidden_states=True (set in constructor).
    We verify indirectly: the ModelOutput contains hidden_states because the inner
    returns them and the default causes them to be requested.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 16
    logits = torch.zeros(B, T, V)
    hs = (torch.zeros(B, T, 8), torch.zeros(B, T, 8))
    inner = _FakeInner(logits=logits, hidden_states=hs)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y", output_hidden_states=True)
    # Verify _default flag was set
    assert model._default_output_hidden_states is True
    # forward without output_hidden_states kwarg → uses default True
    # Patch inner's forward to intercept kwargs (must patch the class-level method)
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert len(called_kwargs) == 1
    assert called_kwargs[0]["output_hidden_states"] is True


def test_invariant_forward_uses_constructor_default_output_attentions(monkeypatch):
    """Lines 91-92: when forward() kwarg output_attentions is None (not passed),
    it falls back to _default_output_attentions=True (set in constructor).
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 16
    logits = torch.zeros(B, T, V)
    attn = (torch.zeros(B, 2, T, T),)
    inner = _FakeInner(logits=logits, attentions=attn)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y", output_attentions=True)
    assert model._default_output_attentions is True
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert called_kwargs[0]["output_attentions"] is True


def test_invariant_forward_explicit_kwarg_overrides_constructor_default(monkeypatch):
    """forward(output_hidden_states=False) overrides constructor default=True."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 16
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y", output_hidden_states=True)
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(
            input_ids=torch.zeros(B, T, dtype=torch.long),
            output_hidden_states=False,
        )
    assert called_kwargs[0]["output_hidden_states"] is False


# ---------------------------------------------------------------------------
# forward — inner() call and logits extraction (lines 97-110)
# ---------------------------------------------------------------------------


def test_invariant_forward_returns_model_output():
    """Lines 97-113: forward returns a ModelOutput with 'logits' key."""
    B, T, V = 2, 4, 32
    logits = _default_logits(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    input_ids = torch.zeros(B, T, dtype=torch.long)
    out = model.forward(input_ids=input_ids)
    assert isinstance(out, ModelOutput)
    assert "logits" in out.outputs
    assert torch.equal(out.outputs["logits"], logits)


def test_invariant_forward_loss_is_always_none():
    """forward never sets loss — the LossFn is responsible for that."""
    logits = _default_logits()
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(input_ids=torch.zeros(2, 4, dtype=torch.long))
    assert out.loss is None


def test_invariant_forward_input_ids_forwarded_to_inner(monkeypatch):
    """Lines 97-103: input_ids tensor is forwarded to inner model."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 5, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    input_ids = torch.arange(B * T, dtype=torch.long).reshape(B, T)
    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=input_ids)
    assert torch.equal(called_kwargs[0]["input_ids"], input_ids)


def test_invariant_forward_attention_mask_forwarded(monkeypatch):
    """attention_mask is forwarded to inner when provided."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 4, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    input_ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -1] = 0
    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=input_ids, attention_mask=mask)
    assert torch.equal(called_kwargs[0]["attention_mask"], mask)


def test_invariant_forward_labels_not_forwarded_to_inner(monkeypatch):
    """Lines 83-96: labels kwarg is intentionally NOT forwarded to inner model
    (would cause double-shift). Verify 'labels' absent from inner's call kwargs.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 4, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    labels = torch.zeros(B, T, dtype=torch.long)
    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=torch.zeros(B, T, dtype=torch.long), labels=labels)
    assert "labels" not in called_kwargs[0]


def test_invariant_forward_logits_from_dict_output(monkeypatch):
    """Lines 105-106: when inner returns a plain dict (not an attribute-bearing object),
    logits are extracted via ``out.get('logits')``.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits, as_dict=True)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert isinstance(out, ModelOutput)
    assert torch.equal(out.outputs["logits"], logits)


def test_invariant_forward_raises_when_no_logits(monkeypatch):
    """Lines 107-110: when inner returns no logits, RuntimeError is raised."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 8
    inner = _FakeInner(logits=torch.zeros(B, T, V))
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    # Patch the inner to return None logits
    no_logits_out = SimpleNamespace(logits=None, hidden_states=None, attentions=None)

    def no_logits_call(self, **kwargs):
        return no_logits_out

    with mock.patch.object(type(model.inner), "__call__", no_logits_call):
        with pytest.raises(RuntimeError, match="returned no logits"):
            model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))


# ---------------------------------------------------------------------------
# forward — hidden_states packaging (lines 111-113)
# ---------------------------------------------------------------------------


def test_invariant_forward_hidden_states_none_when_inner_returns_none():
    """Line 116: when inner returns hidden_states=None, ModelOutput.hidden_states is None."""
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits, hidden_states=None)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert out.hidden_states is None


def test_invariant_forward_hidden_states_packaged_as_tuple():
    """Line 116: inner's hidden_states list/tuple is re-wrapped as a tuple."""
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    hs = [torch.zeros(B, T, 4), torch.zeros(B, T, 4)]
    inner = _FakeInner(logits=logits, hidden_states=hs)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(
        input_ids=torch.zeros(B, T, dtype=torch.long),
        output_hidden_states=True,
    )
    assert isinstance(out.hidden_states, tuple)
    assert len(out.hidden_states) == 2
    for hs_tensor in out.hidden_states:
        assert hs_tensor.shape == (B, T, 4)


def test_invariant_forward_attentions_none_when_inner_returns_none():
    """Line 117: when inner returns attentions=None, ModelOutput.attentions is None."""
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits, attentions=None)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert out.attentions is None


def test_invariant_forward_attentions_packaged_as_tuple():
    """Line 117: inner's attentions list is re-wrapped as a tuple."""
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    attn = [torch.zeros(B, 2, T, T), torch.zeros(B, 2, T, T)]
    inner = _FakeInner(logits=logits, attentions=attn)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(
        input_ids=torch.zeros(B, T, dtype=torch.long),
        output_attentions=True,
    )
    assert isinstance(out.attentions, tuple)
    assert len(out.attentions) == 2


def test_invariant_forward_both_hidden_states_and_attentions_populated():
    """Lines 111-113: both hidden_states and attentions are packaged when present."""
    B, T, V = 1, 4, 8
    logits = torch.zeros(B, T, V)
    hs = (torch.zeros(B, T, 4), torch.zeros(B, T, 4))
    attn = (torch.zeros(B, 2, T, T),)
    inner = _FakeInner(logits=logits, hidden_states=hs, attentions=attn)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    out = model.forward(
        input_ids=torch.zeros(B, T, dtype=torch.long),
        output_hidden_states=True,
        output_attentions=True,
    )
    assert out.hidden_states is not None and len(out.hidden_states) == 2
    assert out.attentions is not None and len(out.attentions) == 1


# ---------------------------------------------------------------------------
# forward — extra kwargs passthrough
# ---------------------------------------------------------------------------


def test_invariant_forward_extra_kwargs_forwarded_to_inner(monkeypatch):
    """forward(**kwargs) extra kwargs are passed through to inner model."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(
            input_ids=torch.zeros(B, T, dtype=torch.long),
            position_ids=torch.arange(T).unsqueeze(0),
        )
    assert "position_ids" in called_kwargs[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_invariant_model_registered_in_registry(clean_registry):
    """@register('model', 'hf_causal') makes the class discoverable via registry."""
    from lighttrain.registry import get_registry

    reg = get_registry()
    cls = reg.get("model", "hf_causal")
    assert cls is HFCausalLM


# ---------------------------------------------------------------------------
# _DTYPE_MAP completeness
# ---------------------------------------------------------------------------


def test_invariant_dtype_map_contains_expected_aliases():
    """_DTYPE_MAP must contain at least the canonical aliases for float32/16/bf16."""
    assert "float32" in _DTYPE_MAP
    assert "float16" in _DTYPE_MAP
    assert "bfloat16" in _DTYPE_MAP
    assert "fp32" in _DTYPE_MAP
    assert "fp16" in _DTYPE_MAP
    assert "bf16" in _DTYPE_MAP
    assert _DTYPE_MAP["fp32"] == torch.float32
    assert _DTYPE_MAP["fp16"] == torch.float16
    assert _DTYPE_MAP["bf16"] == torch.bfloat16


# ---------------------------------------------------------------------------
# Edge cases — forward with attention_mask=None
# ---------------------------------------------------------------------------


def test_invariant_forward_without_attention_mask_passes_none(monkeypatch):
    """forward with no attention_mask passes attention_mask=None to inner."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    B, T, V = 1, 3, 8
    logits = torch.zeros(B, T, V)
    inner = _FakeInner(logits=logits)
    fake_cls = _FakeAutoModelCls(inner)
    with _stub_transformers(fake_cls):
        model = HFCausalLM(pretrained="x/y")
    called_kwargs: list[dict] = []
    original_forward = type(model.inner).__call__

    def patched_call(self, **kwargs):
        called_kwargs.append(dict(kwargs))
        return original_forward(self, **kwargs)

    with mock.patch.object(type(model.inner), "__call__", patched_call):
        model.forward(input_ids=torch.zeros(B, T, dtype=torch.long))
    assert called_kwargs[0].get("attention_mask") is None
