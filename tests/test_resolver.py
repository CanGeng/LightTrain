"""Tests for _import_target() right-peel resolver + sft_chat_hf.yaml recipe smoke."""

from __future__ import annotations

import pytest

from lighttrain.config._exceptions import ConfigResolveError
from lighttrain.config._resolver import _import_target, resolve


# ---------------------------------------------------------------------------
# Unit: multi-level dotted paths
# ---------------------------------------------------------------------------

def test_two_level_import():
    from decimal import Decimal
    result = _import_target("decimal.Decimal")
    assert result is Decimal


def test_three_level_import():
    import os.path
    result = _import_target("os.path.join")
    assert callable(result)
    assert result("a", "b") == os.path.join("a", "b")


def test_four_level_import():
    from lighttrain.data.core.collators import CausalLMCollator
    result = _import_target("lighttrain.data.core.collators.CausalLMCollator")
    assert result is CausalLMCollator


def test_transformers_three_level():
    """Core regression: sft_chat_hf.yaml _target_ must resolve without ConfigResolveError."""
    result = _import_target("transformers.AutoTokenizer.from_pretrained")
    assert callable(result)


def test_nonexistent_package_raises():
    with pytest.raises(ConfigResolveError):
        _import_target("_nonexistent_pkg_xyz.Foo")


def test_nonexistent_attribute_raises():
    with pytest.raises(ConfigResolveError):
        _import_target("os.path.nonexistent_attr_xyz")


def test_colon_syntax_two_part():
    """Colon escape-hatch: 'pkg.module:Attr'."""
    from decimal import Decimal
    result = _import_target("decimal:Decimal")
    assert result is Decimal


# ---------------------------------------------------------------------------
# Recipe-level smoke: sft_chat_hf.yaml tokenizer _target_ resolves
# ---------------------------------------------------------------------------

def test_sft_chat_hf_tokenizer_resolve(monkeypatch):
    """Resolve the tokenizer spec from sft_chat_hf.yaml without hitting the network.

    This is the direct regression protection for REVIEW_ROUND3 finding #2:
    the _target_: transformers.AutoTokenizer.from_pretrained path must not
    raise ConfigResolveError.
    """
    import transformers

    class _StubTokenizer:
        def encode(self, text, **_):
            return list(text.encode("utf-8"))[:64]
        def decode(self, ids, **_):
            return bytes(ids).decode("utf-8", errors="replace")
        def __call__(self, *args, **kwargs):
            return {}

    stub = _StubTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **kw: stub)

    # This is exactly the spec from recipes/sft_chat_hf.yaml:56
    spec = {
        "_target_": "transformers.AutoTokenizer.from_pretrained",
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    result = resolve(spec)
    assert result is stub, "resolve() should return the stub returned by from_pretrained"
