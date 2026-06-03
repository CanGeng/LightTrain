"""HF adapter dry-run: arg propagation + env-token plumbing without network."""

from __future__ import annotations

from unittest import mock


def _import_class():
    """Import HFCausalLM once; subsequent calls are cheap."""
    from lighttrain.builtin_plugins.models.adapters.hf_causal import HFCausalLM

    return HFCausalLM


def _stub_transformers(fake_cls):
    fake_module = mock.MagicMock()
    fake_module.AutoModelForCausalLM = fake_cls
    return mock.patch.dict("sys.modules", {"transformers": fake_module})


def test_hf_causal_passes_pretrained_and_token_to_from_pretrained(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    HFCausalLM = _import_class()

    fake_cls = mock.MagicMock()
    fake_cls.from_pretrained.return_value = mock.MagicMock(spec=["state_dict", "forward"])

    with _stub_transformers(fake_cls):
        HFCausalLM(
            pretrained="meta-llama/fake-7b",
            dtype="bfloat16",
            trust_remote_code=False,
            revision="main",
        )

    fake_cls.from_pretrained.assert_called_once()
    args, kwargs = fake_cls.from_pretrained.call_args
    assert (kwargs.get("pretrained_model_name_or_path") == "meta-llama/fake-7b") or (
        args and args[0] == "meta-llama/fake-7b"
    )
    # trust_remote_code is only forwarded when True; False stays implicit.
    assert kwargs.get("trust_remote_code", False) is False
    assert kwargs.get("revision") == "main"
    forwarded = {
        k: v for k, v in kwargs.items() if "token" in k.lower() or "auth" in k.lower()
    }
    assert any(v == "secret-token" for v in forwarded.values()), forwarded


def test_hf_causal_respects_explicit_use_auth_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    HFCausalLM = _import_class()

    fake_cls = mock.MagicMock()
    fake_cls.from_pretrained.return_value = mock.MagicMock(spec=["state_dict", "forward"])

    with _stub_transformers(fake_cls):
        HFCausalLM(pretrained="x/y", use_auth_token="explicit")

    _, kwargs = fake_cls.from_pretrained.call_args
    forwarded = {
        k: v for k, v in kwargs.items() if "token" in k.lower() or "auth" in k.lower()
    }
    assert any(v == "explicit" for v in forwarded.values()), forwarded
