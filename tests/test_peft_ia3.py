"""IA3Adapter — DESIGN §8.4 (M5)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from lighttrain.builtin_plugins.models.peft import IA3Adapter, dump_peft_spec  # noqa: E402


def _spec_for_tiny() -> dict:
    return {
        "name": "tiny_lm",
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 32,
    }


def _make_ia3():
    return IA3Adapter(base=_spec_for_tiny())


def test_ia3_wraps_base_and_freezes_base_params():
    model = _make_ia3()
    trainable, total = model.trainable_parameters()
    assert 0 < trainable < total
    assert trainable / total < 0.05  # IA³ is tiny


def test_ia3_forward_returns_modeloutput_with_logits():
    from lighttrain.protocols import ModelOutput

    model = _make_ia3()
    ids = torch.randint(0, 64, (1, 4))
    out = model(input_ids=ids)
    assert isinstance(out, ModelOutput)
    assert "logits" in out.outputs


def test_ia3_dump_spec_is_recoverable():
    model = _make_ia3()
    spec = dump_peft_spec(model)
    assert spec["name"] == "ia3"
    assert "base" in spec["params"]
    assert "target_modules" in spec["params"]
    assert "feedforward_modules" in spec["params"]


def test_ia3_state_dict_is_adapter_only_and_round_trips():
    a = _make_ia3()
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = _make_ia3()
    b.load_state_dict(sd)
    sd_b = b.state_dict()
    for k in sd:
        assert torch.allclose(sd[k], sd_b[k], atol=1e-6), f"{k} mismatch"
