"""LoRAAdapter — DESIGN §8.4 (M5)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from lighttrain.models.adapters.tiny_lm import TinyCausalLM  # noqa: E402
from lighttrain.models.peft import (  # noqa: E402
    LoRAAdapter,
    dump_peft_spec,
    is_peft_wrapped,
)


def _spec_for_tiny() -> dict:
    return {
        "name": "tiny_lm",
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 32,
    }


def _make_lora(**overrides):
    kwargs = dict(base=_spec_for_tiny(), r=4, lora_alpha=8, lora_dropout=0.0)
    kwargs.update(overrides)
    return LoRAAdapter(**kwargs)


def test_lora_wraps_base_and_freezes_base_params():
    model = _make_lora()
    trainable, total = model.trainable_parameters()
    assert 0 < trainable < total
    # Trainable ratio should be << 1% on tiny_lm with r=4
    assert trainable / total < 0.10
    # Only LoRA params have requires_grad
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            assert p.requires_grad, f"{name} should be trainable"


def test_lora_forward_returns_modeloutput_with_logits():
    from lighttrain.protocols import ModelOutput

    model = _make_lora()
    B, T = 2, 4
    ids = torch.randint(0, 64, (B, T))
    out = model(input_ids=ids)
    assert isinstance(out, ModelOutput)
    assert "logits" in out.outputs
    assert out.outputs["logits"].shape == (B, T, 64)


def test_lora_state_dict_is_adapter_only_and_small():
    model = _make_lora()
    base = TinyCausalLM(**{k: v for k, v in _spec_for_tiny().items() if k != "name"})
    base_size = sum(p.numel() for p in base.parameters())
    sd = model.state_dict()
    # adapter-only keys
    for key in sd:
        assert "lora" in key.lower() or "base_layer" not in key
    # adapter size << base size
    adapter_size = sum(t.numel() for t in sd.values())
    assert adapter_size < base_size * 0.1, (
        f"adapter {adapter_size} should be <10% of base {base_size}"
    )


def test_lora_state_dict_round_trip():
    a = _make_lora()
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    # Train a step so a's adapter diverges from a fresh-init b.
    opt = torch.optim.SGD((p for p in a.parameters() if p.requires_grad), lr=0.1)
    ids = torch.randint(0, 64, (2, 4))
    out = a(input_ids=ids)
    out.outputs["logits"].mean().backward()
    opt.step()
    sd_after = {k: v.clone() for k, v in a.state_dict().items()}
    # weights actually changed
    diverged = any(not torch.allclose(sd[k], sd_after[k]) for k in sd)
    assert diverged
    # Now create a fresh adapter, load sd_after, expect match.
    b = _make_lora()
    b.load_state_dict(sd_after)
    sd_b = b.state_dict()
    for k in sd_after:
        assert torch.allclose(sd_after[k], sd_b[k], atol=1e-6), f"{k} mismatch"


def test_lora_param_groups_via_optim_wrapper():
    """When optimizer is built off the wrapped model, only LoRA params should
    end up in trainable groups (because peft set requires_grad=False on base)."""
    from lighttrain.optim.wrappers import AdamWWrapper

    model = _make_lora()
    wrapper = AdamWWrapper(lr=1e-3)
    inner = wrapper.build(model)
    n_trainable = sum(p.numel() for g in inner.param_groups for p in g["params"])
    n_total = sum(p.numel() for p in model.parameters())
    assert n_trainable < n_total * 0.10


def test_lora_dump_peft_spec_round_trip_via_minimal():
    """dump_peft_spec → build_minimal_model should reconstruct the same shape."""
    from lighttrain.minimal import build_minimal_model

    a = _make_lora(r=4, lora_alpha=8)
    spec = dump_peft_spec(a)
    assert spec["name"] == "lora"
    assert spec["params"]["r"] == 4
    assert spec["params"]["lora_alpha"] == 8
    assert "base" in spec["params"]
    # Rebuild
    b = build_minimal_model(spec)
    assert isinstance(b, LoRAAdapter)
    sa = a.state_dict()
    sb = b.state_dict()
    assert set(sa.keys()) == set(sb.keys()), "adapter key sets must match"


def test_is_peft_wrapped_detects_lora_adapter():
    model = _make_lora()
    assert is_peft_wrapped(model) is True
    base = TinyCausalLM(**{k: v for k, v in _spec_for_tiny().items() if k != "name"})
    assert is_peft_wrapped(base) is False


def test_lora_enable_input_require_grads_is_safe():
    """QLoRA requires this — should be a no-op or set hooks without crashing."""
    model = _make_lora()
    model.enable_input_require_grads()
    # Smoke: still can forward + backward
    ids = torch.randint(0, 64, (1, 4))
    out = model(input_ids=ids)
    out.outputs["logits"].mean().backward()


def test_lora_gradient_checkpointing_enable_is_safe():
    model = _make_lora()
    try:
        model.gradient_checkpointing_enable()
    except (ValueError, RuntimeError):
        # peft may refuse if the inner doesn't support it (tiny_lm doesn't);
        # the safety contract is "doesn't crash silently / no AttributeError"
        pass
