"""IA3Adapter — DESIGN §8.4 (M5).

Relocated from the flat ``tests/test_peft_ia3.py``. No mirror under
``tests/models/`` covered IA³, so behaviors are preserved (peft required).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")  # whole file requires peft

from lighttrain.builtin_plugins.models.peft import (  # noqa: E402
    IA3Adapter,
    dump_peft_spec,
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


def _make_ia3():
    return IA3Adapter(base=_spec_for_tiny())


def test_invariant_ia3_freezes_base_with_tiny_trainable_ratio():
    """Invariant: IA³ wrapping freezes the base and leaves only a very small
    trainable fraction (``trainable / total < 0.05``) — IA³ adds only per-
    channel rescaling vectors.
    """
    model = _make_ia3()
    trainable, total = model.trainable_parameters()
    assert 0 < trainable < total
    assert trainable / total < 0.05


def test_invariant_ia3_forward_returns_modeloutput_with_logits():
    """Invariant: the wrapped forward returns a ``ModelOutput`` carrying
    ``logits`` in its outputs mapping.
    """
    from lighttrain.protocols import ModelOutput

    model = _make_ia3()
    ids = torch.randint(0, 64, (1, 4))
    out = model(input_ids=ids)
    assert isinstance(out, ModelOutput)
    assert "logits" in out.outputs


def test_invariant_ia3_dump_spec_carries_target_and_feedforward_modules():
    """Invariant: ``dump_peft_spec`` for IA³ records ``name == 'ia3'`` plus the
    ``base``, ``target_modules`` and ``feedforward_modules`` needed to rebuild.
    """
    model = _make_ia3()
    spec = dump_peft_spec(model)
    assert spec["name"] == "ia3"
    assert "base" in spec["params"]
    assert "target_modules" in spec["params"]
    assert "feedforward_modules" in spec["params"]


def test_invariant_ia3_state_dict_round_trips_value_exact():
    """Invariant: an IA³ adapter's state_dict loads into a fresh IA³ adapter
    and every tensor matches within ``atol=1e-6``.
    """
    a = _make_ia3()
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = _make_ia3()
    b.load_state_dict(sd)
    sd_b = b.state_dict()
    for k in sd:
        assert torch.allclose(sd[k], sd_b[k], atol=1e-6), f"{k} mismatch"
