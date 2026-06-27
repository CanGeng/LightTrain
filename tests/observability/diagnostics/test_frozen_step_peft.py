"""Frozen Step Bundle + PEFT — DESIGN §18.1 + §8.4 (M5).

Confirms that a LoRA-wrapped model lands a correctly-shaped ``model_spec``
inside ``step_metadata.json`` so ``build_minimal_model`` can reconstruct
the exact wrap on replay.
"""

from __future__ import annotations

import json
import zipfile

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from lighttrain.builtin_plugins.models.peft import LoRAAdapter  # noqa: E402
from lighttrain.observability.diagnostics.frozen_step import (  # noqa: E402
    FrozenStepWriter,
    read_frozen_step_bundle,
)
from lighttrain.observability.minimal import build_minimal_model  # noqa: E402
from tests._diagnostics import expect_exists  # noqa: E402


def _spec_for_tiny() -> dict:
    return {
        "name": "tiny_lm",
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 32,
    }


def _make_lora() -> LoRAAdapter:
    return LoRAAdapter(base=_spec_for_tiny(), r=4, lora_alpha=8, lora_dropout=0.0)


def _make_ctx(epoch=0):
    class _Ctx:
        pass

    c = _Ctx()
    c.epoch = epoch  # type: ignore[attr-defined]
    return c


def test_frozen_step_writes_lora_model_spec(tmp_path):
    model = _make_lora()
    writer = FrozenStepWriter(run_dir=tmp_path)
    writer.snapshot(
        step=42,
        ctx=_make_ctx(),
        model=model,
        batch={"input_ids": torch.randint(0, 64, (1, 4))},
        optimizer=None,
    )
    out = writer.commit(reason="scheduled")
    expect_exists(out, tmp_path, what="frozen-step zip")
    # Read back the metadata.
    assert out is not None
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("step_metadata.json"))
    spec = meta["model_spec"]
    assert spec["name"] == "lora"
    assert "base" in spec["params"]
    assert spec["params"]["r"] == 4
    assert spec["params"]["lora_alpha"] == 8


def test_frozen_step_bundle_reconstructs_lora_via_minimal(tmp_path):
    a = _make_lora()
    # Train one step so adapter weights diverge from a fresh init.
    opt = torch.optim.SGD((p for p in a.parameters() if p.requires_grad), lr=0.1)
    ids = torch.randint(0, 64, (2, 4))
    out = a(input_ids=ids)
    out.outputs["logits"].mean().backward()
    opt.step()

    writer = FrozenStepWriter(run_dir=tmp_path)
    writer.snapshot(step=7, ctx=_make_ctx(), model=a, batch={"input_ids": ids}, optimizer=opt)
    bundle_path = writer.commit(reason="cli")
    assert bundle_path is not None
    bundle = read_frozen_step_bundle(bundle_path)
    spec = bundle.metadata["model_spec"]
    # Rebuild via the minimal-model path. Same shape contract as M4.
    b = build_minimal_model(spec)
    assert isinstance(b, LoRAAdapter)
    sa = a.state_dict()
    sb = b.state_dict()
    assert set(sa.keys()) == set(sb.keys())
