"""ExtraOutputSpec + ExtrasHookManager — DESIGN §8.2."""

from __future__ import annotations

import torch

from lighttrain.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.models.extras import (
    ExtraOutputSpec,
    ExtrasHookManager,
    compile_pattern,
    flatten_model_output_tensors,
)


def test_compile_pattern_literal_and_glob_and_regex():
    assert compile_pattern("model.lm_head", "literal").match("model.lm_head")
    assert not compile_pattern("model.lm_head", "literal").match("model.lm_head.weight")

    pat = compile_pattern("blocks.{0,2,4}", "glob")
    for hit in ("blocks.0", "blocks.2", "blocks.4"):
        assert pat.match(hit), hit
    assert not pat.match("blocks.1")

    rx = compile_pattern(r"blocks\.\d+\.attn", "regex")
    assert rx.match("blocks.7.attn")


def test_hook_manager_captures_lm_head_output_with_topk_transform():
    model = TinyCausalLM(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    spec = ExtraOutputSpec(
        name="logits_topk_8", source="lm_head", transform={"topk": 8}
    )
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        ids = torch.randint(0, 64, (2, 8))
        out = model(ids)
        captured = mgr.collect()
    finally:
        mgr.detach()
    assert "logits_topk_8" in captured
    payload = captured["logits_topk_8"]
    assert set(payload) == {"values", "indices"}
    assert payload["values"].shape == (2, 8, 8)
    assert payload["indices"].shape == (2, 8, 8)
    # original ModelOutput stays intact
    assert out.outputs["logits"].shape == (2, 8, 64)


def test_flatten_includes_hidden_states_when_requested():
    model = TinyCausalLM(
        vocab_size=64, d_model=32, n_layers=3, n_heads=4, max_seq_len=8,
        output_hidden_states=True,
    )
    ids = torch.randint(0, 64, (1, 4))
    out = model(ids)
    flat = flatten_model_output_tensors(out)
    assert "logits" in flat
    assert "hidden_states_layers" in flat
    # 3 blocks + 1 embedding output => 4 layers in hidden_states tuple
    assert flat["hidden_states_layers"].shape[0] == 4


def test_hook_manager_idempotent_attach_and_detach():
    model = TinyCausalLM(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    spec = ExtraOutputSpec(name="x", source="lm_head")
    mgr = ExtrasHookManager(model, [spec])
    assert not mgr._handles
    mgr.attach()
    assert mgr._handles
    mgr.attach()  # idempotent
    mgr.detach()
    assert not mgr._handles
