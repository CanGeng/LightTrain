"""Adversarial tests for ``lighttrain.models.extras``.

Layered on top of the flat ``tests/test_extras.py`` smoke tests (shape +
idempotency). This file adds:

* **TopK transform closed-form values**: legacy asserts shape only; we
  construct deterministic logits and assert values match ``torch.topk``
  exactly via ``assert_close``.
* **Slice/mean_dim/layer transforms by closed-form**.
* **Detach pin**: captured tensors have ``requires_grad=False``
  (legacy doesn't test this).
* **Pattern matcher edge cases**: ``.input`` vs ``.output`` suffix,
  empty-brace group, escape semantics.
* **``flatten_model_output_tensors``** correctly stacks hidden_states
  (``L, B, T, H``) and skips None.
* **Reset** clears the cache without detaching handles.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.models.extras import (
    ExtraOutputSpec,
    ExtrasHookManager,
    compile_pattern,
    extract_extra_outputs,
    flatten_model_output_tensors,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def test_compile_pattern_literal_does_not_match_prefix():
    """Literal mode anchors both ends.

    Setup: literal pattern ``"a.b"``.
    Expected: matches ``"a.b"`` exactly; does NOT match ``"a.b.c"`` or
    ``"xa.b"``.
    """
    pat = compile_pattern("a.b", "literal")
    assert pat.match("a.b")
    assert not pat.match("a.b.c")
    assert not pat.match("xa.b")


def test_compile_pattern_glob_with_brace_group_matches_listed_members():
    """Brace expansion produces an alternation.

    Setup: pattern ``"blocks.{0,2,4}"``.
    Expected: matches the three listed indices; does NOT match 1 or 3 or 5.
    """
    pat = compile_pattern("blocks.{0,2,4}", "glob")
    for hit in ("blocks.0", "blocks.2", "blocks.4"):
        assert pat.match(hit), f"expected match for {hit}"
    for miss in ("blocks.1", "blocks.3", "blocks.5"):
        assert not pat.match(miss), f"expected NO match for {miss}"


def test_compile_pattern_glob_with_empty_brace_group_collapses():
    """An empty brace group ``{}`` collapses to nothing — the rest of the
    string is matched literally.

    Setup: pattern ``"a{}.b"``.
    Expected: matches literal ``"a.b"`` only.
    """
    pat = compile_pattern("a{}.b", "glob")
    assert pat.match("a.b")
    assert not pat.match("a.x.b")


def test_compile_pattern_glob_star_matches_arbitrary_chars():
    """``*`` becomes ``.*`` (any chars).

    Setup: pattern ``"a.*"``.
    Expected: matches ``"a.foo"``, ``"a.bar.baz"``, but NOT ``"x.foo"``.
    """
    pat = compile_pattern("a.*", "glob")
    assert pat.match("a.foo")
    assert pat.match("a.bar.baz")
    assert not pat.match("x.foo")


def test_compile_pattern_regex_uses_string_verbatim():
    """``kind="regex"`` is verbatim — special chars are not escaped.

    Setup: pattern ``r"blocks\\.\\d+\\.attn"``.
    Expected: matches ``"blocks.7.attn"`` (regex digit class).
    """
    pat = compile_pattern(r"blocks\.\d+\.attn", "regex")
    assert pat.match("blocks.7.attn")
    assert not pat.match("blocks.x.attn")


def test_spec_input_vs_output_suffix_dispatches_to_correct_side():
    """``.input`` and ``.output`` suffixes select which hook side is captured.

    Setup: two ExtraOutputSpec — one for ``"lm_head.input"``, one for
    ``"lm_head.output"``.
    Expected: ``side()`` reports the corresponding string; both match the
    stripped module name "lm_head".
    """
    a = ExtraOutputSpec(name="in", source="lm_head.input")
    b = ExtraOutputSpec(name="out", source="lm_head.output")
    assert a.side() == "input"
    assert b.side() == "output"
    assert a.matches("lm_head")
    assert b.matches("lm_head")


def test_spec_default_side_is_output():
    """No suffix → defaults to ``output`` side."""
    s = ExtraOutputSpec(name="x", source="lm_head")
    assert s.side() == "output"


# ---------------------------------------------------------------------------
# Transform closed-form correctness
# ---------------------------------------------------------------------------

class _Identity(nn.Module):
    """Minimal module whose forward output is exactly the input tensor.

    Used so hook tests can pin the captured tensor to a value we constructed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _model_with_identity(tensor: torch.Tensor) -> nn.Module:
    """Wrap ``_Identity`` so it can be addressed as ``"core"`` in the spec."""
    m = nn.Module()
    m.core = _Identity()
    return m


def test_invariant_topk_transform_returns_values_and_indices_matching_torch_topk():
    """Invariant: ``transform={topk: K}`` returns ``{"values", "indices"}``
    exactly matching ``torch.topk(tensor, k, dim=-1)``.

    Closed-form input: deterministic 2D tensor.
    Expected: ``assert_close`` on values; indices match exactly.
    """
    # 1×5 row with known descending order after topk(2) → [5, 4] at idx [1, 3]
    src = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
    expected_vals, expected_idx = torch.topk(src, k=2, dim=-1)

    spec = ExtraOutputSpec(name="t", source="core", transform={"topk": 2})
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        # drive the hook by running the Identity submodule directly
        cast(Any, model).core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()

    payload = captured["t"]
    torch.testing.assert_close(payload["values"], expected_vals, atol=1e-5, rtol=1e-4)
    # indices are cast to int32 by the transform — compare value-equal.
    assert torch.equal(payload["indices"].to(torch.int64), expected_idx)


def test_invariant_slice_transform_returns_subview_exact():
    """Invariant: ``transform={slice: [i, j]}`` returns ``tensor[..., i:j]``.

    Setup: shape (2, 5) source, slice [1, 4].
    Expected: shape (2, 3) AND values match the slice exactly.
    """
    src = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    expected = src[..., 1:4]

    spec = ExtraOutputSpec(name="s", source="core", transform={"slice": [1, 4]})
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    torch.testing.assert_close(captured["s"], expected, atol=1e-5, rtol=1e-4)


def test_invariant_mean_dim_transform_returns_mean_along_dim():
    """Invariant: ``transform={mean_dim: D}`` returns ``tensor.mean(dim=D)``.

    Setup: shape (3, 4) source, dim=0.
    Expected: shape (4,) AND values match ``tensor.mean(dim=0)`` exactly.
    """
    src = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = src.mean(dim=0)

    spec = ExtraOutputSpec(name="m", source="core", transform={"mean_dim": 0})
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    torch.testing.assert_close(captured["m"], expected, atol=1e-5, rtol=1e-4)


def test_invariant_layer_transform_picks_index_along_dim_zero():
    """Invariant: ``transform={layer: i}`` returns ``tensor[i]``
    (zero-dim indexing into a stacked tuple-like tensor).

    Setup: shape (4, 3) source representing 4 layers × 3 features.
    Expected: ``tensor[2]`` returned for ``layer=2``.
    """
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected = src[2]

    spec = ExtraOutputSpec(name="l", source="core", transform={"layer": 2})
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    torch.testing.assert_close(captured["l"], expected, atol=1e-5, rtol=1e-4)


def test_invariant_no_transform_passes_tensor_through_unchanged():
    """``transform=None`` (default) returns the captured tensor verbatim
    (after detach/CPU).
    """
    src = torch.tensor([[1.0, 2.0, 3.0]])
    spec = ExtraOutputSpec(name="passthrough", source="core")
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    torch.testing.assert_close(captured["passthrough"], src, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Detach / cpu pin
# ---------------------------------------------------------------------------

def test_invariant_captured_tensor_is_detached_by_default():
    """Invariant: ``detach=True`` is the default, so captured tensors have
    ``requires_grad=False`` and ``grad_fn is None``.

    Setup: build Identity model; route a tensor with requires_grad=True
    through it; capture via hook.
    Expected: captured tensor has no gradient graph.
    """
    src = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    # routed through a multiplication so the captured tensor's grad_fn would
    # otherwise be MulBackward
    routed = src * 2.0

    spec = ExtraOutputSpec(name="d", source="core")
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(routed)
        captured = mgr.collect()
    finally:
        mgr.detach()

    t = captured["d"]
    assert t.requires_grad is False
    assert t.grad_fn is None


def test_pin_detach_false_preserves_grad_fn():
    """Pin: ``detach=False`` preserves the gradient graph on captured tensors.

    Setup: ``detach=False, cpu=False`` (to avoid the implicit .cpu() also
    severing the graph).
    Expected: captured tensor has a non-None ``grad_fn`` AND
    ``requires_grad=True`` (inherited from the upstream chain).
    """
    src = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    routed = src * 3.0

    spec = ExtraOutputSpec(name="nd", source="core", detach=False, cpu=False)
    model = _model_with_identity(src)
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        cast(Any, model).core(routed)
        captured = mgr.collect()
    finally:
        mgr.detach()

    t = captured["nd"]
    assert t.requires_grad is True
    assert t.grad_fn is not None


# ---------------------------------------------------------------------------
# Manager lifecycle pins
# ---------------------------------------------------------------------------

def test_attach_is_idempotent_no_duplicate_handles():
    """Invariant: ``attach()`` called twice does not register duplicate hooks.

    Setup: attach, attach again.
    Expected: handle list length unchanged on the second call.
    """
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    spec = ExtraOutputSpec(name="x", source="lm_head")
    mgr = ExtrasHookManager(model, [spec])
    mgr.attach()
    n = len(mgr._handles)
    mgr.attach()  # idempotent
    assert len(mgr._handles) == n
    mgr.detach()


def test_reset_clears_cache_but_keeps_handles():
    """``reset()`` empties the per-spec cache without detaching hooks.

    Setup: attach + forward + verify cached; call reset; cache is empty;
    forward again; cache repopulates.
    """
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    spec = ExtraOutputSpec(name="x", source="lm_head")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        ids = torch.randint(0, 32, (1, 4))
        model(ids)
        assert mgr.collect()  # populated
        mgr.reset()
        assert mgr.collect() == {}  # cleared
        model(ids)
        assert mgr.collect()  # repopulated (handles intact)
    finally:
        mgr.detach()


def test_detach_removes_all_handles_and_clears_cache():
    """Invariant: ``detach()`` releases handles AND clears the cache.

    Setup: attach + forward; detach.
    Expected: ``_handles`` and ``_cache`` are empty after detach.
    """
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    spec = ExtraOutputSpec(name="x", source="lm_head")
    mgr = ExtrasHookManager(model, [spec]).attach()
    ids = torch.randint(0, 32, (1, 4))
    model(ids)
    mgr.detach()
    assert mgr._handles == []
    assert mgr._cache == {}


def test_invariant_topk_capture_through_real_model_leaves_modeloutput_logits_intact():
    """Invariant: capturing ``lm_head`` output via a topk transform on a real
    ``TinyCausalLM`` forward does NOT mutate the returned ModelOutput logits.

    Setup: attach a ``transform={topk: 8}`` spec to ``lm_head``; run a real
    forward.
    Expected: captured payload has ``{"values", "indices"}`` each of shape
    (B, T, K), AND the model's own ``out.outputs["logits"]`` retains full
    vocab width (B, T, vocab) — the hook is non-destructive.
    """
    model = TinyCausalLM(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    spec = ExtraOutputSpec(name="logits_topk_8", source="lm_head", transform={"topk": 8})
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        ids = torch.randint(0, 64, (2, 8))
        out = model(ids)
        captured = mgr.collect()
    finally:
        mgr.detach()
    payload = captured["logits_topk_8"]
    assert set(payload) == {"values", "indices"}
    assert payload["values"].shape == (2, 8, 8)
    assert payload["indices"].shape == (2, 8, 8)
    # The original ModelOutput is untouched — full vocab logits survive.
    assert out.outputs["logits"].shape == (2, 8, 64)


def test_invariant_flatten_hidden_states_from_real_model_has_n_layers_plus_one():
    """Invariant: a real ``TinyCausalLM`` built with ``output_hidden_states=True``
    yields a ``hidden_states`` tuple of ``n_layers + 1`` entries (one per block
    plus the embedding output), and ``flatten_model_output_tensors`` stacks them
    under ``"hidden_states_layers"``.

    Setup: 3-layer model; real forward; flatten.
    Expected: ``"logits"`` present; ``hidden_states_layers`` leading dim == 4.
    """
    model = TinyCausalLM(
        vocab_size=64, d_model=32, n_layers=3, n_heads=4, max_seq_len=8,
        output_hidden_states=True,
    )
    ids = torch.randint(0, 64, (1, 4))
    out = model(ids)
    flat = flatten_model_output_tensors(out)
    assert "logits" in flat
    assert "hidden_states_layers" in flat
    assert flat["hidden_states_layers"].shape[0] == 4


def test_unmatched_source_yields_empty_cache_no_error():
    """A spec whose ``source`` matches nothing is silently skipped.

    Setup: spec pointing at a module name that does not exist.
    Expected: attach succeeds; collect returns empty dict; no error.
    """
    model = TinyCausalLM(vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    spec = ExtraOutputSpec(name="x", source="no_such_module_xyz")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        ids = torch.randint(0, 32, (1, 4))
        model(ids)
        assert mgr.collect() == {}
    finally:
        mgr.detach()


# ---------------------------------------------------------------------------
# flatten_model_output_tensors
# ---------------------------------------------------------------------------

def test_invariant_flatten_stacks_hidden_states_along_dim_zero():
    """Invariant: when ModelOutput.hidden_states is a tuple of L tensors each
    of shape ``(B, T, H)``, ``flatten_model_output_tensors`` stacks them
    along ``dim=0`` to produce shape ``(L, B, T, H)``.

    Closed-form: build a ModelOutput with hand-crafted hidden_states (3
    tensors of shape (1, 2, 4)). Flatten. Expected key
    ``"hidden_states_layers"`` of shape (3, 1, 2, 4) AND values equal
    ``torch.stack(list, dim=0)``.
    """
    h0 = torch.ones(1, 2, 4) * 0.0
    h1 = torch.ones(1, 2, 4) * 1.0
    h2 = torch.ones(1, 2, 4) * 2.0
    mo = ModelOutput(
        outputs={"logits": torch.zeros(1, 2, 4)},
        hidden_states=(h0, h1, h2),
    )
    flat = flatten_model_output_tensors(mo)
    assert "hidden_states_layers" in flat
    assert flat["hidden_states_layers"].shape == (3, 1, 2, 4)
    expected = torch.stack([h0, h1, h2], dim=0)
    torch.testing.assert_close(
        flat["hidden_states_layers"], expected, atol=1e-5, rtol=1e-4
    )


def test_invariant_flatten_omits_hidden_states_key_when_none():
    """``hidden_states=None`` → no ``"hidden_states_layers"`` key in output."""
    mo = ModelOutput(outputs={"logits": torch.zeros(1, 2, 4)}, hidden_states=None)
    flat = flatten_model_output_tensors(mo)
    assert "hidden_states_layers" not in flat


def test_flatten_expands_extras_mapping_values_with_dotted_subkey():
    """When an extras value is itself a mapping (e.g. topk → values+indices),
    flatten emits ``<key>.<subkey>`` keys.

    Setup: extras = {"top": {"values": t1, "indices": t2}}.
    Expected: flat contains both ``"top.values"`` and ``"top.indices"``,
    each matching the source tensors.
    """
    vals = torch.tensor([1.0, 2.0])
    idx = torch.tensor([10, 20])
    mo = ModelOutput(
        outputs={"logits": torch.zeros(1, 4)},
        extras={"top": {"values": vals, "indices": idx}},  # type: ignore[dict-item]
    )
    flat = flatten_model_output_tensors(mo)
    assert "top.values" in flat and "top.indices" in flat
    torch.testing.assert_close(flat["top.values"], vals, atol=1e-5, rtol=1e-4)
    assert torch.equal(flat["top.indices"], idx)


def test_extract_extra_outputs_flattens_topk_mapping():
    """``extract_extra_outputs`` flattens topk-style mappings into dotted keys.

    Setup: ModelOutput with extras containing a mapping.
    Expected: returned dict has ``key.values`` AND ``key.indices``.
    """
    mo = ModelOutput(
        outputs={"logits": torch.zeros(1, 4)},
        extras={
            "scores": {"values": torch.tensor([5.0]), "indices": torch.tensor([2])},  # type: ignore[dict-item]
            "plain": torch.tensor([7.0]),
        },
    )
    out = extract_extra_outputs(mo)
    assert "scores.values" in out
    assert "scores.indices" in out
    assert "plain" in out
