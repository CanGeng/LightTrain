"""Coverage supplement for ``lighttrain.models.extras``.

Targets the lines NOT reached by ``tests/models/test_extras.py``:

  - line  67 : ``_expand_braces`` → ``?`` glob metachar → ``out.append(".")``.
  - line 157 : ``apply_transform`` unknown-key fallthrough → bare ``return tensor``.
  - line 200 : hook receives a tuple output → ``t = t[0]`` unpack branch.
  - line 202 : hook receives a non-tensor output → early ``return`` guard.
  - line 227 : ``ExtrasHookManager.matched_modules`` property getter.
  - line 249 : ``extract_extra_outputs`` called WITH specs arg → second ``return out``.
  - lines 275-278 : ``flatten_model_output_tensors`` with non-None ``attentions``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.models.extras import (
    ExtraOutputSpec,
    ExtrasHookManager,
    compile_pattern,
    extract_extra_outputs,
    flatten_model_output_tensors,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _Identity(nn.Module):
    """Forward returns exactly its input tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TupleOutputModule(nn.Module):
    """Forward returns ``(tensor, extra_sentinel)`` — a tuple — to exercise
    the tuple-unpack branch (line 200).
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        return x * 2.0, "sentinel"


class _NonTensorOutputModule(nn.Module):
    """Forward returns a plain dict (not a tensor, not a tuple-of-tensor)
    to exercise the non-tensor guard (line 202).
    """

    def forward(self, x: torch.Tensor) -> dict:
        return {"status": "ok"}


def _wrap(submodule: nn.Module, name: str = "core") -> nn.Module:
    """Nest ``submodule`` under ``name`` so it can be addressed in a spec."""
    m = nn.Module()
    setattr(m, name, submodule)
    return m


# ---------------------------------------------------------------------------
# line 67 — _expand_braces "?" metachar (via compile_pattern glob)
# ---------------------------------------------------------------------------


def test_invariant_glob_question_mark_matches_exactly_one_char():
    """``?`` in a glob pattern matches any single character.

    ``compile_pattern`` delegates to ``_expand_braces`` which converts ``?``
    to the regex ``.`` (line 67). The resulting pattern therefore matches
    strings that differ in exactly that one position and rejects strings
    where that position is absent.

    Setup: pattern ``"blk.?.attn"`` (glob).
    Expected: matches ``"blk.3.attn"`` and ``"blk.x.attn"``; does NOT match
    ``"blk.12.attn"`` (two chars) or ``"blk..attn"`` (zero chars).
    """
    pat = compile_pattern("blk.?.attn", "glob")
    assert pat.match("blk.3.attn"), "single digit should match"
    assert pat.match("blk.x.attn"), "single letter should match"
    assert not pat.match("blk.12.attn"), "two-char segment must NOT match"
    assert not pat.match("blk..attn"), "empty segment must NOT match"


def test_invariant_glob_question_mark_auto_kind_also_expands():
    """``kind="auto"`` detects ``?`` as a glob metachar and expands it.

    Same semantics as explicit ``"glob"``; validated independently so we pin
    the auto-detect path as well.
    """
    pat = compile_pattern("m?.head", "auto")
    assert pat.match("m0.head")
    assert not pat.match("m10.head")


# ---------------------------------------------------------------------------
# line 157 — apply_transform unknown-key fallthrough
# ---------------------------------------------------------------------------


def test_pin_current_behavior_unknown_transform_key_returns_tensor_unchanged():
    """Pin: when the transform dict contains an unrecognised key (not topk /
    slice / layer / mean_dim), ``apply_transform`` falls through to the bare
    ``return tensor`` on line 157.

    This is arguably a silent-ignore bug (the user made a typo and nothing
    warns them), but it is the current behaviour, so we pin it here.

    Setup: ``transform={"typo_key": 5}`` on a (2, 3) tensor.
    Expected: returned value is the same tensor object.
    """
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    spec = ExtraOutputSpec(name="x", source="core", transform={"typo_key": 5})
    model = _wrap(_Identity())
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        model.core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    torch.testing.assert_close(captured["x"], src, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# line 200 — tuple output unpack in hook
# ---------------------------------------------------------------------------


def test_invariant_hook_unpacks_tuple_output_and_captures_first_element():
    """When a module's forward returns ``(tensor, ...)`` the hook unpacks the
    tuple and captures ``t[0]`` (line 200).

    Setup: ``_TupleOutputModule`` whose forward returns ``(x * 2, "sentinel")``;
    attach a no-transform spec for its output.
    Expected: cached tensor equals ``src * 2``.
    """
    torch.manual_seed(0)
    src = torch.tensor([[1.0, 2.0, 4.0]])
    model = _wrap(_TupleOutputModule())
    spec = ExtraOutputSpec(name="t", source="core")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        model.core(src)
        captured = mgr.collect()
    finally:
        mgr.detach()
    expected = src * 2.0
    torch.testing.assert_close(captured["t"], expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# line 202 — non-tensor guard in hook
# ---------------------------------------------------------------------------


def test_invariant_hook_skips_non_tensor_output_leaving_cache_empty():
    """When the captured value is not a ``torch.Tensor`` (and not a tuple
    whose first element is one), the hook returns early (line 202) without
    writing to the cache.

    Setup: ``_NonTensorOutputModule`` whose forward returns a plain dict;
    spec captures its output.
    Expected: cache stays empty — no error raised.
    """
    model = _wrap(_NonTensorOutputModule())
    spec = ExtraOutputSpec(name="nt", source="core")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        model.core(torch.zeros(1, 3))
        captured = mgr.collect()
    finally:
        mgr.detach()
    assert captured == {}, f"expected empty cache; got {captured}"


def test_invariant_hook_skips_non_tensor_input_side_leaving_cache_empty():
    """Same non-tensor guard but on the ``.input`` side.

    When the first positional input to a module is not a Tensor (e.g. a
    plain int), the hook also returns early without writing to the cache.
    """

    class _IntInputModule(nn.Module):
        def forward(self, x: int) -> torch.Tensor:  # type: ignore[override]
            return torch.tensor([float(x)])

    model = _wrap(_IntInputModule())
    spec = ExtraOutputSpec(name="ni", source="core.input")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        model.core(42)
        captured = mgr.collect()
    finally:
        mgr.detach()
    assert captured == {}


# ---------------------------------------------------------------------------
# line 227 — matched_modules property
# ---------------------------------------------------------------------------


def test_invariant_matched_modules_reflects_attached_specs():
    """``matched_modules`` property returns a dict mapping module name → list
    of specs whose pattern matched it (line 227).

    Setup: nested model with a ``core`` sub-module; spec matches ``"core"``.
    Expected: after ``attach()``, ``matched_modules["core"]`` contains the
    spec; the returned dict is a copy (mutating it does not affect internal state).
    """
    model = _wrap(_Identity())
    spec = ExtraOutputSpec(name="q", source="core")
    mgr = ExtrasHookManager(model, [spec]).attach()
    try:
        mm = mgr.matched_modules
        assert "core" in mm, f"'core' not in matched_modules: {mm}"
        assert any(s.name == "q" for s in mm["core"])
        # Mutating the copy must NOT affect internal state.
        mm.clear()
        assert "core" in mgr.matched_modules
    finally:
        mgr.detach()


def test_invariant_matched_modules_empty_before_attach():
    """``matched_modules`` is empty before ``attach()`` is called."""
    model = _wrap(_Identity())
    spec = ExtraOutputSpec(name="q", source="core")
    mgr = ExtrasHookManager(model, [spec])
    assert mgr.matched_modules == {}


# ---------------------------------------------------------------------------
# line 249 — extract_extra_outputs called WITH specs arg
# ---------------------------------------------------------------------------


def test_pin_current_behavior_extract_extra_outputs_with_specs_arg_returns_same_dict():
    """Pin: when ``extract_extra_outputs`` is called WITH a non-None ``specs``
    argument, it falls through to line 249 (``return out``).

    The current implementation ignores the ``specs`` argument entirely and
    returns the same dict as the no-specs path.  This is likely a stub / work
    in progress — the function signature implies specs-based filtering but
    the body never uses it.  We pin the current (no-op) behavior.

    Setup: ModelOutput with two extras entries; pass a non-None specs list.
    Expected: returned dict is identical to calling without specs.
    """
    vals = torch.tensor([1.0, 2.0])
    idx = torch.tensor([10, 20])
    mo = ModelOutput(
        outputs={"logits": torch.zeros(1, 4)},
        extras={
            "top": {"values": vals, "indices": idx},
            "plain": torch.tensor([7.0]),
        },
    )
    spec = ExtraOutputSpec(name="plain", source="some_module")
    out_with = extract_extra_outputs(mo, specs=[spec])
    out_without = extract_extra_outputs(mo, specs=None)
    assert set(out_with.keys()) == set(out_without.keys())
    for k in out_without:
        torch.testing.assert_close(out_with[k], out_without[k], atol=1e-6, rtol=1e-6)


def test_invariant_extract_extra_outputs_with_empty_specs_list_returns_full_dict():
    """Passing an empty list as specs hits the ``specs is None`` check as
    False (an empty list is not None) so also exercises line 249.
    """
    mo = ModelOutput(
        outputs={},
        extras={"x": torch.tensor([3.0])},
    )
    out = extract_extra_outputs(mo, specs=[])
    assert "x" in out
    torch.testing.assert_close(out["x"], torch.tensor([3.0]), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# lines 275-278 — flatten_model_output_tensors with attentions
# ---------------------------------------------------------------------------


def test_invariant_flatten_stacks_attentions_along_dim_zero():
    """``flatten_model_output_tensors`` stacks ``model_output.attentions``
    into ``"attentions_layers"`` exactly like it does for ``hidden_states``
    (lines 275-278).

    Setup: ModelOutput with attentions = 3 tensors of shape (1, 2, 4).
    Expected: ``"attentions_layers"`` in result; shape (3, 1, 2, 4); values
    match ``torch.stack(list, dim=0)``.
    """
    a0 = torch.ones(1, 2, 4) * 0.5
    a1 = torch.ones(1, 2, 4) * 1.5
    a2 = torch.ones(1, 2, 4) * 2.5
    mo = ModelOutput(
        outputs={"logits": torch.zeros(1, 2, 4)},
        attentions=(a0, a1, a2),
    )
    flat = flatten_model_output_tensors(mo)
    assert "attentions_layers" in flat, "expected 'attentions_layers' key"
    assert flat["attentions_layers"].shape == (3, 1, 2, 4)
    expected = torch.stack([a0, a1, a2], dim=0)
    torch.testing.assert_close(flat["attentions_layers"], expected, atol=1e-6, rtol=1e-6)


def test_invariant_flatten_omits_attentions_key_when_none():
    """When ``attentions=None`` (default), ``"attentions_layers"`` must NOT
    appear in the flattened output.
    """
    mo = ModelOutput(outputs={"logits": torch.zeros(1, 4)}, attentions=None)
    flat = flatten_model_output_tensors(mo)
    assert "attentions_layers" not in flat


def test_invariant_flatten_attentions_detaches_and_moves_to_cpu():
    """The stacked attentions tensor must be detached (no grad_fn) and on CPU
    (mirrors the hidden_states path).

    Setup: attention tensor with ``requires_grad=True``.
    Expected: ``attentions_layers`` in output has ``grad_fn is None``.
    """
    a = (torch.ones(1, 2, 4, requires_grad=True) * 1.5,)
    mo = ModelOutput(outputs={}, attentions=a)
    flat = flatten_model_output_tensors(mo)
    result = flat["attentions_layers"]
    assert result.grad_fn is None
    assert result.device.type == "cpu"


def test_invariant_flatten_skips_non_tensor_items_in_attentions():
    """Non-tensor items in ``attentions`` are filtered out; only tensors are
    stacked (the list-comprehension on line 275 excludes non-tensors).

    Setup: attentions = (tensor, "garbage", tensor).
    Expected: ``"attentions_layers"`` has leading dim 2 (only the two tensors).
    """
    a0 = torch.ones(1, 2, 4)
    a1 = torch.ones(1, 2, 4) * 2.0
    mo = ModelOutput(outputs={}, attentions=(a0, "garbage", a1))  # type: ignore[arg-type]
    flat = flatten_model_output_tensors(mo)
    assert flat["attentions_layers"].shape[0] == 2


def test_invariant_flatten_omits_attentions_key_when_all_non_tensor():
    """If every element in ``attentions`` is a non-tensor, ``ats`` is empty
    so ``if ats:`` is False and ``"attentions_layers"`` is not emitted.
    """
    mo = ModelOutput(outputs={}, attentions=("a", "b"))  # type: ignore[arg-type]
    flat = flatten_model_output_tensors(mo)
    assert "attentions_layers" not in flat


def test_invariant_flatten_both_hidden_states_and_attentions_present():
    """Both ``hidden_states`` and ``attentions`` can be stacked in the same
    call — they populate independent keys and do not interfere.
    """
    h = (torch.zeros(1, 3, 4), torch.ones(1, 3, 4))
    a = (torch.ones(1, 3, 4) * 0.1,)
    mo = ModelOutput(outputs={}, hidden_states=h, attentions=a)
    flat = flatten_model_output_tensors(mo)
    assert "hidden_states_layers" in flat
    assert "attentions_layers" in flat
    assert flat["hidden_states_layers"].shape == (2, 1, 3, 4)
    assert flat["attentions_layers"].shape == (1, 1, 3, 4)
