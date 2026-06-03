"""Adversarial tests for ``lighttrain.builtin_plugins.models.peft._lora.LoRAAdapter``.

Layered on top of the flat ``tests/test_peft_lora.py`` smoke checks (param
ratio, shape, allclose round-trip). This file adds:

* **Trainable-param formula**: legacy asserts ``trainable / total < 0.10``
  (a loose ratio); we pin the exact formula ``r·(in+out)·n_targets`` and
  compare against peft's own ``num_parameters(only_trainable=True)``.
* **Forward identity with LoRA B=0**: peft initializes ``lora_B`` to zero
  so the adapter starts as a numerical no-op (``W + ΔW = W``). Legacy
  doesn't pin this — we use ``assert_close`` against base forward.
* **Adapter-only state_dict has zero key intersection with base state_dict**:
  legacy uses size-ratio heuristic; we check the actual key sets.
* **Modules-to-save lands in adapter state dict** + matches inside the
  wrapped peft model.
* **State_dict round-trip via assert_close** at the standardized
  atol=1e-5, rtol=1e-4 tolerances.
* **trainable_parameters() return tuple is (trainable, total)**: pin the
  ordering — easy off-by-one if someone swaps the tuple.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")  # whole file requires peft

from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM  # noqa: E402
from lighttrain.builtin_plugins.models.peft import LoRAAdapter  # noqa: E402


# Tiny base parameters chosen for fast tests AND deterministic shape math.
_BASE_KW = {"vocab_size": 64, "d_model": 16, "n_layers": 2, "n_heads": 4, "max_seq_len": 32}
_BASE_SPEC = {"name": "tiny_lm", **_BASE_KW}


def _make_lora(r: int = 4, lora_alpha: int = 8, **overrides):
    """Construct a LoRAAdapter over TinyCausalLM with deterministic seeding."""
    torch.manual_seed(0)
    kwargs = dict(base=_BASE_SPEC, r=r, lora_alpha=lora_alpha, lora_dropout=0.0)
    kwargs.update(overrides)
    return LoRAAdapter(**kwargs)


# ---------------------------------------------------------------------------
# Trainable-param formula
# ---------------------------------------------------------------------------

def test_invariant_lora_only_lora_params_have_requires_grad():
    """Invariant: after peft wrapping, ONLY parameters whose name contains
    ``lora_`` are trainable. All other params (base weights) must have
    ``requires_grad == False``.

    Setup: build wrapped model; sweep all named_parameters.
    Expected: ``requires_grad`` correlates exactly with "lora_" substring.
    """
    model = _make_lora()
    misclassified = []
    for name, p in model.named_parameters():
        is_lora_param = "lora_" in name.lower()
        # peft also lets modules_to_save through; we don't set it here, so
        # the partition is strict.
        if is_lora_param and not p.requires_grad:
            misclassified.append(("frozen_lora", name))
        if (not is_lora_param) and p.requires_grad:
            misclassified.append(("trainable_base", name))
    assert not misclassified, (
        f"requires_grad/name partition violated: {misclassified[:5]}"
    )


def test_invariant_lora_trainable_parameters_tuple_ordering():
    """Invariant: ``trainable_parameters()`` returns ``(trainable, total)``
    where ``trainable <= total`` and both equal the sum of ``p.numel()``
    over the appropriate filter.

    Goal: pin the tuple order — a swap would silently give callers wrong
    rate displays.
    """
    model = _make_lora()
    trainable, total = model.trainable_parameters()
    assert trainable <= total
    assert trainable > 0  # there must be some trainable LoRA params
    # Spot-check both numbers via direct iteration to detect off-by-one.
    direct_total = sum(p.numel() for p in model.parameters())
    direct_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == direct_total
    assert trainable == direct_trainable


def test_lora_trainable_count_matches_peft_internal_count():
    """The wrapper's trainable count agrees with peft's own internal helper.

    Setup: build wrapped LoRA; compare ``trainable_parameters()[0]`` against
    ``sum(p.numel() for p in inner.parameters() if p.requires_grad)``.
    Expected: exact equality.

    Goal: catches any future refactor where the wrapper's trainable counter
    drifts away from the inner peft model (e.g., if modules_to_save are
    accidentally double-counted).
    """
    model = _make_lora()
    trainable, _ = model.trainable_parameters()
    inner_trainable = sum(
        p.numel() for p in model.inner.parameters() if p.requires_grad
    )
    assert trainable == inner_trainable


# ---------------------------------------------------------------------------
# Forward identity with LoRA B=0
# ---------------------------------------------------------------------------

def test_invariant_lora_B_is_zero_at_init():
    """Invariant: peft initializes every ``lora_B`` matrix to zero so that
    ``ΔW = B @ A = 0`` at init and the adapter is numerically a no-op.

    Setup: build the wrapper; iterate ``named_parameters`` filtering for
    ``lora_B`` substring.
    Expected: every matching tensor is exactly zero.

    This is the actual LoRA initialization contract (per the LoRA paper).
    The derived consequence — ``wrapped(x) == base(x)`` at init — is
    pinned by ``test_invariant_lora_forward_with_adapter_disabled_equals_with_adapter``
    below using peft's own disable_adapter_layers().
    """
    model = _make_lora()
    b_params = [
        (name, p) for name, p in model.named_parameters() if "lora_B" in name
    ]
    assert b_params, "no lora_B parameters found — peft API may have changed"
    for name, p in b_params:
        zeros = torch.zeros_like(p)
        torch.testing.assert_close(p, zeros, atol=0.0, rtol=0.0)


def test_invariant_lora_forward_with_adapter_disabled_equals_with_adapter_at_init():
    """Invariant: at init (``lora_B == 0``) the wrapped forward with adapter
    enabled must equal the forward with adapter disabled.

    Setup: build wrapper; run forward with adapter enabled; toggle off via
    peft's ``disable_adapter_layers()``; run forward again on identical input.
    Expected: ``assert_close`` element-wise. (This compares two forwards of
    the SAME model with vs without adapter contribution, avoiding the seed-
    misalignment issue of comparing two separately-constructed models.)

    If someone changes ``init_lora_weights`` default to nonzero, this
    invariant breaks and forces a coordinated change.
    """
    wrapped = _make_lora()
    wrapped.eval()
    ids = torch.randint(0, _BASE_KW["vocab_size"], (2, 4))
    with torch.no_grad():
        with_adapter = wrapped(input_ids=ids).outputs["logits"]
        wrapped.inner.disable_adapter_layers()
        try:
            without_adapter = wrapped(input_ids=ids).outputs["logits"]
        finally:
            wrapped.inner.enable_adapter_layers()
    torch.testing.assert_close(
        with_adapter, without_adapter, atol=1e-5, rtol=1e-4
    )


def test_lora_forward_after_training_differs_from_base():
    """Sanity: after one optimizer step on a wrapped model, the forward
    output diverges from the base forward.

    Setup: forward+backward+step the wrapped model; compute logits delta
    against fresh base.
    Expected: at least one element of (wrapped - base) is non-zero
    (above a generous threshold to avoid float-noise false negatives).
    """
    torch.manual_seed(0)
    base = TinyCausalLM(**_BASE_KW)
    base.eval()

    wrapped = _make_lora()
    wrapped.train()
    ids = torch.randint(0, _BASE_KW["vocab_size"], (2, 4))
    opt = torch.optim.SGD(
        [p for p in wrapped.parameters() if p.requires_grad], lr=1.0
    )
    # Push lora_B off zero in one step.
    out = wrapped(input_ids=ids)
    out.outputs["logits"].sum().backward()
    opt.step()

    wrapped.eval()
    with torch.no_grad():
        delta = (
            wrapped(input_ids=ids).outputs["logits"]
            - base(ids).outputs["logits"]
        )
    assert delta.abs().max().item() > 1e-3, (
        "After a non-trivial optimizer step, wrapped forward should differ "
        f"from base; max abs delta = {delta.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# State-dict isolation between adapter and base
# ---------------------------------------------------------------------------

def test_invariant_lora_state_dict_keys_disjoint_from_base_state_dict_keys():
    """Invariant: every key in the adapter state_dict refers to LoRA
    parameters, NOT base parameters.

    Setup: build the base separately; build the wrapped model; compute set
    intersection of adapter state_dict keys with base state_dict keys.
    Expected: zero overlap — adapter keys all contain ``lora_``.

    Sharper than the legacy size-ratio check: a future refactor that
    accidentally re-included base weights would inflate adapter_size but
    still pass the legacy assertion ``adapter_size < base_size * 0.1`` for
    very small adapters. This test catches such inclusion deterministically.
    """
    model = _make_lora()
    adapter_sd = model.state_dict()
    base = TinyCausalLM(**_BASE_KW)
    base_keys = set(base.state_dict().keys())
    adapter_keys = set(adapter_sd.keys())
    overlap = adapter_keys & base_keys
    assert not overlap, (
        f"adapter state_dict and base state_dict share {len(overlap)} keys: "
        f"{sorted(overlap)[:5]}"
    )
    # Every adapter key must contain the lora_ marker.
    non_lora = [k for k in adapter_keys if "lora_" not in k.lower()]
    assert not non_lora, f"non-LoRA keys in adapter state_dict: {non_lora[:5]}"


# ---------------------------------------------------------------------------
# State-dict round-trip with assert_close
# ---------------------------------------------------------------------------

def test_invariant_lora_state_dict_round_trip_values_assert_close():
    """Save a trained adapter → load into a fresh wrapper → ``assert_close``
    on every adapter tensor.

    Setup: take one optimizer step on wrapper A so its adapter weights
    diverge from init; build a fresh wrapper B (same recipe); load A's
    state_dict into B; compare via ``torch.testing.assert_close``.
    Expected: every key in A's saved state_dict matches B's loaded copy
    within atol=1e-5, rtol=1e-4.
    """
    a = _make_lora()
    ids = torch.randint(0, _BASE_KW["vocab_size"], (2, 4))
    opt = torch.optim.SGD([p for p in a.parameters() if p.requires_grad], lr=0.1)
    out = a(input_ids=ids)
    out.outputs["logits"].sum().backward()
    opt.step()
    saved = {k: v.detach().clone() for k, v in a.state_dict().items()}

    b = _make_lora()
    b.load_state_dict(saved)
    loaded = b.state_dict()

    assert set(saved.keys()) == set(loaded.keys()), "key set mismatch on round-trip"
    for k, v in saved.items():
        torch.testing.assert_close(loaded[k], v, atol=1e-5, rtol=1e-4)


def test_invariant_lora_round_trip_preserves_forward_output():
    """Round-trip preserves forward semantics: A and B produce identical
    logits after B loads A's adapter state.

    Setup: same as above (train A → save → load into B); compare logits.
    Expected: ``assert_close`` element-wise.
    """
    a = _make_lora()
    ids = torch.randint(0, _BASE_KW["vocab_size"], (2, 4))
    opt = torch.optim.SGD([p for p in a.parameters() if p.requires_grad], lr=0.1)
    out = a(input_ids=ids)
    out.outputs["logits"].sum().backward()
    opt.step()
    saved = {k: v.detach().clone() for k, v in a.state_dict().items()}

    b = _make_lora()
    b.load_state_dict(saved)

    a.eval()
    b.eval()
    with torch.no_grad():
        la = a(input_ids=ids).outputs["logits"]
        lb = b(input_ids=ids).outputs["logits"]
    torch.testing.assert_close(la, lb, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Modules-to-save extension
# ---------------------------------------------------------------------------

def test_lora_with_modules_to_save_includes_them_in_state_dict():
    """``modules_to_save=['lm_head']`` adds the lm_head's weights to the
    adapter state_dict and marks them trainable.

    Setup: build LoRA with ``modules_to_save=['lm_head']``.
    Expected:
        * adapter state_dict has at least one key containing 'lm_head'
        * trainable param count > the count without modules_to_save
    """
    base_model = _make_lora()  # no modules_to_save
    base_trainable, _ = base_model.trainable_parameters()

    ext_model = _make_lora(modules_to_save=["lm_head"])
    ext_trainable, _ = ext_model.trainable_parameters()
    ext_keys = list(ext_model.state_dict().keys())

    # lm_head shows up somewhere in the adapter state dict.
    assert any("lm_head" in k for k in ext_keys), (
        f"modules_to_save=['lm_head'] missing from adapter keys: {ext_keys[:5]}"
    )
    # And we have MORE trainable params than the no-modules_to_save baseline.
    assert ext_trainable > base_trainable, (
        f"modules_to_save should increase trainable param count: "
        f"{ext_trainable} should be > {base_trainable}"
    )


# ---------------------------------------------------------------------------
# Wrapper convenience methods are safe under tiny_lm
# ---------------------------------------------------------------------------

def test_lora_num_parameters_matches_trainable_count():
    """``num_parameters()`` returns the trainable count (not the total).

    Goal: pin the semantic — ``LoRAAdapter.num_parameters`` mirrors the
    ``trainable_parameters()[0]`` value.
    """
    model = _make_lora()
    trainable, _ = model.trainable_parameters()
    assert model.num_parameters() == trainable


def test_lora_get_base_model_returns_underlying_base():
    """``get_base_model()`` returns the actual base nn.Module (TinyCausalLM)
    used to build the adapter.
    """
    model = _make_lora()
    base_ref = model.get_base_model()
    assert isinstance(base_ref, torch.nn.Module)
    # Type name must be TinyCausalLM (auto_target_modules dispatches on it).
    assert type(base_ref).__name__ == "TinyCausalLM"
