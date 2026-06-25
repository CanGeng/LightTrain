"""Edge-case coverage for ``lighttrain.builtin_plugins.models.peft._adalora``.

The repo ships with PEFT installed, so ``AdaLoRAAdapter`` defaults to its
``_use_peft=True`` branch. This file drives the *manual* fallback path (and the
PEFT export/load failure fallbacks) that the existing
``tests/models/test_peft_adalora.py`` does not reach.

What we pin:

* The manual fallback is selected when ``from peft import ...`` raises
  ``ImportError`` (we force this via ``monkeypatch.setitem(sys.modules,
  "peft", None)``) — ``_use_peft`` becomes False and ``_build_manual`` runs.
* ``_build_manual`` replaces only target ``nn.Linear`` submodules: non-Linear
  modules and non-matching Linear modules are skipped; both top-level
  (``parent_name == ""``) and nested (``"block.0"``) children are rewired.
* ``maybe_reallocate_rank`` step/interval/empty-layer/PEFT guards, and the
  uniform prune-to-``target_r`` reallocation.
* ``forward`` normalises the three base-model return shapes (ModelOutput,
  ``.logits`` carrier, raw tensor) plus the empty-dict fallback.
* ``state_dict`` / ``load_state_dict`` manual collection and the PEFT-helper
  failure fallbacks (logged + degrade gracefully).
* ``AdaLoRALinear.adapter_state_dict`` keys/contents (line 105).

``AdaLoRALinear`` is always manual (no PEFT dependency); ``AdaLoRAAdapter``
is path-dependent, hence the import forcing.
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.peft._adalora import (
    AdaLoRAAdapter,
    AdaLoRALinear,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Stub base models (forwards accept **batch like real lighttrain models)
# ---------------------------------------------------------------------------

class _TopLevelLinears(nn.Module):
    """Direct children so ``rpartition('.')`` yields an empty parent name."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)      # targeted -> rewired (parent_name == "")
        self.other = nn.Linear(4, 4)    # not targeted -> skipped

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.lin(batch["x"])


class _NestedLinear(nn.Module):
    """Linear lives at ``block.0`` so a non-empty parent name is resolved."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4))

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.block(batch["x"])


class _ReturnsModelOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, **batch: torch.Tensor) -> ModelOutput:
        return ModelOutput(outputs={"logits": self.lin(batch["x"])})


class _LogitsCarrier:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _ReturnsLogitsAttr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, **batch: torch.Tensor) -> _LogitsCarrier:
        return _LogitsCarrier(self.lin(batch["x"]))


class _ReturnsTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.lin(batch["x"])


class _ReturnsWeird(nn.Module):
    """Returns a non-tensor, non-``.logits`` object -> empty-outputs branch."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, **batch: torch.Tensor) -> tuple[str, str]:
        return ("weird", "tuple")


@pytest.fixture
def no_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``from peft import ...`` to raise ImportError inside __init__.

    Setting a ``None`` entry in ``sys.modules`` makes Python raise
    ``ImportError`` on import — the canonical way to simulate a missing
    optional dependency without touching the installed package.
    """
    monkeypatch.setitem(sys.modules, "peft", None)


# ---------------------------------------------------------------------------
# AdaLoRALinear.adapter_state_dict (line 105)
# ---------------------------------------------------------------------------

def test_invariant_adapter_state_dict_has_three_named_tensors():
    """``adapter_state_dict`` returns exactly the A/B/Lambda data tensors."""
    base = nn.Linear(8, 16)
    ada = AdaLoRALinear(base, r=4, lora_alpha=8)
    sd = ada.adapter_state_dict()
    assert set(sd) == {"lora_A", "lora_B", "lora_Lambda"}
    assert sd["lora_A"].shape == (4, 8)
    assert sd["lora_B"].shape == (16, 4)
    assert sd["lora_Lambda"].shape == (4,)
    # The returned tensors are the live ``.data`` views, not copies.
    assert sd["lora_A"].data_ptr() == ada.lora_A.data.data_ptr()


# ---------------------------------------------------------------------------
# Manual path selection + _build_manual (lines 158-159, 167, 187-201)
# ---------------------------------------------------------------------------

def test_invariant_import_failure_selects_manual_path(no_peft):
    """When ``peft`` import fails, ``_use_peft`` is False and a manual layer map
    is built (covers the ``except ImportError: pass`` and ``_build_manual`` call).
    """
    adapter = AdaLoRAAdapter(
        base=_TopLevelLinears(), r=2, target_modules=["lin"], total_step=10
    )
    assert adapter._use_peft is False
    assert hasattr(adapter, "_adalora_layers")


def test_invariant_build_manual_rewires_only_targeted_top_level_linear(no_peft):
    """Only the targeted top-level ``lin`` becomes ``AdaLoRALinear``; the
    untargeted ``other`` Linear is left untouched (line 193-194 skip + the
    ``parent_name == ""`` branch of line 197).
    """
    adapter = AdaLoRAAdapter(
        base=_TopLevelLinears(), r=2, target_modules=["lin"], total_step=10
    )
    assert list(adapter._adalora_layers) == ["lin"]
    assert isinstance(adapter.model.lin, AdaLoRALinear)
    assert isinstance(adapter.model.other, nn.Linear)
    assert not isinstance(adapter.model.other, AdaLoRALinear)


def test_invariant_build_manual_resolves_nested_parent(no_peft):
    """A Linear at ``block.0`` is rewired via a non-empty parent name lookup
    (the second half of line 197).
    """
    adapter = AdaLoRAAdapter(
        base=_NestedLinear(), r=4, target_modules=["block"], total_step=10
    )
    assert list(adapter._adalora_layers) == ["block.0"]
    assert isinstance(adapter.model.block[0], AdaLoRALinear)


def test_invariant_build_manual_skips_non_linear_and_unmatched(no_peft):
    """Non-Linear modules (line 191-192) and Linear modules whose name does not
    contain any target (line 193-194) are skipped.

    The base also exposes a Linear (``other``) that is *not* targeted, so the
    only rewired entry is the explicitly named one.
    """
    adapter = AdaLoRAAdapter(
        base=_TopLevelLinears(), r=2, target_modules=["lin"], total_step=10
    )
    # ReLU/Sequential-style non-Linear modules never enter the layer map, and
    # neither does the untargeted ``other`` Linear.
    assert "other" not in adapter._adalora_layers
    assert all(isinstance(layer, AdaLoRALinear) for layer in adapter._adalora_layers.values())


# ---------------------------------------------------------------------------
# maybe_reallocate_rank (lines 205-215)
# ---------------------------------------------------------------------------

def test_invariant_reallocate_increments_step_but_no_prune_off_interval(no_peft):
    """``maybe_reallocate_rank`` always increments ``_step`` but only prunes on
    interval multiples (line 205 increment + line 208-209 guard).
    """
    adapter = AdaLoRAAdapter(
        base=_NestedLinear(), r=4, target_modules=["block"],
        target_r=2, update_interval=2, total_step=10,
    )
    layer = adapter._adalora_layers["block.0"]
    with torch.no_grad():
        layer.lora_Lambda.copy_(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    adapter.maybe_reallocate_rank()  # step 1 -> 1 % 2 != 0, no prune
    assert adapter._step == 1
    assert (layer.lora_Lambda.abs() > 1e-6).sum().item() == 4


def test_invariant_reallocate_prunes_to_target_r_on_interval(no_peft):
    """On an interval boundary, every manual layer is pruned to ``target_r``
    non-zero lambdas (line 214-215 uniform allocation).
    """
    adapter = AdaLoRAAdapter(
        base=_NestedLinear(), r=4, target_modules=["block"],
        target_r=2, update_interval=2, total_step=10,
    )
    layer = adapter._adalora_layers["block.0"]
    with torch.no_grad():
        layer.lora_Lambda.copy_(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    adapter.maybe_reallocate_rank()  # step 1
    adapter.maybe_reallocate_rank()  # step 2 -> prune
    assert adapter._step == 2
    assert (layer.lora_Lambda.abs() > 1e-6).sum().item() == 2
    # The two surviving components are the largest-magnitude ones.
    survivors = (layer.lora_Lambda.abs() > 1e-6).tolist()
    assert survivors == [True, True, False, False]


def test_invariant_reallocate_noops_when_no_layers(no_peft):
    """When the target matches nothing, ``_adalora_layers`` is empty and the
    ``n_layers == 0`` guard short-circuits before any prune (line 210-212).
    """
    adapter = AdaLoRAAdapter(
        base=_NestedLinear(), r=4, target_modules=["does_not_exist"],
        update_interval=1, total_step=10,
    )
    assert adapter._adalora_layers == {}
    adapter.maybe_reallocate_rank()  # interval hit but n_layers == 0
    assert adapter._step == 1


def test_invariant_reallocate_returns_early_under_peft():
    """Under the PEFT path, ``maybe_reallocate_rank`` increments the step then
    returns immediately (line 206-207 guard).

    Requires PEFT installed (skipped otherwise); the installed PEFT then drives
    ``_use_peft=True``.
    """
    pytest.importorskip("peft")
    adapter = AdaLoRAAdapter(
        base=nn.Sequential(nn.Linear(4, 4)), r=2, target_modules=["0"],
        update_interval=1, total_step=10,
    )
    assert adapter._use_peft is True
    adapter.maybe_reallocate_rank()
    assert adapter._step == 1
    assert not hasattr(adapter, "_adalora_layers")


# ---------------------------------------------------------------------------
# forward normalisation (lines 220-227)
# ---------------------------------------------------------------------------

def test_invariant_forward_passes_through_model_output(no_peft):
    """A base returning ``ModelOutput`` is returned unchanged (line 221-222)."""
    torch.manual_seed(0)
    adapter = AdaLoRAAdapter(
        base=_ReturnsModelOutput(), r=2, target_modules=["lin"], total_step=10
    )
    out = adapter.forward(x=torch.randn(2, 4))
    assert isinstance(out, ModelOutput)
    assert "logits" in out.outputs


def test_invariant_forward_wraps_logits_attribute(no_peft):
    """A base returning an object with ``.logits`` is wrapped (line 224-225)."""
    torch.manual_seed(0)
    adapter = AdaLoRAAdapter(
        base=_ReturnsLogitsAttr(), r=2, target_modules=["lin"], total_step=10
    )
    out = adapter.forward(x=torch.randn(2, 4))
    assert isinstance(out, ModelOutput)
    assert out.outputs["logits"].shape == (2, 4)


def test_invariant_forward_wraps_raw_tensor(no_peft):
    """A base returning a raw tensor is wrapped under ``logits`` (line 227)."""
    torch.manual_seed(0)
    adapter = AdaLoRAAdapter(
        base=_ReturnsTensor(), r=2, target_modules=["lin"], total_step=10
    )
    out = adapter.forward(x=torch.randn(2, 4))
    assert isinstance(out, ModelOutput)
    assert out.outputs["logits"].shape == (2, 4)


def test_pin_current_behavior_forward_non_tensor_yields_empty_outputs(no_peft):
    """Pin: a base returning a non-tensor, non-``.logits`` value produces a
    ``ModelOutput`` with an EMPTY ``outputs`` dict (line 227 ``else {}``).

    This is current behaviour: silently dropping the model output rather than
    raising. Flagged as debatable — a caller expecting logits gets nothing.
    """
    adapter = AdaLoRAAdapter(
        base=_ReturnsWeird(), r=2, target_modules=["lin"], total_step=10
    )
    out = adapter.forward(x=torch.randn(2, 4))
    assert isinstance(out, ModelOutput)
    assert out.outputs == {}


# ---------------------------------------------------------------------------
# state_dict / load_state_dict — manual collection (lines 237-243, 252)
# ---------------------------------------------------------------------------

def test_invariant_manual_state_dict_collects_prefixed_adapter_tensors(no_peft):
    """Manual ``state_dict`` flattens each layer's adapter tensors under a
    ``"<name>.<key>"`` prefix (line 237-243).
    """
    adapter = AdaLoRAAdapter(
        base=_TopLevelLinears(), r=3, target_modules=["lin"], total_step=10
    )
    sd = adapter.state_dict()
    assert set(sd) == {"lin.lora_A", "lin.lora_B", "lin.lora_Lambda"}
    assert sd["lin.lora_A"].shape == (3, 4)


def test_invariant_manual_load_state_dict_returns_incompatible_keys(no_peft):
    """Manual ``load_state_dict`` delegates to the base model with
    ``strict=False`` (line 252) and returns the torch ``_IncompatibleKeys``
    report — adapter-only keys are accepted as missing/unexpected, not raised.
    """
    adapter = AdaLoRAAdapter(
        base=_TopLevelLinears(), r=2, target_modules=["lin"], total_step=10
    )
    result = adapter.load_state_dict(adapter.state_dict(), strict=True)
    # torch returns a NamedTuple-like report with these two fields.
    assert hasattr(result, "missing_keys")
    assert hasattr(result, "unexpected_keys")


# ---------------------------------------------------------------------------
# state_dict / load_state_dict — PEFT helper failure fallbacks (234-235, 250-252)
# ---------------------------------------------------------------------------

def test_pin_current_behavior_peft_state_dict_failure_falls_back(monkeypatch, caplog):
    """Pin: if ``get_peft_model_state_dict`` raises, ``state_dict`` logs a
    warning and degrades to ``super().state_dict()`` (line 234-235).

    Because the PEFT path never sets ``_adalora_layers``, the manual-collection
    branch is skipped and the FULL base state dict is returned instead of an
    adapter-only one — pinned as current behaviour.
    """
    peft = pytest.importorskip("peft")

    adapter = AdaLoRAAdapter(
        base=nn.Sequential(nn.Linear(4, 4)), r=2, target_modules=["0"], total_step=10
    )
    assert adapter._use_peft is True

    def _boom(*_a, **_k):
        raise RuntimeError("peft export exploded")

    monkeypatch.setattr(peft, "get_peft_model_state_dict", _boom)
    with caplog.at_level("WARNING"):
        sd = adapter.state_dict()
    assert "PEFT export failed" in caplog.text
    # Fell through to super().state_dict() (full module weights, non-empty).
    assert not hasattr(adapter, "_adalora_layers")
    assert len(sd) > 0


def test_pin_current_behavior_peft_load_failure_falls_back(monkeypatch, caplog):
    """Pin: if ``set_peft_model_state_dict`` raises, ``load_state_dict`` logs a
    warning and falls back to ``self.model.load_state_dict(..., strict=False)``
    (line 250-252).
    """
    peft = pytest.importorskip("peft")

    adapter = AdaLoRAAdapter(
        base=nn.Sequential(nn.Linear(4, 4)), r=2, target_modules=["0"], total_step=10
    )
    assert adapter._use_peft is True

    def _boom(*_a, **_k):
        raise RuntimeError("peft load exploded")

    monkeypatch.setattr(peft, "set_peft_model_state_dict", _boom)
    with caplog.at_level("WARNING"):
        result = adapter.load_state_dict({}, strict=True)
    assert "PEFT load failed" in caplog.text
    assert hasattr(result, "missing_keys")
