"""Edge + error-path tests for ``lighttrain.models.surgery``.

Complements tests/models/test_surgery.py (happy paths) by pinning the branches
it didn't reach:

* **_embedding**: HF-adapter delegation; ``normal`` init; ``_init_rows``
  ``mean``-guard + unknown-mode errors; tied-head-with-bias and untied-head
  resize paths.
* **_reinit**: every ``_apply_dist`` kind (zeros / ones / xavier / orthogonal
  on 2-D vs 1-D / uniform / unknown) and iterable patterns.
* **_replace**: invalid dotted path, missing intermediate, non-Module factory,
  empty-path ``get_submodule``.
* **_tie**: missing-weight (tie & untie) and shape-mismatch guards.
* **_freeze**: iterable patterns; ``unfreeze`` skips already-trainable params.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.models.surgery import (
    freeze_modules,
    get_submodule,
    reinit_module,
    replace_module,
    resize_embedding,
    tie_weights,
    unfreeze_modules,
    untie_weights,
)
from lighttrain.models.surgery._embedding import _init_rows


def _make_tiny() -> TinyCausalLM:
    return TinyCausalLM(vocab_size=64, d_model=16, n_layers=2, n_heads=4, max_seq_len=32)


# ===========================================================================
# _embedding
# ===========================================================================

class _FakeInner:
    def __init__(self) -> None:
        self.resized_to: int | None = None

    def resize_token_embeddings(self, n: int) -> None:
        self.resized_to = n


class _FakeHF(nn.Module):
    """Mimics HFCausalLM: delegates resize to an ``inner`` model."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _FakeInner()
        self.vocab_size = 10


def test_resize_embedding_delegates_to_hf_inner():
    """A model exposing ``inner.resize_token_embeddings`` is delegated to, and
    its ``vocab_size`` mirror is updated."""
    m = _FakeHF()
    resize_embedding(m, 20)
    assert m.inner.resized_to == 20
    assert m.vocab_size == 20


def test_resize_embedding_normal_init_fills_new_rows():
    """``init="normal"`` populates the new rows (non-zero with prob ~1)."""
    model = _make_tiny()
    old = model.tok_emb.weight.size(0)
    resize_embedding(model, old + 4, init="normal")
    new_rows = model.tok_emb.weight[old:]
    assert new_rows.shape == (4, model.d_model)
    assert new_rows.abs().sum().item() > 0.0


def test_init_rows_mean_mode_is_guarded():
    """``_init_rows`` rejects ``mean`` — that mode is handled by
    resize_embedding directly (defensive guard)."""
    with pytest.raises(RuntimeError, match="mean"):
        _init_rows(torch.zeros(2, 4), mode="mean")


def test_init_rows_unknown_mode_raises():
    """``_init_rows`` rejects an unknown mode."""
    with pytest.raises(ValueError, match="Unknown init mode"):
        _init_rows(torch.zeros(2, 4), mode="bogus")  # type: ignore[arg-type]


def test_resize_embedding_tied_head_with_bias_copies_and_zeros():
    """Growing a tied head that has a bias: weight stays shared with the new
    embedding; old bias rows are preserved, new bias rows zeroed."""
    model = _make_tiny()
    old = model.tok_emb.weight.size(0)
    head = nn.Linear(model.d_model, old, bias=True)
    head.weight = model.tok_emb.weight  # re-tie storage
    with torch.no_grad():
        head.bias.fill_(3.0)
    model.lm_head = head

    resize_embedding(model, old + 4, init="zeros")

    assert model.lm_head.weight is model.tok_emb.weight  # still tied
    assert torch.allclose(model.lm_head.bias[:old], torch.full((old,), 3.0))
    assert torch.all(model.lm_head.bias[old:] == 0.0)


def test_resize_embedding_untied_head_mean_init_and_bias():
    """An untied head (independent weight) grows on its own: old rows kept,
    new rows = mean of old head rows, new bias zeroed."""
    model = _make_tiny()
    old = model.tok_emb.weight.size(0)
    head = nn.Linear(model.d_model, old, bias=True)  # fresh, NOT tied
    with torch.no_grad():
        head.bias.fill_(2.0)
    head_snapshot = head.weight.detach().clone()
    model.lm_head = head

    resize_embedding(model, old + 4, init="mean")

    assert model.lm_head.weight is not model.tok_emb.weight  # untied
    assert torch.allclose(model.lm_head.weight[:old], head_snapshot)
    assert torch.allclose(model.lm_head.weight[old], head_snapshot.mean(dim=0), atol=1e-6)
    assert torch.all(model.lm_head.bias[old:] == 0.0)


def test_resize_embedding_untied_head_zeros_init():
    """Untied head with ``init="zeros"`` zero-fills the new weight rows."""
    model = _make_tiny()
    old = model.tok_emb.weight.size(0)
    model.lm_head = nn.Linear(model.d_model, old, bias=False)  # untied, no bias
    resize_embedding(model, old + 3, init="zeros")
    assert torch.all(model.lm_head.weight[old:] == 0.0)


# ===========================================================================
# _reinit  (_apply_dist kinds)
# ===========================================================================

class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)      # weight 2-D, bias 1-D
        self.norm = nn.LayerNorm(4)     # weight 1-D, bias 1-D


def test_reinit_zeros_kind_zeros_weight_and_bias():
    net = _Net()
    hits = reinit_module(net, "lin", dist={"kind": "zeros"})
    assert hits == 1
    assert torch.all(net.lin.weight == 0.0)
    assert torch.all(net.lin.bias == 0.0)


def test_reinit_ones_kind_sets_weight_to_one():
    net = _Net()
    reinit_module(net, "lin", dist={"kind": "ones"})
    assert torch.all(net.lin.weight == 1.0)
    assert torch.all(net.lin.bias == 0.0)  # bias always zeroed


def test_reinit_xavier_uniform_on_2d_changes_weight():
    net = _Net()
    before = net.lin.weight.detach().clone()
    reinit_module(net, "lin", dist={"kind": "xavier_uniform"})
    assert not torch.allclose(net.lin.weight, before)


def test_reinit_xavier_uniform_on_1d_falls_back_to_zeros():
    """xavier on a 1-D parameter (LayerNorm weight) → the dim<2 → zeros branch."""
    net = _Net()
    reinit_module(net, "norm", dist={"kind": "xavier_uniform"})
    assert torch.all(net.norm.weight == 0.0)


def test_reinit_orthogonal_on_2d_is_orthogonal():
    net = _Net()
    reinit_module(net, "lin", dist={"kind": "orthogonal"})
    w = net.lin.weight.detach()
    assert torch.allclose(w @ w.t(), torch.eye(4), atol=1e-5)


def test_reinit_orthogonal_on_1d_falls_back_to_zeros():
    net = _Net()
    reinit_module(net, "norm", dist={"kind": "orthogonal"})
    assert torch.all(net.norm.weight == 0.0)


def test_reinit_uniform_kind_respects_bounds():
    net = _Net()
    reinit_module(net, "lin", dist={"kind": "uniform", "low": -0.1, "high": 0.1})
    assert net.lin.weight.min().item() >= -0.1
    assert net.lin.weight.max().item() <= 0.1


def test_reinit_unknown_kind_raises():
    net = _Net()
    with pytest.raises(ValueError, match="Unknown reinit dist kind"):
        reinit_module(net, "lin", dist={"kind": "bogus"})


def test_reinit_accepts_iterable_pattern():
    """A list of regexes matches multiple modules (the iterable _compile path)."""
    net = _Net()
    hits = reinit_module(net, ["lin", "norm"], dist={"kind": "zeros"})
    assert hits == 2


# ===========================================================================
# _replace
# ===========================================================================

def test_replace_module_rejects_invalid_dotted_path():
    with pytest.raises(ValueError, match="Invalid dotted path"):
        replace_module(_make_tiny(), "a..b", nn.Identity())


def test_replace_module_rejects_missing_intermediate():
    with pytest.raises(AttributeError, match="not found"):
        replace_module(_make_tiny(), "ghost.leaf", nn.Identity())


def test_replace_module_rejects_non_module_factory_result():
    with pytest.raises(TypeError, match="must yield nn.Module"):
        replace_module(_make_tiny(), "lm_head", lambda old: 123)  # type: ignore[arg-type, return-value]


def test_get_submodule_empty_path_returns_model():
    model = _make_tiny()
    assert get_submodule(model, "") is model


# ===========================================================================
# _tie
# ===========================================================================

class _TieNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.c = nn.Linear(4, 8)   # different shape from a
        self.act = nn.ReLU()       # no ``.weight``


def test_tie_weights_requires_weight_on_both_sides():
    with pytest.raises(TypeError, match="must expose a ``.weight``"):
        tie_weights(_TieNet(), "a", "act")


def test_tie_weights_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        tie_weights(_TieNet(), "a", "c")


def test_untie_weights_requires_weight():
    with pytest.raises(TypeError, match="does not expose a ``.weight``"):
        untie_weights(_TieNet(), "act")


# ===========================================================================
# _freeze
# ===========================================================================

def test_freeze_modules_accepts_iterable_pattern():
    model = _make_tiny()
    n = freeze_modules(model, [r"blocks\.0\.", r"tok_emb"])
    assert n > 0
    assert not model.tok_emb.weight.requires_grad


def test_unfreeze_skips_already_trainable_params():
    """``unfreeze_modules`` over a partially-frozen model only flips the frozen
    params (already-trainable ones hit the early ``continue``)."""
    model = _make_tiny()
    frozen = freeze_modules(model, r"blocks\.0\.")
    unfrozen = unfreeze_modules(model, r".*")  # matches everything
    # Only the previously-frozen params are flipped back.
    assert unfrozen == frozen
    assert all(p.requires_grad for p in model.parameters())
