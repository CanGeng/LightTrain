"""AdaLoRAAdapter + AdaLoRALinear — DESIGN §8.4 (M7, M5 defer).

Relocated from the flat ``tests/test_peft_adalora.py``. No mirror under
``tests/models/`` covered AdaLoRA, so behaviors are preserved. ``AdaLoRALinear``
is always manual (no PEFT dependency); ``AdaLoRAAdapter`` is path-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.peft._adalora import (
    AdaLoRAAdapter,
    AdaLoRALinear,
)


def _make_model():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


# ---------------------------------------------------------------------------
# AdaLoRALinear (always manual — no PEFT dependency)
# ---------------------------------------------------------------------------

def test_invariant_adalora_linear_preserves_base_output_shape():
    """Invariant: wrapping ``nn.Linear(16, 32)`` in ``AdaLoRALinear`` leaves the
    output feature dimension intact — forward of a (2, 16) input yields (2, 32).
    """
    base = nn.Linear(16, 32)
    ada = AdaLoRALinear(base, r=4, lora_alpha=8)
    x = torch.randn(2, 16)
    out = ada(x)
    assert out.shape == (2, 32)


def test_invariant_adalora_linear_exposes_lambda_of_rank_length():
    """Invariant: ``AdaLoRALinear`` holds a learnable ``lora_Lambda`` vector
    whose length equals the rank ``r``.
    """
    base = nn.Linear(8, 16)
    ada = AdaLoRALinear(base, r=4, lora_alpha=4)
    assert hasattr(ada, "lora_Lambda")
    assert ada.lora_Lambda.shape == (4,)


def test_invariant_adalora_prune_rank_keeps_at_most_k_nonzero_lambdas():
    """Invariant: ``prune_rank(keep=k)`` zeros all but (at most) the ``k``
    highest-importance lambdas.

    Setup: set Lambda to a single dominant entry; prune to keep=1.
    Expected: at most one lambda remains non-zero.
    """
    base = nn.Linear(8, 16)
    ada = AdaLoRALinear(base, r=4, lora_alpha=4)
    with torch.no_grad():
        ada.lora_Lambda.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    ada.prune_rank(keep=1)
    assert (ada.lora_Lambda.abs() > 1e-6).sum() <= 1


def test_invariant_adalora_importance_scores_are_nonnegative_rank_length():
    """Invariant: ``importance_scores()`` returns a length-``r`` vector of
    non-negative values.
    """
    base = nn.Linear(4, 8)
    ada = AdaLoRALinear(base, r=3, lora_alpha=3)
    scores = ada.importance_scores()
    assert scores.shape == (3,)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# AdaLoRAAdapter (path-agnostic — PEFT or manual)
# ---------------------------------------------------------------------------

def test_adalora_adapter_constructs_over_sequential_targets():
    """``AdaLoRAAdapter`` constructs without error over a plain ``nn.Sequential``
    base when targeting submodules ``"0"`` and ``"2"``.
    """
    model = _make_model()
    adapter = AdaLoRAAdapter(base=model, r=4, target_modules=["0", "2"], total_step=100)
    assert adapter is not None


def test_adalora_state_dict_is_nonempty_on_either_path():
    """Invariant: regardless of the PEFT-or-manual code path, the adapter's
    ``state_dict()`` carries at least one tensor.
    """
    model = _make_model()
    adapter = AdaLoRAAdapter(base=model, r=4, target_modules=["0", "2"], total_step=100)
    sd = adapter.state_dict()
    assert len(sd) > 0
