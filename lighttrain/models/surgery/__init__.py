"""Model surgery.

Imperative tools for editing a model **after** it's been constructed:
freeze / unfreeze parameters, resize the token embedding, replace or insert
named submodules (used by HiddenStatesMSELoss(project=True) and by the LoRA
adapter to inject low-rank deltas), reset selected layers, and tie /untie
weights.

All helpers are pure functions over ``nn.Module``; no registry coupling.
"""

from __future__ import annotations

from ._embedding import resize_embedding
from ._freeze import count_trainable, freeze_modules, unfreeze_modules
from ._reinit import reinit_module
from ._replace import add_named_module, get_submodule, replace_module
from ._tie import tie_weights, untie_weights

__all__ = [
    # Freeze
    "freeze_modules",
    "unfreeze_modules",
    "count_trainable",
    # Embedding
    "resize_embedding",
    # Replace / insert
    "replace_module",
    "add_named_module",
    "get_submodule",
    # Re-init
    "reinit_module",
    # Tie / untie
    "tie_weights",
    "untie_weights",
]
