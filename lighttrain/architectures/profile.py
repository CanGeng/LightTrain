"""ArchitectureProfile and ObjectiveProfile.

ArchitectureProfile
    Describes the structural seams of an architecture without hard-coding
    Transformer assumptions.  Consumed by LayerOffloadEngine (block slicing),
    StatefulTrainer (state reset), and the training loop (loss_family routing).

ObjectiveProfile
    A callable that acts as loss_fn *and* owns the batch-preparation step
    (noise injection for diffusion, masking for JEPA / MLM, etc.).  The
    LossContext.loss_family field is set from objective.loss_family before
    dispatch, enabling downstream loss fns to specialise by paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

import torch.nn as nn


# ---------------------------------------------------------------------------
# ArchitectureProfile
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureProfile:
    """Structural description of an architecture.

    All fields except ``name`` and ``loss_family`` are optional; callers must
    guard against ``None`` before invoking seam functions.
    """

    name: str
    loss_family: str
    """Canonical loss paradigm: next_token | mlm | diffusion | flow_matching
    | jepa | masked_denoising."""

    state_mode: str = "stateless"
    """'stateless' for standard Transformers; 'stateful' for RWKV / Mamba."""

    # Seam functions — all accept the *instantiated* nn.Module as first arg.

    block_iterator_fn: Callable[[nn.Module], Iterator[nn.Module]] | None = None
    """Yield the trainable blocks in depth order (for LayerOffload slicing)."""

    embedding_layer_fn: Callable[[nn.Module], nn.Module] | None = None
    """Return the token/patch embedding layer (for resize_embedding surgery)."""

    head_layer_fn: Callable[[nn.Module], nn.Module] | None = None
    """Return the output projection / head (for head surgery)."""

    reset_state_fn: Callable[[nn.Module], None] | None = None
    """Zero or reinitialise the recurrent state (stateful architectures only).
    Called at document boundaries detected by the training loop."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary plugin-specific metadata."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def iter_blocks(self, model: nn.Module) -> Iterator[nn.Module]:
        if self.block_iterator_fn is None:
            raise NotImplementedError(
                f"ArchitectureProfile '{self.name}' has no block_iterator_fn."
            )
        return self.block_iterator_fn(model)

    def get_embedding(self, model: nn.Module) -> nn.Module:
        if self.embedding_layer_fn is None:
            raise NotImplementedError(
                f"ArchitectureProfile '{self.name}' has no embedding_layer_fn."
            )
        return self.embedding_layer_fn(model)

    def get_head(self, model: nn.Module) -> nn.Module:
        if self.head_layer_fn is None:
            raise NotImplementedError(
                f"ArchitectureProfile '{self.name}' has no head_layer_fn."
            )
        return self.head_layer_fn(model)

    def reset_state(self, model: nn.Module) -> None:
        if self.reset_state_fn is None:
            raise NotImplementedError(
                f"ArchitectureProfile '{self.name}' has no reset_state_fn "
                f"(state_mode='{self.state_mode}')."
            )
        self.reset_state_fn(model)


# ---------------------------------------------------------------------------
# ObjectiveProfile Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ObjectiveProfile(Protocol):
    """A training objective that owns both batch prep and loss computation.

    Implement this protocol to plug non-standard objectives (diffusion,
    flow-matching, JEPA, …) into PretrainTrainer without subclassing it.

    Lifecycle per training step::

        batch = objective.prepare_batch(raw_batch, step=step, device=device)
        loss_dict = objective(model_output, batch, loss_ctx)
    """

    loss_family: str

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """Transform the raw collated batch before the forward pass.

        Examples: inject Gaussian noise (diffusion), randomly mask tokens
        (MLM / masked-denoising), sample JEPA target patch indices.

        Returns a new dict (or the same dict mutated in-place) that is passed
        to ``model(**batch)`` and later to ``__call__``.
        """
        ...

    def __call__(
        self,
        outputs: Any,  # ModelOutput
        batch: dict,
        ctx: Any,       # LossContext
    ) -> dict:
        """Compute the loss from model outputs and the prepared batch.

        Must return a dict with at least ``{"loss": Tensor}``.
        Additional scalar entries (e.g. ``"mse_loss"``, ``"kl_loss"``) are
        logged automatically by the update rule.
        """
        ...


__all__ = ["ArchitectureProfile", "ObjectiveProfile"]
