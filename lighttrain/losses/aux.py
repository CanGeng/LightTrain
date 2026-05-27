"""Auxiliary losses — InfoNCE / MoEBalance.

These are typically added to a primary loss via CompositeLoss.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..protocols import LossContext, ModelOutput
from ..registry import register


# ---------------------------------------------------------------------------
# InfoNCE
# ---------------------------------------------------------------------------


@register("loss", "info_nce")
class InfoNCELoss:
    """InfoNCE / NT-Xent contrastive loss (van den Oord et al., 2018).

    Expects embeddings under two keys in the batch (e.g. from two augmented
    views or anchor/positive pairs):

        batch[anchor_key]    (B, D)
        batch[positive_key]  (B, D)

    The loss is cross-entropy where each anchor's positive is the diagonal
    entry of the similarity matrix and all other B-1 samples are in-batch
    negatives.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        anchor_key: str = "embeddings_anchor",
        positive_key: str = "embeddings_positive",
        normalize: bool = True,
    ) -> None:
        self.temperature = float(temperature)
        self.anchor_key = anchor_key
        self.positive_key = positive_key
        self.normalize = bool(normalize)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        if self.anchor_key not in batch:
            raise KeyError(f"InfoNCELoss needs '{self.anchor_key}' in batch.")
        if self.positive_key not in batch:
            raise KeyError(f"InfoNCELoss needs '{self.positive_key}' in batch.")

        z1: torch.Tensor = batch[self.anchor_key]
        z2: torch.Tensor = batch[self.positive_key]

        if self.normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        B = z1.size(0)
        # Similarity matrix (B, B)
        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=z1.device)
        # Symmetric loss
        loss = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        ) / 2.0
        return {"loss": loss}


# ---------------------------------------------------------------------------
# MoE Balance
# ---------------------------------------------------------------------------


@register("loss", "moe_balance")
class MoEBalanceLoss:
    """Expert load-balance auxiliary loss for Mixture-of-Experts models.

    Minimizes the correlation between router probabilities and expert usage
    fraction, encouraging uniform expert utilization.

    Expects:

        ctx.extras["router_probs"]   (B, T, E)  — per-token expert softmax probs
        ctx.extras["expert_mask"]    (B, T, E)  — one-hot top-k selection (optional)

    If ``expert_mask`` is absent, uses ``router_probs`` directly as a soft
    load estimate.

    Loss = num_experts * sum(fraction_i * prob_i)   (Switch Transformer style)
    """

    def __init__(self, *, weight: float = 1e-2, num_experts: int | None = None) -> None:
        self.weight = float(weight)
        self.num_experts = num_experts

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],  # noqa: ARG002
        batch: Mapping[str, Any],  # noqa: ARG002
        ctx: LossContext,
    ) -> dict[str, Any]:
        router_probs: torch.Tensor | None = ctx.extras.get("router_probs")
        if router_probs is None:
            # Try to read from model_output.extras
            if isinstance(model_output, ModelOutput) and "router_probs" in model_output.extras:
                router_probs = model_output.extras["router_probs"]
            else:
                raise KeyError(
                    "MoEBalanceLoss needs ctx.extras['router_probs'] (B, T, E). "
                    "Ensure the MoE model stores router_probs in ctx.extras or model_output.extras."
                )

        expert_mask: torch.Tensor | None = ctx.extras.get("expert_mask")
        if expert_mask is None and isinstance(model_output, ModelOutput):
            expert_mask = model_output.extras.get("expert_mask")

        # router_probs: (B, T, E) → mean over (B, T) → (E,)
        B, T, E = router_probs.shape
        n_experts = self.num_experts or E

        # fraction_e = fraction of tokens dispatched to expert e
        if expert_mask is not None:
            fraction = expert_mask.float().mean(dim=(0, 1))  # (E,)
        else:
            # soft load: use router probs directly
            fraction = router_probs.mean(dim=(0, 1))  # (E,)

        # mean router prob per expert
        prob_mean = router_probs.mean(dim=(0, 1))  # (E,)

        aux_loss = self.weight * n_experts * (fraction * prob_mean).sum()
        return {"loss": aux_loss, "moe_balance_aux": float(aux_loss.detach())}


__all__ = ["InfoNCELoss", "MoEBalanceLoss"]
