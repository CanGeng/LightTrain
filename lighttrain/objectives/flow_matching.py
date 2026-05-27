"""FlowMatchingObjective — rectified flow / OT-CFM.

Implements two flow-matching variants:

    * "rectified_flow" (Liu et al., 2022) — straight-line paths from noise
      to data; the velocity field is v(x_t, t) = x₁ − x₀.
    * "ot_cfm" (Tong et al., 2023) — optimal-transport conditioned flow
      matching; same objective but path constructed via OT minibatch coupling.

Both variants interpolate ``x_t = (1-t)·x₀ + t·x₁`` and train the velocity
field ``v_θ(x_t, t)`` to match ``x₁ − x₀`` with MSE.

Model contract: ``outputs.outputs["v"]`` — predicted velocity (B, *data_shape).

Usage in YAML::

    objective:
      name: flow_matching
      variant: rectified_flow   # rectified_flow | ot_cfm
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..protocols import LossContext, ModelOutput
from ..registry import register


@register("objective", "flow_matching")
class FlowMatchingObjective:
    """Rectified flow / OT-CFM training objective.

    Args:
        variant: "rectified_flow" (default) or "ot_cfm".
        sigma_min: Minimum noise scale added to OT couplings (OT-CFM only).
    """

    loss_family: str = "flow_matching"

    def __init__(
        self,
        variant: str = "rectified_flow",
        sigma_min: float = 1e-4,
    ) -> None:
        if variant not in ("rectified_flow", "ot_cfm"):
            raise ValueError(
                f"Unknown flow_matching variant '{variant}'. "
                "Use 'rectified_flow' or 'ot_cfm'."
            )
        self.variant = variant
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # ObjectiveProfile protocol
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """Interpolate between noise x₀ and data x₁.

        Expects ``batch["x"]`` — clean data (B, *shape).
        Adds:
            * ``batch["x0"]``  — source (pure noise)
            * ``batch["x1"]``  — target (clean data)
            * ``batch["x_t"]`` — interpolated at sampled t
            * ``batch["t"]``   — scalar timestep ∈ (0,1)  shape (B,)
            * ``batch["ut"]``  — target velocity x₁ − x₀  shape (B, *shape)
        """
        x1 = batch["x"]
        B = x1.shape[0]
        x0 = torch.randn_like(x1)

        if self.variant == "ot_cfm":
            # Minibatch OT: sort x0 and x1 by their L2 distance (greedy row-wise)
            # Full LP is expensive; greedy coupling is a common approximation.
            x0 = _greedy_ot_coupling(x0, x1)

        t = torch.rand(B, device=x1.device)
        t_view = t.view(B, *([1] * (x1.dim() - 1)))

        # OT-CFM adds sigma_min perturbation to keep paths non-degenerate
        sigma = self.sigma_min if self.variant == "ot_cfm" else 0.0
        x_t = (1.0 - (1.0 - sigma) * t_view) * x0 + t_view * x1
        ut = x1 - (1.0 - sigma) * x0  # target velocity (constant along path)

        batch = {**batch, "x0": x0, "x1": x1, "x_t": x_t, "t": t, "ut": ut}
        return batch

    def __call__(
        self,
        outputs: ModelOutput,
        batch: dict,
        ctx: LossContext,
    ) -> dict:
        ctx.loss_family = self.loss_family

        v_pred = outputs.outputs.get("v")
        if v_pred is None:
            raise KeyError(
                "FlowMatchingObjective expects ModelOutput.outputs['v']. "
                "Ensure the model returns {'v': tensor}."
            )
        ut = batch["ut"]
        mse = F.mse_loss(v_pred, ut, reduction="mean")
        return {"loss": mse, "flow_mse": mse.detach()}


# ---------------------------------------------------------------------------
# Greedy OT coupling helper
# ---------------------------------------------------------------------------

def _greedy_ot_coupling(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Greedy row-permutation approximation of minibatch OT.

    For each sample in x1, finds the closest sample in x0 (L2) without
    replacement.  O(B²) — only used in prepare_batch for small batches.
    """
    B = x0.shape[0]
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    # Pairwise L2²
    dists = torch.cdist(x1_flat, x0_flat)  # (B, B)
    assigned = torch.full((B,), -1, dtype=torch.long, device=x0.device)
    used = set()
    for i in range(B):
        row = dists[i]
        for j in torch.argsort(row):
            j_int = int(j)
            if j_int not in used:
                assigned[i] = j_int
                used.add(j_int)
                break
    return x0[assigned]


__all__ = ["FlowMatchingObjective"]
