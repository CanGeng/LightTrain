"""JEPAObjective — masked patch prediction with EMA target encoder.

Implements the I-JEPA training objective (Assran et al., 2023):

1.  ``prepare_batch`` samples context and target patch indices from ``batch["patches"]``.
2.  During forward the model encodes *context patches only* and passes
    context embeddings + target positions to a predictor.
3.  The EMA target encoder encodes *all patches*; predictions are compared
    to the target encoder's masked-patch embeddings with MSE.

Model contract:
    * Input:  ``batch["context_patches"]``, ``batch["target_positions"]``
    * Output: ``outputs.outputs["pred_embeddings"]``  — predictor output (B, M, D)

The EMA update is triggered each step via ``objective.ema_step(student_encoder)``.

Usage in YAML::

    objective:
      name: jepa
      num_context_patches: 96
      num_target_patches: 16
      ema_momentum: 0.996
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


@register("objective", "jepa")
class JEPAObjective:
    """JEPA masked-patch prediction objective.

    Args:
        num_context_patches: Number of visible context patches fed to the encoder.
        num_target_patches:  Number of target patches the predictor must predict.
        ema_momentum:        EMA decay coefficient for the target encoder.
    """

    loss_family: str = "jepa"

    def __init__(
        self,
        num_context_patches: int = 96,
        num_target_patches: int = 16,
        ema_momentum: float = 0.996,
    ) -> None:
        self.num_context = num_context_patches
        self.num_target = num_target_patches
        self.ema_momentum = ema_momentum
        self._target_encoder: Any = None

    # ------------------------------------------------------------------
    # EMA management
    # ------------------------------------------------------------------

    def set_target_encoder(self, encoder: torch.nn.Module) -> None:
        """Register the target (EMA) encoder.

        Called once by the trainer after model construction.  The encoder
        should be a separate copy of the context encoder, managed here.
        """
        self._target_encoder = encoder
        for p in encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_step(self, student_encoder: torch.nn.Module) -> None:
        """Update target encoder via EMA: θ_t ← m·θ_t + (1−m)·θ_s."""
        if self._target_encoder is None:
            return
        m = self.ema_momentum
        for ps, pt in zip(
            student_encoder.parameters(), self._target_encoder.parameters()
        ):
            pt.data.mul_(m).add_((1.0 - m) * ps.data)

    # ------------------------------------------------------------------
    # ObjectiveProfile protocol
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """Sample context and target patch indices.

        Expects ``batch["patches"]`` — shape (B, N, D) where N = total patches.
        Adds:
            * ``batch["context_patches"]``  — (B, num_context, D)
            * ``batch["target_patches"]``   — (B, num_target, D)  (ground truth)
            * ``batch["context_idx"]``      — (B, num_context)
            * ``batch["target_idx"]``       — (B, num_target)
        """
        patches = batch["patches"]
        B, N, D = patches.shape
        nc = min(self.num_context, N - 1)
        nt = min(self.num_target, N - nc)

        context_idx = torch.zeros(B, nc, dtype=torch.long, device=patches.device)
        target_idx = torch.zeros(B, nt, dtype=torch.long, device=patches.device)
        for i in range(B):
            perm = torch.randperm(N, device=patches.device)
            context_idx[i] = perm[:nc]
            target_idx[i] = perm[nc: nc + nt]

        ctx_patches = patches[torch.arange(B).unsqueeze(1), context_idx]   # (B, nc, D)
        tgt_patches = patches[torch.arange(B).unsqueeze(1), target_idx]    # (B, nt, D)

        batch = {
            **batch,
            "context_patches": ctx_patches,
            "target_patches": tgt_patches,
            "context_idx": context_idx,
            "target_idx": target_idx,
        }
        return batch

    def __call__(
        self,
        outputs: ModelOutput,
        batch: dict,
        ctx: LossContext,
    ) -> dict:
        """MSE between predictor output and target encoder embeddings."""
        ctx.loss_family = self.loss_family

        pred_emb = outputs.outputs.get("pred_embeddings")
        if pred_emb is None:
            raise KeyError(
                "JEPAObjective expects ModelOutput.outputs['pred_embeddings']. "
                "Make sure the JEPAModel returns {'pred_embeddings': tensor}."
            )

        # Target: stop-gradient embeddings from target encoder
        # If target encoder is available, compute on-the-fly; otherwise
        # expect 'target_embeddings' pre-computed in extras.
        target_emb = outputs.extras.get("target_embeddings")
        if target_emb is None:
            target_emb = batch.get("target_patches")  # fallback to raw patches
        if target_emb is None:
            raise KeyError(
                "JEPAObjective: cannot find target embeddings. "
                "Provide 'target_embeddings' in ModelOutput.extras or "
                "'target_patches' in batch."
            )

        loss = F.mse_loss(pred_emb, target_emb.detach())
        return {"loss": loss, "jepa_mse": loss.detach()}


__all__ = ["JEPAObjective"]
