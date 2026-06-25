"""MaskedDenoisingObjective — BERT-style MLM / masked token denoising.

Randomly masks a fraction of input token IDs and trains the model to predict
the original tokens at masked positions (cross-entropy).  Compatible with
any model whose forward returns ``outputs.outputs["logits"]`` of shape (B, T, V).

Masking strategy (identical to BERT):
    * With probability ``mask_prob``:  replace with ``[MASK]`` id (default 103)
    * Of those, with probability 0.1: replace with random token id
    * Of those, with probability 0.1: keep the original token

Usage in YAML::

    objective:
      name: masked_denoising
      mask_prob: 0.15
      mask_token_id: 103
      vocab_size: 32000   # used for random-token replacement
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


@register("objective", "masked_denoising")
class MaskedDenoisingObjective:
    """BERT-style masked token denoising objective.

    Args:
        mask_prob:     Fraction of tokens to mask (default 0.15).
        mask_token_id: Token ID used for the ``[MASK]`` replacement.
        vocab_size:    Vocabulary size (for random-token replacement).
        ignore_index:  Label value for non-masked tokens (not counted in loss).
    """

    loss_family: str = "masked_denoising"

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 103,
        vocab_size: int = 32000,
        ignore_index: int = -100,
    ) -> None:
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------
    # ObjectiveProfile protocol
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, *, step: int, device: Any) -> dict:
        """Randomly mask tokens in ``batch["input_ids"]``.

        Adds / replaces:
            * ``batch["input_ids"]``  — corrupted ids (MASK / random / original)
            * ``batch["mlm_labels"]`` — original ids at masked positions,
                                        ignore_index elsewhere (B, T)
        """
        input_ids: torch.Tensor = batch["input_ids"]
        B, T = input_ids.shape

        # 1. Decide which positions to process
        probability_matrix = torch.full((B, T), self.mask_prob, device=input_ids.device)
        # Don't mask padding (attention_mask == 0)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            probability_matrix = probability_matrix * attn_mask.float()

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 2. Build labels
        labels = input_ids.clone()
        labels[~masked_indices] = self.ignore_index

        # 3. Replace tokens
        corrupted = input_ids.clone()

        # 80 %: [MASK]
        replace_with_mask = masked_indices & (torch.rand_like(probability_matrix) < 0.8)
        corrupted[replace_with_mask] = self.mask_token_id

        # 10 %: random token
        replace_with_random = (
            masked_indices & ~replace_with_mask & (torch.rand_like(probability_matrix) < 0.5)
        )
        random_tokens = torch.randint(
            0, self.vocab_size, (int(replace_with_random.sum()),), device=input_ids.device
        )
        corrupted[replace_with_random] = random_tokens

        # 10 %: keep original (no change needed)

        batch = {**batch, "input_ids": corrupted, "mlm_labels": labels}
        return batch

    def __call__(
        self,
        outputs: ModelOutput,
        batch: dict,
        ctx: LossContext,
    ) -> dict:
        ctx.loss_family = self.loss_family

        logits = outputs.outputs.get("logits")
        if logits is None:
            raise KeyError(
                "MaskedDenoisingObjective expects ModelOutput.outputs['logits'] "
                "of shape (B, T, V)."
            )
        labels = batch.get("mlm_labels")
        if labels is None:
            raise KeyError(
                "MaskedDenoisingObjective: 'mlm_labels' not found in batch. "
                "Call prepare_batch before forward."
            )

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            labels.view(B * T),
            ignore_index=self.ignore_index,
        )
        return {"loss": loss, "mlm_loss": loss.detach()}


__all__ = ["MaskedDenoisingObjective"]
