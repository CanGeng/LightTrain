"""Reference policy management for RL training.

A reference policy holds a frozen copy of the model (or points to the
LoRA base weights) so that KL-penalized objectives can compute
log π_ref(y|x) without another forward pass through the training model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ReferencePolicy:
    """Frozen reference model for KL-constrained RL/DPO.

    Attributes
    ----------
    model :
        A frozen ``nn.Module`` or ``None`` if LoRA-base-as-ref mode is used
        (the base weights live inside the LoRA wrapper and are already frozen).
    lora_base_as_ref : bool
        When ``True``, :meth:`log_probs` calls the live model with
        PEFT adapters disabled, reading the bare base-model outputs.
    ignore_index : int
        Token index excluded from log-prob computation (padding).
    """

    model: Any = None
    lora_base_as_ref: bool = False
    ignore_index: int = -100
    _device: torch.device | None = field(default=None, repr=False)

    @property
    def device(self) -> torch.device | None:
        if self._device is not None:
            return self._device
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return None

    @torch.no_grad()
    def log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        live_model: Any | None = None,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Compute log-probs under the reference policy.

        Parameters
        ----------
        input_ids : (B, T)
        attention_mask : (B, T) or None
        labels : (B, T)  — padding marked with ``ignore_index``
        live_model :
            Required when ``lora_base_as_ref=True``; the live LoRA-wrapped model.
        per_token :
            When ``False`` (default) return ``(B,)`` mean per-token log-probs
            (negative NLL average) — the sequence-level signal used by DPO/PPO
            monitoring. When ``True`` return ``(B, T)`` per-token log-probs of
            the realized tokens (next-token targets ``input_ids[:, 1:]`` with a
            leading 0 column), aligned position-for-position with the GRPO
            trainer's ``log_probs_new`` for the per-token k3 KL estimator.

        Returns
        -------
        ``(B,)`` if ``per_token=False`` else ``(B, T)``.
        """
        if per_token and self.lora_base_as_ref:
            raise RuntimeError(
                "ReferencePolicy.log_probs(per_token=True) is not supported with "
                "lora_base_as_ref=True: the LoRA-base reference path needs adapter "
                "toggling + eval-state handling that is not wired yet. Use a "
                "deep-copy reference (lora_base_as_ref=False)."
            )
        if self.lora_base_as_ref:
            return self._lora_base_log_probs(input_ids, attention_mask, labels, live_model)
        if self.model is None:
            raise RuntimeError("ReferencePolicy: model is None and lora_base_as_ref=False.")
        if per_token:
            return self._frozen_log_probs_per_token(self.model, input_ids, attention_mask)
        return self._frozen_log_probs(self.model, input_ids, attention_mask, labels)

    def _frozen_log_probs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = model(**kwargs)
        logits = out.outputs["logits"] if hasattr(out, "outputs") else out["logits"]
        return _sequence_log_probs(logits, labels, self.ignore_index)

    def _frozen_log_probs_per_token(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = model(**kwargs)
        logits = out.outputs["logits"] if hasattr(out, "outputs") else out["logits"]
        return _per_token_log_probs(logits, input_ids)

    def _lora_base_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        live_model: Any,
    ) -> torch.Tensor:
        if live_model is None:
            raise RuntimeError(
                "ReferencePolicy(lora_base_as_ref=True) needs live_model in log_probs()."
            )
        # Disable LoRA adapters to expose base weights only.
        try:
            live_model.disable_adapter_layers()
            result = self._frozen_log_probs(live_model, input_ids, attention_mask, labels)
        finally:
            live_model.enable_adapter_layers()
        return result


def _sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-sample mean log-prob (negative mean NLL) from logits + labels.

    Performs the causal shift internally so both logits and labels are the
    full-length sequences as returned by the model.

    Returns (B,) tensor.
    """
    # Shift: predict token t+1 from token t context.
    shift_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()          # (B, T-1)

    B, T, V = shift_logits.shape
    log_probs = F.log_softmax(shift_logits, dim=-1)    # (B, T-1, V)
    # Gather the log-prob for the target token at each position.
    target_ids = shift_labels.clamp(min=0)             # avoid -100 index errors
    gathered = torch.gather(
        log_probs, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)                                       # (B, T-1)

    mask = (shift_labels != ignore_index).float()
    per_sample = (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_sample  # (B,)


def _per_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probs of the realized ``input_ids``, shape ``(B, T)``.

    Mirrors the GRPO trainer's ``log_probs_new``: causal shift, gather the
    log-prob of the *actual* next token (``input_ids[:, 1:]`` — **not** labels),
    then prepend a 0 column so the result aligns position-for-position with the
    policy log-probs the KL estimator subtracts against.
    """
    shift_logits = logits[:, :-1, :].contiguous()        # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()         # (B, T-1)
    lp = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(lp, -1, shift_targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return torch.cat([torch.zeros_like(gathered[:, :1]), gathered], dim=1)    # (B, T)


def freeze_as_ref(
    model: Any,
    *,
    lora_base_as_ref: bool = False,
    ignore_index: int = -100,
) -> ReferencePolicy:
    """Create a frozen reference policy from a model.

    Parameters
    ----------
    model :
        The training model (``nn.Module``).
    lora_base_as_ref : bool
        If ``True``, no copy is made; ``log_probs`` will temporarily disable
        LoRA adapters on the live model to read base-model outputs.
    ignore_index : int
        Padding token index.

    Returns
    -------
    :class:`ReferencePolicy`
    """
    if lora_base_as_ref:
        return ReferencePolicy(
            model=None,
            lora_base_as_ref=True,
            ignore_index=ignore_index,
        )
    # Deep-copy and freeze.
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ReferencePolicy(
        model=ref_model,
        lora_base_as_ref=False,
        ignore_index=ignore_index,
    )


def ref_log_probs(
    ref_policy: ReferencePolicy,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    live_model: Any | None = None,
) -> torch.Tensor:
    """Convenience wrapper around :meth:`ReferencePolicy.log_probs`."""
    return ref_policy.log_probs(
        input_ids, attention_mask, labels, live_model=live_model
    )


__all__ = [
    "ReferencePolicy",
    "freeze_as_ref",
    "ref_log_probs",
]
