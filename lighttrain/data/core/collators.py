"""Causal-LM collator: right-pad → build shifted labels.

Returns a dict of tensors:
    input_ids: (B, T) int64
    attention_mask: (B, T) int64 (1 for real, 0 for pad)
    labels: (B, T) int64 (-100 on pad positions)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...registry import register


@register("collator", "causal_lm")
class CausalLMCollator:
    """Right-pad to longest in batch (capped at ``max_len``).

    Labels mirror ``input_ids`` with ``-100`` on pad positions. The
    ``CrossEntropyLoss`` (see :mod:`lighttrain.losses.core`) performs the
    next-token off-by-one shift: ``logits[:, :-1, :]`` vs ``labels[:, 1:]``.
    The collator itself does not pre-shift — keep labels aligned with input
    positions so SFT / masked-LM / response-only-mask flows can use the same
    pad logic without bookkeeping.
    """

    def __init__(self, pad_id: int, max_len: int = 1024,
                 label_ignore: int = -100) -> None:
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.label_ignore = int(label_ignore)

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Empty batch.")
        max_len = min(self.max_len, max(len(s["input_ids"]) for s in samples))
        bsz = len(samples)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), self.label_ignore, dtype=torch.long)

        for i, s in enumerate(samples):
            ids = list(s["input_ids"])[:max_len]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, : len(ids)] = 1
            label_src = s.get("labels", ids)
            label_src = list(label_src)[:max_len]
            labels[i, : len(label_src)] = torch.tensor(label_src, dtype=torch.long)

        out: dict[str, Any] = {
            "input_ids": input_ids, "attention_mask": attention, "labels": labels,
        }
        # Stateful-architecture document-boundary signal (opt-in chunked dataset).
        # It is a per-step batch-level flag, so it only makes sense at batch_size=1
        # (sequential streaming) — fail loudly rather than silently dropping the
        # other samples' boundary flags and corrupting the recurrent-state resets.
        if any("_doc_boundary" in s for s in samples):
            if len(samples) > 1:
                raise ValueError(
                    "_doc_boundary (chunk_size streaming) requires batch_size=1; "
                    f"got a batch of {len(samples)}. Set data.batch_size: 1."
                )
            out["_doc_boundary"] = bool(samples[0].get("_doc_boundary", False))
        # Preserve ``aux.*`` keys from ArtifactJoinedDataset by stacking
        # the matching tensor across the batch. Hidden-state stacks come in
        # as (L, T, H); we permute the layer axis to the front of the batch
        # so loss code can index per-layer with batch-as-second-axis.
        aux_keys = sorted({k for s in samples for k in s.keys() if k.startswith("aux.")})
        for k in aux_keys:
            tensors = []
            for s in samples:
                v = s.get(k)
                if v is None:
                    continue
                tensors.append(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
            if not tensors:
                continue
            try:
                stacked = torch.stack(tensors, dim=0)
            except RuntimeError:
                continue  # variable-shape aux can't be stacked — caller must reshape
            # For hidden_states_layers: producer stored (L, T, H) per sample;
            # the loss expects (L, B, T, H). torch.stack along dim 0 gives
            # (B, L, T, H), so transpose the first two dims.
            if stacked.dim() == 4 and k.endswith(".hidden_states_layers"):
                stacked = stacked.transpose(0, 1).contiguous()
            elif stacked.dim() == 5 and k.endswith(".attentions_layers"):
                # (B, L, H, T, T) -> (L, B, H, T, T)
                stacked = stacked.transpose(0, 1).contiguous()
            out[k] = stacked
        return out


@register("collator", "preference")
class PreferenceCollator:
    """Right-pad chosen and rejected sequences for offline preference training.

    Expects each sample to have ``chosen_input_ids``, ``chosen_labels``,
    ``rejected_input_ids``, ``rejected_labels``.  Builds attention masks and
    pads/truncates to ``max_len``; ``ignore_index`` fills pad label positions.

    Output keys (all ``(B, T)`` int64):
        chosen_input_ids, chosen_attention_mask, chosen_labels,
        rejected_input_ids, rejected_attention_mask, rejected_labels
    """

    def __init__(self, pad_id: int, max_len: int = 1024,
                 ignore_index: int = -100) -> None:
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.ignore_index = int(ignore_index)

    def _pad_side(
        self, samples: list[Mapping[str, Any]], ids_key: str, labels_key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = min(
            self.max_len, max(len(s[ids_key]) for s in samples)
        )
        bsz = len(samples)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), self.ignore_index, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = list(s[ids_key])[:max_len]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, : len(ids)] = 1
            lbl = list(s[labels_key])[:max_len]
            labels[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
        return input_ids, attention, labels

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Empty batch.")
        c_ids, c_attn, c_lbl = self._pad_side(samples, "chosen_input_ids", "chosen_labels")
        r_ids, r_attn, r_lbl = self._pad_side(samples, "rejected_input_ids", "rejected_labels")
        out: dict[str, torch.Tensor] = {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_attn,
            "chosen_labels": c_lbl,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_attn,
            "rejected_labels": r_lbl,
        }
        # Preserve aux.* keys (e.g. aux.ref.chosen_logprobs) injected by
        # ArtifactJoinedDataset so that DPO/KTO losses can read ref logprobs.
        aux_keys = sorted({k for s in samples for k in s.keys() if k.startswith("aux.")})
        for k in aux_keys:
            tensors = []
            for s in samples:
                v = s.get(k)
                if v is None:
                    continue
                tensors.append(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
            if not tensors:
                continue
            try:
                out[k] = torch.stack(tensors, dim=0)
            except RuntimeError:
                continue  # variable-shape aux — caller must reshape
        return out


__all__ = ["CausalLMCollator", "PreferenceCollator"]
