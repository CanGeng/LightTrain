"""Core losses.

Each loss is a callable conforming to :class:`LossFnProtocol`:
``__call__(model_output, batch, ctx) -> dict``. The dict must include a
``"loss"`` tensor; auxiliary scalars/tensors may be returned for logging
and downstream callbacks (z_loss / aux_loss / kl / ...).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


def _logits(model_output: ModelOutput | Mapping[str, Any]) -> torch.Tensor:
    if isinstance(model_output, ModelOutput):
        if "logits" not in model_output.outputs:
            raise KeyError("ModelOutput.outputs missing 'logits'.")
        return model_output.outputs["logits"]
    if isinstance(model_output, Mapping):
        if "logits" in model_output:
            return model_output["logits"]
    raise TypeError(f"Cannot extract logits from {type(model_output).__name__}.")


def _labels(batch: Mapping[str, Any]) -> torch.Tensor:
    if "labels" not in batch:
        raise KeyError("Batch missing 'labels'.")
    return batch["labels"]


@register("loss", "cross_entropy")
@register("loss", "ce")
class CrossEntropyLoss:
    """Causal-LM next-token cross-entropy.

    Expects ``logits`` of shape ``(B, T, V)`` and ``labels`` of shape
    ``(B, T)`` mirroring ``input_ids`` (padding marked with ``-100``).
    The shift ``logits[:, :-1, :]`` vs ``labels[:, 1:]`` is performed here,
    so the collator does not need to anticipate the loss target.
    """

    # Paradigm tag so ``LossOnlyObjective`` inherits a meaningful loss_family
    # (not "generic") when a recipe writes ``loss: cross_entropy`` explicitly.
    loss_family: str = "next_token"

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        logits = _logits(model_output)
        labels = _labels(batch)
        if logits.dim() >= 2 and labels.dim() >= 1 and logits.size(-2) == labels.size(-1):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1).long(),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return {"loss": loss}


@register("loss", "mlm")
class MaskedLMLoss:
    """Masked-LM cross-entropy. Same math as CE but kept distinct for clarity."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = int(ignore_index)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        logits = _logits(model_output)
        labels = _labels(batch)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=self.ignore_index,
        )
        return {"loss": loss}


@register("loss", "z_loss")
class ZLoss:
    """Z-loss regularizer (penalizes ``log Z(x)^2``). Cheap MoE-style aux loss."""

    def __init__(self, weight: float = 1e-4) -> None:
        self.weight = float(weight)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],  # noqa: ARG002
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        logits = _logits(model_output)
        log_z = torch.logsumexp(logits, dim=-1)
        loss = self.weight * (log_z**2).mean()
        return {"loss": loss}


@register("loss", "composite")
class CompositeLoss:
    """Weighted sum of N child losses, registered by short name or factory.

    ``children`` is a list of ``{name, weight, params?}`` dicts; each entry is
    resolved via the registry. The composite returns ``{"loss": total,
    "components": {name: float}}``.
    """

    def __init__(self, children: list[dict[str, Any]]) -> None:
        from lighttrain.config._resolver import resolve as _resolve

        if not children:
            raise ValueError("CompositeLoss requires at least one child.")
        self.children = []
        for entry in children:
            if "name" not in entry and "_target_" not in entry:
                raise ValueError(f"Composite child needs `name` or `_target_`: {entry}")
            weight = float(entry.pop("weight", 1.0))
            obj = _resolve(entry, category="loss")
            self.children.append((weight, obj))

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,
    ) -> dict[str, Any]:
        total: torch.Tensor | None = None
        components: dict[str, float] = {}
        for weight, child in self.children:
            sub = child(model_output, batch, ctx)
            value = sub["loss"] * weight
            total = value if total is None else total + value
            components[type(child).__name__] = float(sub["loss"].detach())
        assert total is not None
        return {"loss": total, "components": components}


__all__ = [
    "CompositeLoss",
    "CrossEntropyLoss",
    "MaskedLMLoss",
    "ZLoss",
]
