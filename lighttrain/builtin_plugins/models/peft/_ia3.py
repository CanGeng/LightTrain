"""IA³ adapter — ``@register("model", "ia3")``.

Thin wrapper around ``peft.IA3Config``. IA³ scales activations channel-wise
inside attention + MLP, with no rank-decomposition matmul — adapter size is
even smaller than LoRA but representational capacity is also lower. Same
checkpoint / forward contract as :class:`LoRAAdapter`.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register
from ._common import (
    adapter_state_dict,
    import_peft,
    load_adapter_state_dict,
    resolve_base_model,
)
from ._lora import _normalize_output


def _auto_ia3_targets(base: nn.Module) -> tuple[list[str], list[str]]:
    """Return ``(target_modules, feedforward_modules)`` for IA³."""
    cls_name = type(base).__name__
    if cls_name == "TinyCausalLM":
        # qkv = attention; fc1/fc2 = MLP. feedforward_modules must be a
        # subset of target_modules.
        return ["qkv", "fc2"], ["fc2"]
    if cls_name == "HFCausalLM":
        inner = getattr(base, "inner", None)
        if inner is not None:
            inner_cls = type(inner).__name__.lower()
            if "llama" in inner_cls or "mistral" in inner_cls:
                return (
                    ["k_proj", "v_proj", "down_proj"],
                    ["down_proj"],
                )
    # Conservative fallback.
    return ["key", "value", "dense"], ["dense"]


@register("model", "ia3")
class IA3Adapter(nn.Module):
    """Lighttrain-side IA³ wrapper over ``peft.PeftModel``."""

    def __init__(
        self,
        *,
        base: Mapping[str, Any] | nn.Module,
        target_modules: list[str] | str | None = None,
        feedforward_modules: list[str] | str | None = None,
        task_type: str = "CAUSAL_LM",
        modules_to_save: list[str] | None = None,
        init_ia3_weights: bool = True,
    ) -> None:
        super().__init__()
        peft = import_peft()
        base_model, base_spec = resolve_base_model(base)
        if str(task_type).upper() == "CAUSAL_LM" and not hasattr(
            base_model, "prepare_inputs_for_generation"
        ):
            task_type = None
        if target_modules is None or feedforward_modules is None:
            tm_auto, ff_auto = _auto_ia3_targets(base_model)
            if target_modules is None:
                target_modules = tm_auto
            if feedforward_modules is None:
                feedforward_modules = ff_auto

        config_kwargs: dict[str, Any] = {
            "target_modules": list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules,
            "feedforward_modules": list(feedforward_modules) if isinstance(feedforward_modules, (list, tuple)) else feedforward_modules,
            "task_type": str(task_type) if task_type is not None else None,
            "init_ia3_weights": bool(init_ia3_weights),
        }
        config = peft.IA3Config(**config_kwargs)
        if modules_to_save:
            config.modules_to_save = list(modules_to_save)

        self.inner = peft.get_peft_model(base_model, config)
        self._base_spec: Mapping[str, Any] | None = base_spec
        self._ia3_kwargs: dict[str, Any] = config_kwargs.copy()
        if modules_to_save:
            self._ia3_kwargs["modules_to_save"] = list(modules_to_save)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # noqa: ARG002 — protocol parity
        **kwargs: Any,
    ) -> ModelOutput:
        out = self.inner(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return _normalize_output(out)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
        _ = args, kwargs
        return adapter_state_dict(self.inner)

    def load_state_dict(  # type: ignore[override]
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = False
    ) -> Any:
        _ = strict
        load_adapter_state_dict(self.inner, state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])  # type: ignore[attr-defined]

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.inner.state_dict())

    def enable_input_require_grads(self) -> None:
        if hasattr(self.inner, "enable_input_require_grads"):
            self.inner.enable_input_require_grads()

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.inner, "gradient_checkpointing_enable"):
            self.inner.gradient_checkpointing_enable(**kwargs)

    def get_base_model(self) -> nn.Module:
        return self.inner.get_base_model()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = 0
        total = 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return trainable, total


__all__ = ["IA3Adapter"]
