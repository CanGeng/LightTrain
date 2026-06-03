"""LoRA adapter — ``@register("model", "lora")``.

Thin wrapper around ``peft.LoraConfig`` + ``peft.get_peft_model``. The
recipe form is::

    model:
      name: lora
      base:                    # nested model spec, recursively resolved
        name: tiny_lm
        d_model: 256
        n_layers: 4
      r: 8
      lora_alpha: 16
      target_modules: [qkv, proj]    # optional; auto-detected if omitted
      lora_dropout: 0.05
      bias: none
      task_type: CAUSAL_LM
      modules_to_save: []            # optional; e.g. ["embed_tokens", "lm_head"]

Checkpoint behavior:

* ``state_dict()`` returns **adapter-only** weights (via
  ``peft.get_peft_model_state_dict``). ``CheckpointManager`` writes them to
  ``model.safetensors`` unchanged — files are small.
* ``load_state_dict(sd, strict=False)`` calls
  ``peft.set_peft_model_state_dict``. Base weights are reconstructed by
  rebuilding the recipe (``setup_run_from_config`` re-resolves the model)
  so resume reads the snapshot YAML to recreate base, then loads adapter.
* The full (base + adapter) state_dict is exposed as ``full_state_dict()``
  for callers that need it (e.g. ``merge_and_unload`` exports).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import register

from ._common import (
    adapter_state_dict,
    auto_target_modules,
    import_peft,
    load_adapter_state_dict,
    resolve_base_model,
)


def _normalize_output(out: Any) -> ModelOutput:
    if isinstance(out, ModelOutput):
        return out
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, dict):
        logits = out.get("logits")
    if logits is None:
        raise RuntimeError(
            f"LoRA inner model returned no `logits` (got {type(out).__name__})"
        )
    hidden_states = getattr(out, "hidden_states", None)
    attentions = getattr(out, "attentions", None)
    return ModelOutput(
        outputs={"logits": logits},
        loss=None,
        hidden_states=tuple(hidden_states) if hidden_states else None,
        attentions=tuple(attentions) if attentions else None,
    )


@register("model", "lora")
class LoRAAdapter(nn.Module):
    """Lighttrain-side LoRA wrapper over ``peft.PeftModel``."""

    def __init__(
        self,
        *,
        base: Mapping[str, Any] | nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: list[str] | str | None = None,
        lora_dropout: float = 0.0,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        modules_to_save: list[str] | None = None,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        peft = import_peft()
        base_model, base_spec = resolve_base_model(base)
        if target_modules is None:
            target_modules = auto_target_modules(base_model)
        # peft.PeftModelForCausalLM expects `prepare_inputs_for_generation` on
        # the base (HF generation interface). Bases that don't expose it
        # (e.g. lighttrain's TinyCausalLM) should fall back to the generic
        # PeftModel, which only needs forward. Lighttrain trains via its own
        # loop so we never call HF generate from inside the wrapper.
        if str(task_type).upper() == "CAUSAL_LM" and not hasattr(
            base_model, "prepare_inputs_for_generation"
        ):
            task_type = None

        config_kwargs: dict[str, Any] = {
            "r": int(r),
            "lora_alpha": int(lora_alpha),
            "target_modules": list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules,
            "lora_dropout": float(lora_dropout),
            "bias": str(bias),
            "task_type": str(task_type) if task_type is not None else None,
            "init_lora_weights": init_lora_weights,
        }
        # use_rslora is new in peft 0.7+, set conditionally to avoid TypeError on old versions.
        try:
            config_kwargs["use_rslora"] = bool(use_rslora)
            config = peft.LoraConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("use_rslora", None)
            config = peft.LoraConfig(**config_kwargs)
        if modules_to_save:
            config.modules_to_save = list(modules_to_save)

        self.inner = peft.get_peft_model(base_model, config)
        # Save provenance for frozen_step + lineage replay.
        self._base_spec: Mapping[str, Any] | None = base_spec
        self._lora_kwargs: dict[str, Any] = config_kwargs.copy()
        if modules_to_save:
            self._lora_kwargs["modules_to_save"] = list(modules_to_save)

    # ---- forward ---------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # noqa: ARG002 — protocol parity
        **kwargs: Any,
    ) -> ModelOutput:
        out = self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return _normalize_output(out)

    # ---- checkpoint (adapter-only) ---------------------------------------

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Adapter-only state_dict."""
        # Strip kwargs PyTorch internals occasionally pass that we don't need.
        # ``destination`` / ``prefix`` / ``keep_vars`` are ignored on purpose;
        # adapter checkpoints are flat and never composed by parent modules.
        _ = args, kwargs
        return adapter_state_dict(self.inner)

    def load_state_dict(  # type: ignore[override]
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = False
    ) -> Any:
        _ = strict  # peft handles missing keys quietly
        load_adapter_state_dict(self.inner, state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])  # type: ignore[attr-defined]

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Escape hatch: returns base + adapter weights (large)."""
        return dict(self.inner.state_dict())

    # ---- PEFT / QLoRA glue exposed to trainer ----------------------------

    def enable_input_require_grads(self) -> None:
        if hasattr(self.inner, "enable_input_require_grads"):
            self.inner.enable_input_require_grads()

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.inner, "gradient_checkpointing_enable"):
            self.inner.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.inner, "gradient_checkpointing_disable"):
            self.inner.gradient_checkpointing_disable()

    def merge_and_unload(self) -> nn.Module:
        """Bake LoRA deltas into the base and return the plain base model.

        Useful for export pipelines that need a single set of merged weights.
        """
        return self.inner.merge_and_unload()

    # ---- introspection ---------------------------------------------------

    def get_base_model(self) -> nn.Module:
        return self.inner.get_base_model()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable_params, all_params)`` — matches peft's helper."""
        trainable = 0
        total = 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return trainable, total


__all__ = ["LoRAAdapter"]
