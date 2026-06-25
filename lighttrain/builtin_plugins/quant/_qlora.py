"""QLoRAAdapter — bnb 4-bit base + LoRA delta.

``model: { name: qlora }`` is sugar for the two-step recipe:

1. construct the ``base`` model spec (typically ``hf_causal``);
2. swap every ``nn.Linear`` for ``bnb.nn.Linear4bit``;
3. wrap with ``peft.LoraConfig`` (target_modules adapter friendly to bnb).

The wrapper inherits ``LoRAAdapter``'s adapter-only checkpoint contract,
so on-disk size is the same as plain LoRA (only the deltas).

Linux + CUDA + bitsandbytes only — Windows raises a friendly install error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from lighttrain.registry import register

from ..layer_offload._adapters import (
    register_layered_view,  # noqa: F401
)
from ._bnb import bnb_quantize


@register("model", "qlora")
class QLoRAAdapter:
    """QLoRA: bnb 4-bit base + LoRA on top.

    Returns a ``LoRAAdapter`` instance (so downstream code sees the same
    state_dict / param_groups contract). The wrapper class only exists to
    keep the recipe schema short.

    Config form::

        model:
          name: qlora
          bits: 4
          base:
            name: hf_causal
            pretrained: TinyLlama/TinyLlama-1.1B-Chat-v1.0
            dtype: bfloat16
          lora:
            r: 16
            lora_alpha: 32
            target_modules: [q_proj, k_proj, v_proj, o_proj]
            lora_dropout: 0.05
    """

    def __new__(
        cls,
        *,
        base: Mapping[str, Any] | nn.Module,
        bits: int = 4,
        lora: Mapping[str, Any] | None = None,
        skip: list[str] | None = None,
        compute_dtype: Any = None,
        quant_type: str = "nf4",
    ):
        from lighttrain.builtin_plugins.models.peft import LoRAAdapter
        from lighttrain.builtin_plugins.models.peft._common import resolve_base_model

        base_module, base_spec = resolve_base_model(base)
        bnb_quantize(
            base_module,
            bits=int(bits),
            skip=tuple(skip or ("lm_head",)),
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )
        # Hand the quantized base to LoRAAdapter as a constructed nn.Module
        # so it skips the recursive resolve (already done).
        lora_kwargs = dict(lora or {})
        wrapped = LoRAAdapter(base=base_module, **lora_kwargs)
        # Tag the wrapper so dump_peft_spec records the QLoRA provenance.
        wrapped._base_spec = base_spec
        wrapped._qlora_kwargs = {  # type: ignore[assignment]
            "bits": int(bits),
            "lora": lora_kwargs,
            "skip": list(skip or ("lm_head",)),
            "quant_type": quant_type,
        }
        # Force the type-name path used by dump_peft_spec.
        wrapped.__class__ = type(
            "QLoRAAdapter", (type(wrapped),),
            {"__module__": cls.__module__}
        )
        return wrapped


__all__ = ["QLoRAAdapter"]
