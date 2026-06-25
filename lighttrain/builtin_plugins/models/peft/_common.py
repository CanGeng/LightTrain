"""Shared PEFT plumbing — lazy import, base resolve, adapter-only ckpt.

peft / QLoRA / IA3 / AdaLoRA are thin wrappers around HuggingFace ``peft``.
We do not reimplement low-rank math; we re-export it under our registry
surface so a recipe says ``model: { name: lora, ... }`` and gets the same
ergonomics as every other lighttrain component.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


def import_peft():
    """Lazy import the ``peft`` package; raise a clear hint if missing."""
    try:
        import peft

        return peft
    except ImportError as e:
        raise ImportError(
            "lighttrain PEFT adapters require the `peft` package. "
            "Install with `pip install -e .[peft]` (and `.[quant]` for QLoRA)."
        ) from e


def resolve_base_model(base: Mapping[str, Any] | nn.Module) -> tuple[nn.Module, Mapping[str, Any] | None]:
    """Return ``(base_module, base_spec_or_None)``.

    Accepts either an already-constructed ``nn.Module`` (e.g. from a Python
    test) or a recipe spec ``{name, params}`` / ``{_target_, params}`` that
    is recursively passed to the standard config resolver. Storing the spec
    lets ``dump_peft_spec`` reconstruct the wrapped model later.
    """
    if isinstance(base, nn.Module):
        return base, None
    from lighttrain.config._resolver import resolve as _resolve

    spec = dict(base)
    return _resolve(spec, category="model"), spec


def auto_target_modules(base: nn.Module) -> list[str]:
    """Best-effort default ``target_modules`` for common architectures.

    LoRA needs to know which linear layers to inject into. For
    ``TinyCausalLM`` we target the attention qkv + projection; for HF
    Llama-family we target q/k/v/o; for unknown shapes we fall back to a
    catch-all linear scan that won't blow up but may over-cover.
    """
    cls_name = type(base).__name__
    if cls_name == "TinyCausalLM":
        return ["qkv", "proj"]
    if cls_name == "HFCausalLM":
        # Delegate one level down — the inner is a real HF model.
        inner = getattr(base, "inner", None)
        if inner is not None:
            inner_cls = type(inner).__name__.lower()
            if "llama" in inner_cls or "mistral" in inner_cls or "qwen" in inner_cls:
                return ["q_proj", "k_proj", "v_proj", "o_proj"]
            if "gpt2" in inner_cls or "gptneo" in inner_cls:
                return ["c_attn", "c_proj"]
            if "gptj" in inner_cls:
                return ["q_proj", "k_proj", "v_proj", "out_proj"]
    # Conservative fallback: catch-all linear modules. Caller can override.
    return ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"]


def is_peft_wrapped(model: nn.Module) -> bool:
    """Return True if ``model`` is one of our PEFT adapters or a raw peft
    model. Used by frozen_step ``_infer_model_spec``."""
    cls_name = type(model).__name__
    if cls_name in {"LoRAAdapter", "IA3Adapter", "QLoRAAdapter"}:
        return True
    try:
        import peft

        return isinstance(model, peft.PeftModel)
    except ImportError:
        return False


def dump_peft_spec(model: nn.Module) -> dict[str, Any]:
    """Serialize a peft-wrapped model into a lighttrain model spec.

    Format mirrors a normal recipe model section::

        {
            "name": "lora",
            "params": {
                "base": <base spec>,
                "r": 8, "lora_alpha": 16,
                "target_modules": [...],
                ...
            },
        }

    If the spec for the base model isn't known (model was constructed
    programmatically without a recipe), uses ``_target_`` fallback for the
    base layer — same convention as ``_infer_model_spec``.
    """
    cls_name = type(model).__name__
    if cls_name == "LoRAAdapter":
        kwargs = dict(getattr(model, "_lora_kwargs", {}))
        base_spec = getattr(model, "_base_spec", None) or _fallback_base_spec(
            getattr(model, "inner", None)
        )
        return {
            "name": "lora",
            "params": {"base": base_spec, **kwargs},
        }
    if cls_name == "IA3Adapter":
        kwargs = dict(getattr(model, "_ia3_kwargs", {}))
        base_spec = getattr(model, "_base_spec", None) or _fallback_base_spec(
            getattr(model, "inner", None)
        )
        return {
            "name": "ia3",
            "params": {"base": base_spec, **kwargs},
        }
    if cls_name == "QLoRAAdapter":
        kwargs = dict(getattr(model, "_qlora_kwargs", {}))
        base_spec = getattr(model, "_base_spec", None)
        return {
            "name": "qlora",
            "params": {"base": base_spec, **kwargs},
        }
    # Raw peft.PeftModel — degrade to _target_ on the inner.
    return _fallback_base_spec(model)


def _fallback_base_spec(model: nn.Module | None) -> dict[str, Any]:
    if model is None:
        return {"_target_": "torch.nn:Identity", "params": {}}
    # Walk PeftModel to get the underlying base.
    base = model
    for attr in ("get_base_model", "base_model"):
        next_base = getattr(base, attr, None)
        if callable(next_base):
            base = next_base()
    cls = type(base)
    return {"_target_": f"{cls.__module__}:{cls.__name__}", "params": {}}


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:  # noqa: F821
    """Return adapter-only state dict via peft helpers."""
    peft = import_peft()
    return peft.get_peft_model_state_dict(model)


def load_adapter_state_dict(model: nn.Module, sd: Mapping[str, Any]) -> None:
    peft = import_peft()
    peft.set_peft_model_state_dict(model, dict(sd))


__all__ = [
    "import_peft",
    "resolve_base_model",
    "auto_target_modules",
    "is_peft_wrapped",
    "dump_peft_spec",
    "adapter_state_dict",
    "load_adapter_state_dict",
]
