"""PEFT adapters.

Lighttrain doesn't reinvent LoRA / IA3 / QLoRA math — we thinly wrap the
HuggingFace ``peft`` package so a recipe says ``model: { name: lora,
base: {...}, r: 8, ... }`` and gets the same registry / checkpoint /
lineage ergonomics as any other lighttrain component.

Public symbols:

* :class:`LoRAAdapter` — ``@register("model", "lora")``
* :class:`IA3Adapter`  — ``@register("model", "ia3")``
* :func:`is_peft_wrapped` / :func:`dump_peft_spec` — used by frozen_step
  ``_infer_model_spec`` to record adapter provenance.

Heavy dependencies (``peft``, ``bitsandbytes``) are imported lazily; the
recipe parses fine without them, but constructing the adapter requires
``pip install -e .[peft]`` (and ``.[quant]`` for QLoRA).
"""

from __future__ import annotations

from ._adalora import AdaLoRAAdapter
from ._common import (
    adapter_state_dict,
    auto_target_modules,
    dump_peft_spec,
    import_peft,
    is_peft_wrapped,
    load_adapter_state_dict,
    resolve_base_model,
)
from ._ia3 import IA3Adapter
from ._lora import LoRAAdapter

__all__ = [
    "AdaLoRAAdapter",
    "LoRAAdapter",
    "IA3Adapter",
    # introspection helpers
    "is_peft_wrapped",
    "dump_peft_spec",
    "adapter_state_dict",
    "load_adapter_state_dict",
    "auto_target_modules",
    "resolve_base_model",
    "import_peft",
]
