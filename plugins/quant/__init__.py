"""Quantization plug-in.

Importing this package registers:

* ``@register("model", "qlora")`` — :class:`QLoRAAdapter` (bnb 4-bit base +
  LoRA delta).

…and exposes:

* :func:`bnb_quantize` — in-place 4/8-bit swap helper.

Linux + CUDA only. On Windows the registry entry exists but constructing
the model raises a friendly install error. See ``docs/M5_INTERFACES.md``
for the WSL acceptance recipe (R13).
"""

from __future__ import annotations

from ._bnb import bnb_quantize
from ._qlora import QLoRAAdapter

__all__ = ["QLoRAAdapter", "bnb_quantize"]
