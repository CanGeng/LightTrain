"""bitsandbytes 4-/8-bit quantization glue.

We walk a model and swap every ``nn.Linear`` for ``bnb.nn.Linear4bit`` or
``bnb.nn.Linear8bitLt``. ``LayerNorm`` / ``nn.Embedding`` / output-head
``Linear`` (configurable, default ``"lm_head"``) are skipped so the quant
artefacts don't bleed into the loss path.

This is a Linux-only / GPU-only path. On platforms where ``bitsandbytes``
isn't importable we raise ``ImportError`` with a clear "install with
``pip install -e .[quant]``" hint. The CLI ``dry-run`` path doesn't
actually construct the model (it only loads the recipe), so a recipe that
mentions ``qlora`` parses fine on Windows.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def _import_bnb():
    try:
        import bitsandbytes as bnb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "bitsandbytes is required for 4-/8-bit quantization. "
            "Install with `pip install -e .[quant]` (Linux + CUDA only)."
        ) from e
    return bnb


def _should_skip(name: str, skip_patterns: Iterable[str]) -> bool:
    return any(pat in name for pat in skip_patterns)


def _replace_linear(parent: nn.Module, child_name: str, *, bits: int, **kw) -> None:
    """In-place: swap ``parent.<child_name>`` (an nn.Linear) with bnb quant."""
    bnb = _import_bnb()
    old: nn.Linear = getattr(parent, child_name)
    if bits == 4:
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            compute_dtype=kw.get("compute_dtype", None),
            quant_type=kw.get("quant_type", "nf4"),
        )
    elif bits == 8:
        new = bnb.nn.Linear8bitLt(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            has_fp16_weights=kw.get("has_fp16_weights", False),
            threshold=kw.get("threshold", 6.0),
        )
    else:
        raise ValueError(f"bnb_quantize: bits must be 4 or 8, got {bits}")
    # Copy over weights & bias for the initial state (bnb's Linear types
    # handle the actual quantization at .to(cuda) time).
    new.weight = bnb.nn.Params4bit(old.weight.data, requires_grad=False) if bits == 4 else new.weight
    if old.bias is not None:
        new.bias.data = old.bias.data.detach().clone()
    setattr(parent, child_name, new)


def bnb_quantize(
    model: nn.Module,
    *,
    bits: int = 4,
    skip: Iterable[str] = ("lm_head",),
    compute_dtype=None,
    quant_type: str = "nf4",
    has_fp16_weights: bool = False,
    threshold: float = 6.0,
) -> nn.Module:
    """In-place: walk ``model`` and swap every ``nn.Linear`` for the
    bnb quantized equivalent. Returns ``model`` for chaining."""
    _import_bnb()
    # Walk modules; iterate over parents so we can setattr.
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if _should_skip(full_name, skip):
                continue
            _replace_linear(
                parent,
                child_name,
                bits=bits,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                has_fp16_weights=has_fp16_weights,
                threshold=threshold,
            )
    return model


__all__ = ["bnb_quantize"]
