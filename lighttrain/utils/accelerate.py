"""HF Accelerate wiring for single-GPU AMP.

Why a separate helper:

* Keeps ``cli/_runtime.py`` free of HF-specific construction.
* ``cfg.engine.mixed_precision`` ∈ ``{"no", "fp16", "bf16"}`` maps 1:1 to
  ``Accelerator(mixed_precision=...)``.
* On single GPU we deliberately do **not** call ``accelerator.prepare(...)``
  (which is for distributed device placement); we only use the
  ``autocast`` / ``backward`` / ``clip_grad_norm_`` / ``scaler`` API surface
  to remove the autocast + GradScaler boilerplate from ``StandardUpdateRule``.
"""

from __future__ import annotations

from typing import Any

from ..config._exceptions import ConfigError


def build_accelerator(
    mixed_precision: str = "no",
    *,
    gradient_accumulation_steps: int = 1,
) -> Any | None:
    """Return an ``Accelerator`` instance or None for ``mixed_precision='no'``.

    ``mixed_precision='no'`` short-circuits and returns ``None`` so the
    common case of disabled AMP keeps the update-rule fast-path (raw
    ``loss.backward()`` / ``clip_grad_norm_``). Any other value requires
    ``accelerate`` to be importable; missing the package raises
    :class:`ConfigError` (it is a declared dependency in ``pyproject.toml``).
    """
    mp = (mixed_precision or "no").lower()
    if mp in ("no", "none", "off", "false", "0", ""):
        return None
    if mp not in ("fp16", "bf16"):
        raise ConfigError(
            f"engine.mixed_precision must be one of no|fp16|bf16, got {mixed_precision!r}"
        )

    try:
        from accelerate import Accelerator
    except ImportError as e:  # pragma: no cover — declared dependency
        raise ConfigError(
            "HF accelerate not available — install with `pip install accelerate`."
        ) from e

    return Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
    )


__all__ = ["build_accelerator"]
