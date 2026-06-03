"""Advance a model's *internal* EMA target after each optimizer step.

Distinct from :class:`~lighttrain.builtin_plugins.callbacks.builtins.ema.EMACallback` (name
``ema``), which keeps a Polyak parameter *shadow* and swaps it in for eval.
This callback instead calls the model's own ``update_ema()`` so an architecture
that maintains an internal EMA target encoder (e.g. JEPA's ``target_encoder``,
read inside ``forward``) actually advances — otherwise the target stays at its
random init and training has no signal.

Generic duck-typing: a no-op for any model without ``update_ema``. Hooked on
``on_optimizer_step_post`` so it fires once per *real* optimizer step (skipped
steps and gradient-accumulation micro-steps don't trigger it).
"""

from __future__ import annotations

from typing import Any

from lighttrain.registry import register


@register("callback", "target_ema")
class TargetEMACallback:
    """Call ``model.update_ema()`` after each optimizer step, if present."""

    def on_optimizer_step_post(self, *, model: Any = None, ctx: Any = None, **_: Any) -> None:
        if model is None and ctx is not None:
            model = getattr(ctx, "model", None)
        update = getattr(model, "update_ema", None)
        if callable(update):
            update()


__all__ = ["TargetEMACallback"]
