"""Exponential moving average over model parameters."""

from __future__ import annotations

from typing import Any

from lighttrain.registry import register


@register("callback", "ema")
class EMACallback:
    """Maintain a shadow copy of params updated by EMA after each optim step.

    On ``on_eval_begin`` the trainer model's params are swapped with the
    shadow; ``on_eval_end`` restores. Operates lazily on first event so the
    callback can be registered without a model handle up-front.
    """

    def __init__(self, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, Any] = {}
        self.backup: dict[str, Any] = {}

    def _params(self, model: Any) -> dict[str, Any]:
        if model is None or not hasattr(model, "named_parameters"):
            return {}
        return {n: p for n, p in model.named_parameters() if p.requires_grad}

    def on_optimizer_step_post(self, *, model: Any = None, **_: Any) -> None:
        params = self._params(model)
        if not params:
            return
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        for n, p in params.items():
            data = p.detach()
            if n not in self.shadow:
                self.shadow[n] = data.clone()
                continue
            with torch.no_grad():
                self.shadow[n].mul_(self.decay).add_(data, alpha=1 - self.decay)

    def on_eval_begin(self, *, model: Any = None, **_: Any) -> None:
        params = self._params(model)
        if not params or not self.shadow:
            return
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            return
        self.backup.clear()
        for n, p in params.items():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def on_eval_end(self, *, model: Any = None, **_: Any) -> None:
        params = self._params(model)
        for n, p in params.items():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()


__all__ = ["EMACallback"]
