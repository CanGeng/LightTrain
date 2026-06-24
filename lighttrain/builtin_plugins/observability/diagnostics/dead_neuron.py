"""DeadNeuronCallback.

Rolling window over per-channel activation statistics. After every
``every_n_steps`` we compute the *zero ratio* and *variance* of each
captured tensor across the window and dump the result.

We sample activations via forward hooks on a subset of named modules
matched by an optional regex (``module_pattern``); default matches all
Linear / SiLU / GELU outputs. Non-critical — failures swallowed and the
window resets.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from lighttrain.registry import register

_log = logging.getLogger(__name__)


@register("callback", "dead_neuron")
class DeadNeuronCallback:
    """Track per-channel activation statistics in a rolling window."""

    def __init__(
        self,
        *,
        window: int = 100,
        every_n_steps: int = 100,
        module_pattern: str | None = None,
        zero_threshold: float = 1e-6,
    ) -> None:
        self.window = max(1, int(window))
        self.every_n_steps = max(1, int(every_n_steps))
        self.zero_threshold = float(zero_threshold)
        self._regex = re.compile(module_pattern) if module_pattern else None
        self._buf: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[Any] = []
        self._run_dir: Path | None = None
        self._step = 0

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        model = getattr(ctx, "model", None) if ctx is not None else None
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            return
        rd = getattr(ctx, "run_dir", None) if ctx is not None else None
        if rd is None and trainer is not None:
            rd = getattr(trainer, "_run_dir", None)
        self._run_dir = Path(rd) if rd is not None else None
        for name, mod in model.named_modules():
            if mod is model:
                continue
            if self._regex is not None:
                if not self._regex.search(name):
                    continue
            else:
                cls = type(mod).__name__.lower()
                if not any(k in cls for k in ("linear", "silu", "gelu", "relu")):
                    continue
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def on_train_end(self, **_: Any) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                _log.warning(
                    "dead_neuron: failed to remove an activation forward hook; leftover hook may add overhead",
                    exc_info=True,
                )
        self._handles.clear()

    def on_step_end(self, *, step: int = 0, **_: Any) -> None:
        self._step = int(step)
        if step <= 0 or step % self.every_n_steps != 0:
            return
        if self._run_dir is None:
            return
        report: dict[str, dict[str, float]] = {}
        for name, samples in list(self._buf.items()):
            if not samples:
                continue
            # Flatten the window over batch+positions, keep last channel dim.
            stacked = torch.cat(
                [s.reshape(-1, s.shape[-1]) for s in samples], dim=0
            )
            zero_ratio = (stacked.abs() < self.zero_threshold).float().mean(dim=0)
            var = stacked.var(dim=0, unbiased=False)
            report[name] = {
                "zero_ratio_mean": float(zero_ratio.mean().item()),
                "zero_ratio_max": float(zero_ratio.max().item()),
                "var_mean": float(var.mean().item()),
                "var_min": float(var.min().item()),
                "n_channels": int(stacked.shape[-1]),
            }
        out_dir = self._run_dir / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"dead_neurons_{int(step)}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._buf.clear()

    def _make_hook(self, name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            t = output
            if isinstance(t, (list, tuple)) and t:
                t = t[0]
            if not isinstance(t, torch.Tensor):
                return
            buf = self._buf[name]
            buf.append(t.detach().to("cpu", copy=True))
            if len(buf) > self.window:
                buf.pop(0)

        return hook


__all__ = ["DeadNeuronCallback"]
