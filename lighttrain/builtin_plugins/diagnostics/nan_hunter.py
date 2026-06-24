"""NaNHunterCallback.

Critical callback that hooks every named submodule of the model with a
forward hook detecting NaN/Inf in inputs/outputs. On the first hit it:

1. Dumps the offending module's I/O tensors + a decoded view of the
   batch to ``runs/<...>/diagnostics/nan_dumps/<step>/<module>.pt``.
2. Calls :func:`write_nan_repro` to drop a self-contained
   ``repro_nan_<ts>/`` reproduction kit.
3. Raises ``RuntimeError`` so the trainer's top-level exception handler
   packages a crash bundle and Lineage records a ``frozen_step`` node.

The hooks are attached on ``on_train_start`` and removed on
``on_train_end`` / ``on_exception``. CPU-only smoke runs can use this
callback without issue — no CUDA reach.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from lighttrain.observability.diagnostics.nan_repro import write_nan_repro
from lighttrain.registry import register

_log = logging.getLogger(__name__)


@register("callback", "nan_hunter")
class NanHunterCallback:
    """Detect NaN/Inf in any named module's forward I/O."""

    critical: bool = True

    def __init__(
        self,
        *,
        check_inputs: bool = True,
        check_outputs: bool = True,
        raise_on_hit: bool = True,
    ) -> None:
        self.check_inputs = bool(check_inputs)
        self.check_outputs = bool(check_outputs)
        self.raise_on_hit = bool(raise_on_hit)
        self._handles: list[Any] = []
        self._fired = False
        self._step: int = 0
        self._batch: Any = None
        self._run_dir: Path | None = None
        self._model: Any = None

    # ----- lifecycle -------------------------------------------------------

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        model = getattr(ctx, "model", None) if ctx is not None else None
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            return
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        if run_dir is None and trainer is not None:
            run_dir = getattr(trainer, "_run_dir", None)
        self._model = model
        self._run_dir = Path(run_dir) if run_dir is not None else None
        for name, mod in model.named_modules():
            if mod is model:
                continue
            handle = mod.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def on_train_end(self, **_: Any) -> None:
        self._detach()

    def on_exception(self, **_: Any) -> None:
        self._detach()

    def on_step_begin(self, *, step: int = 0, batch: Any = None, **_: Any) -> None:
        self._step = int(step)
        self._batch = batch
        self._fired = False

    # ----- internals -------------------------------------------------------

    def _detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                _log.warning(
                    "nan_hunter: failed to remove a forward hook during detach; leftover hook may add overhead",
                    exc_info=True,
                )
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            if self._fired:
                return
            bad = False
            if self.check_inputs:
                for x in _flatten_tensors(inputs):
                    if not torch.isfinite(x).all():
                        bad = True
                        break
            if not bad and self.check_outputs:
                for x in _flatten_tensors(output):
                    if not torch.isfinite(x).all():
                        bad = True
                        break
            if not bad:
                return
            self._fired = True
            self._handle_hit(name, inputs, output)

        return hook

    def _handle_hit(self, name: str, inputs: Any, output: Any) -> None:
        # Dump module I/O.
        if self._run_dir is not None:
            dump_dir = self._run_dir / "diagnostics" / "nan_dumps" / f"step_{self._step}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe = {
                "inputs": [
                    x.detach().cpu() for x in _flatten_tensors(inputs)
                ],
                "outputs": [
                    x.detach().cpu() for x in _flatten_tensors(output)
                ],
                "module": name,
                "step": int(self._step),
            }
            torch.save(safe, str(dump_dir / f"{_safe_name(name)}.pt"))

            # NaN repro kit.
            if self._model is not None and isinstance(self._batch, dict):
                try:
                    write_nan_repro(
                        self._run_dir,
                        step=self._step,
                        model=self._model,
                        batch=self._batch,
                        exception=RuntimeError(
                            f"NaN/Inf detected in module {name!r} at step {self._step}"
                        ),
                        module_name=name,
                    )
                except Exception:  # noqa: BLE001 — best effort
                    _log.warning(
                        "nan_hunter: write_nan_repro failed for module %r at step %s; repro kit not written",
                        name,
                        self._step,
                        exc_info=True,
                    )
        if self.raise_on_hit:
            raise RuntimeError(
                f"NaN/Inf detected in module {name!r} at step {self._step}"
            )


def _flatten_tensors(obj: Any):
    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _flatten_tensors(x)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from _flatten_tensors(x)
        return
    # other types ignored


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_") or "root"


__all__ = ["NanHunterCallback"]
