"""SamplePreviewCallback.

Dumps decoded text of the first N batches to
``runs/<...>/diagnostics/sample_preview/`` so the user can sanity-check
tokenization / chat template / label masking before the run goes far.
CPU-only safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lighttrain.registry import register


@register("callback", "sample_preview")
class SamplePreviewCallback:
    """Decode the first N batches and dump them to disk."""

    def __init__(
        self,
        *,
        max_batches: int = 3,
        max_chars: int = 1024,
    ) -> None:
        self.max_batches = max(1, int(max_batches))
        self.max_chars = max(64, int(max_chars))
        self._tokenizer: Any = None
        self._run_dir: Path | None = None
        self._dumped = 0

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        rd = getattr(ctx, "run_dir", None) if ctx is not None else None
        if rd is None and trainer is not None:
            rd = getattr(trainer, "_run_dir", None)
        self._run_dir = Path(rd) if rd is not None else None
        if trainer is not None:
            dm = getattr(trainer, "data_module", None)
            if dm is not None:
                self._tokenizer = getattr(dm, "tokenizer", None)

    def on_train_batch_start(self, *, batch: Any = None, step: int = 0, **_: Any) -> None:
        if self._dumped >= self.max_batches or self._run_dir is None:
            return
        if not isinstance(batch, dict):
            return
        ids = batch.get("input_ids")
        if not isinstance(ids, torch.Tensor):
            return
        out_dir = self._run_dir / "diagnostics" / "sample_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [f"# step={step}  batch_size={ids.shape[0]}"]
        for i in range(ids.shape[0]):
            row = ids[i].tolist()
            decoded = ""
            if self._tokenizer is not None and hasattr(self._tokenizer, "decode"):
                try:
                    decoded = str(self._tokenizer.decode(row))
                except Exception:  # noqa: BLE001
                    decoded = "<decode error>"
            label_kept = ""
            labels = batch.get("labels")
            if isinstance(labels, torch.Tensor) and labels.shape == ids.shape:
                kept = int((labels[i] != -100).sum().item())
                label_kept = f"  kept={kept}/{labels.shape[1]}"
            lines.append(f"## sample[{i}] len={len(row)}{label_kept}")
            lines.append(decoded[: self.max_chars])
            lines.append("")
        (out_dir / f"batch_{self._dumped:03d}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        self._dumped += 1


__all__ = ["SamplePreviewCallback"]
