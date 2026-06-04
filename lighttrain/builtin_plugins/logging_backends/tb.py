"""TensorBoard backend — wraps ``torch.utils.tensorboard.SummaryWriter``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lighttrain.registry import register


@register("logger", "tensorboard")
@register("logger", "tb")
class TensorBoardLogger:
    """Write ``events.out.tfevents.*`` into ``<run_dir>/logs/`` (or ``log_dir``)."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        run_dir: str | Path | None = None,
    ) -> None:
        if log_dir is None:
            if run_dir is None:
                raise ValueError("TensorBoardLogger needs `log_dir` or `run_dir`.")
            log_dir = Path(run_dir) / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-import so users without TB installed at import time don't fail.
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        for k, v in scalars.items():
            try:
                self._writer.add_scalar(k, float(v), int(step))
            except (TypeError, ValueError):
                continue

    def log_text(self, text: str, step: int) -> None:
        self._writer.add_text("text", text, int(step))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self._writer.add_text("artifact", f"{name or ''} -> {path}")

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        except Exception:  # pragma: no cover  # noqa: BLE001
            pass


__all__ = ["TensorBoardLogger"]
