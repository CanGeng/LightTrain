"""JSONL logger — one record per line, atomic line append."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from lighttrain.registry import register


@register("logger", "jsonl")
class JSONLLogger:
    """Append-only JSONL writer.

    Path defaults to ``<run_dir>/logs/metrics.jsonl`` if ``run_dir`` is given;
    otherwise ``path`` is used directly. The file handle is kept open and
    flushed after every record so a SIGKILL still preserves prior lines.

    Thread-safety: NOT thread-safe. Call only from the main training thread.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        run_dir: str | Path | None = None,
        filename: str = "metrics.jsonl",
    ) -> None:
        if path is None:
            if run_dir is None:
                raise ValueError("JSONLLogger needs `path` or `run_dir`.")
            path = Path(run_dir) / "logs" / filename
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        self._fp.write(json.dumps(record, ensure_ascii=False, default=str))
        self._fp.write("\n")
        self._fp.flush()

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        rec: dict[str, Any] = {"step": int(step), "kind": "scalar"}
        rec.update({k: _coerce(v) for k, v in scalars.items()})
        self._write(rec)

    def log_text(self, text: str, step: int) -> None:
        self._write({"step": int(step), "kind": "text", "text": text})

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self._write({"kind": "artifact", "path": str(path), "name": name})

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:  # pragma: no cover
            pass


def _coerce(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


__all__ = ["JSONLLogger"]
