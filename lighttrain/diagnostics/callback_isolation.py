"""CallbackIsolationSink.

The EventBus already does quarantine + critical-callback re-raise.
This adds the **persistence + report** layer:

* Sink:  write every isolated exception to
         ``runs/<...>/diagnostics/callback_failures.jsonl`` with
         ``{ts, step, callback, event, exc_type, traceback}``.
* Report: aggregate the JSONL into ``callback_report.md`` at
          ``on_train_end`` (or whenever ``write_callback_report`` is
          called by ``diagnostics/index_page``).

This callback is itself non-critical — if writing the JSONL fails it's
silent (the alternative would mask the original callback failure).
"""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


class CallbackIsolationSink:
    """Hook into the EventBus's ``on_error`` callback to persist failures.

    Installed in two complementary ways:

    1. Direct hook — :func:`install` wires ``bus._on_error`` so every
       non-critical callback exception goes to disk.
    2. Lifecycle observer — also implements ``on_train_end`` so the
       report regenerates even when ``install`` wasn't called.
    """

    def __init__(self, *, max_recent: int = 100) -> None:
        self.max_recent = int(max_recent)
        self._run_dir: Path | None = None
        self._bus: Any = None
        self._recent: list[dict[str, Any]] = []
        self._installed = False
        self._step = 0

    def on_train_start(self, *, trainer: Any = None, ctx: Any = None, **_: Any) -> None:
        rd = getattr(ctx, "run_dir", None) if ctx is not None else None
        if rd is None and trainer is not None:
            rd = getattr(trainer, "_run_dir", None)
        self._run_dir = Path(rd) if rd is not None else None
        bus = getattr(trainer, "bus", None) if trainer is not None else None
        if bus is not None:
            self.install(bus)

    def on_step_begin(self, *, step: int = 0, **_: Any) -> None:
        self._step = int(step)

    def on_train_end(self, *, trainer: Any = None, **_: Any) -> None:
        if self._run_dir is None:
            return
        try:
            write_callback_report(
                self._run_dir, bus=self._bus or getattr(trainer, "bus", None)
            )
        except Exception:  # noqa: BLE001
            pass

    def install(self, bus: Any) -> None:
        """Replace ``bus._on_error`` with a sink that persists to disk."""
        if self._installed:
            return
        self._bus = bus
        original = getattr(bus, "_on_error", None)

        def _sink(event: str, cb: Any, exc: BaseException) -> None:
            self._record(event, cb, exc)
            if callable(original):
                try:
                    original(event, cb, exc)
                except Exception:  # noqa: BLE001
                    pass

        try:
            bus._on_error = _sink
            self._installed = True
        except Exception:  # noqa: BLE001
            pass

    def _record(self, event: str, cb: Any, exc: BaseException) -> None:
        entry = {
            "ts": time.time(),
            "step": int(self._step),
            "callback": type(cb).__name__,
            "event": str(event),
            "exc_type": type(exc).__name__,
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-4000:],
        }
        self._recent.append(entry)
        if len(self._recent) > self.max_recent:
            self._recent.pop(0)
        if self._run_dir is None:
            return
        out = self._run_dir / "diagnostics"
        try:
            out.mkdir(parents=True, exist_ok=True)
            with (out / "callback_failures.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001
            pass


def write_callback_report(run_dir: Path, *, bus: Any | None = None) -> Path | None:
    """Aggregate ``callback_failures.jsonl`` into ``callback_report.md``.

    Returns the path written (or ``None`` if the input file is missing).
    Safe to call multiple times — the report is regenerated each call.
    """
    run_dir = Path(run_dir)
    src = run_dir / "diagnostics" / "callback_failures.jsonl"
    if not src.exists():
        return None
    lines: list[dict[str, Any]] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except Exception:  # noqa: BLE001
            continue
    by_cb: dict[str, int] = defaultdict(int)
    by_event: dict[str, int] = defaultdict(int)
    for e in lines:
        by_cb[e.get("callback", "<unknown>")] += 1
        by_event[e.get("event", "<unknown>")] += 1
    quarantined: list[str] = []
    if bus is not None and hasattr(bus, "quarantined"):
        try:
            quarantined = list(bus.quarantined)
        except Exception:  # noqa: BLE001
            quarantined = []
    out_md = [
        f"# Callback failure report",
        "",
        f"- Total isolated failures: **{len(lines)}**",
        f"- Currently quarantined: {', '.join(quarantined) or '_none_'}",
        "",
        "## By callback",
        "",
    ]
    for k, v in sorted(by_cb.items(), key=lambda kv: kv[1], reverse=True):
        out_md.append(f"- `{k}` :: {v}")
    out_md += ["", "## By event", ""]
    for k, v in sorted(by_event.items(), key=lambda kv: kv[1], reverse=True):
        out_md.append(f"- `{k}` :: {v}")
    if lines:
        out_md += ["", "## Last 5 failures", ""]
        for e in lines[-5:]:
            out_md.append(
                f"- step={e.get('step', '?')}  cb=`{e.get('callback')}`  "
                f"event=`{e.get('event')}`  err=`{e.get('exc_type')}`"
            )
    out_path = run_dir / "diagnostics" / "callback_report.md"
    out_path.write_text("\n".join(out_md), encoding="utf-8")
    return out_path


__all__ = ["CallbackIsolationSink", "write_callback_report"]
