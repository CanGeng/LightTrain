"""Cross-domain CLI helpers shared by more than one command module.

Domain-specific helpers live next to their command (e.g. ``_export_primary_model``
in ``commands/artifacts.py``, ``_open_lineage`` in ``commands/lineage.py``).

NOTE: ``_app`` re-exports ``_flatten_patch_to_overrides`` from here — tests import
it as ``from lighttrain.cli._app import _flatten_patch_to_overrides`` (that private
import path is part of the contract).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from lighttrain.cli._context import console

_log = logging.getLogger(__name__)


def _todo(milestone: str, what: str = "") -> None:
    """Emit a friendly not-yet-implemented message and exit non-zero.

    TODO(P3): currently unused (no caller in the repo) — candidate for removal.
    """
    msg = f"[yellow]not yet implemented ({milestone})[/]"
    if what:
        msg = f"{msg} — {what}"
    console.print(msg)
    raise typer.Exit(code=2)


def _flatten_patch_to_overrides(patch: object, prefix: str = "") -> list[str]:
    """Turn a nested dict from ``--apply-degrade patch.yaml`` into
    ``++a.b.c=value`` OmegaConf overrides.

    Strings, ints, floats, bools, and None map to the literal yaml repr.
    Lists are passed through ``yaml.safe_dump`` so OmegaConf parses them
    as sequences.
    """
    out: list[str] = []
    if not isinstance(patch, dict):
        return out
    for k, v in patch.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_patch_to_overrides(v, key))
        elif isinstance(v, (list, tuple)):
            try:
                import yaml as _yaml

                # Flow style is required: _parse_override_value only dispatches
                # to YAML for inputs starting with ``[ { ' "``, so block-style
                # sequences would otherwise be stored as a multi-line string.
                out.append(
                    f"++{key}={_yaml.safe_dump(list(v), default_flow_style=True).strip()}"
                )
            except Exception:  # noqa: BLE001
                _log.warning(
                    "cli helpers: YAML flow-dump of override %r failed; "
                    "falling back to repr() encoding",
                    key,
                    exc_info=True,
                )
                out.append(f"++{key}={v!r}")
        elif v is None:
            out.append(f"++{key}=null")
        else:
            out.append(f"++{key}={v}")
    return out


def _final_loss_from_run(run_dir: Path) -> float | None:
    """Return the last logged ``loss`` from ``<run_dir>/logs/metrics.jsonl``."""
    import json

    metrics = Path(run_dir) / "logs" / "metrics.jsonl"
    if not metrics.exists():
        return None
    last: float | None = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in row:
            last = float(row["loss"])
    return last


def _eval_perplexity(trainer: object, max_batches: int) -> float | None:
    """Perplexity on the val loader, falling back to the train loader.

    Mirrors ``eval_cmd``'s perplexity path so ``train --eval`` and
    ``lighttrain eval`` agree. Returns ``None`` if no loader / eval fails.
    """
    from lighttrain.eval.metrics import perplexity

    model = getattr(trainer, "model", None)
    data_module = getattr(trainer, "data_module", None)
    if model is None or data_module is None:
        return None
    loader = None
    if hasattr(data_module, "val_loader"):
        loader = data_module.val_loader()
    if loader is None and hasattr(data_module, "train_loader"):
        loader = data_module.train_loader()
    if loader is None:
        return None
    mb = max_batches if max_batches > 0 else None
    try:
        return perplexity(model, loader, device=getattr(trainer, "device", None), max_batches=mb)
    except Exception as exc:  # noqa: BLE001
        _log.warning("cli helpers: perplexity eval failed; returning None", exc_info=True)
        console.print(f"[yellow]perplexity eval failed:[/] {exc}")
        return None


def _append_run_summary(path: Path, row: dict) -> None:
    """Append ``row`` to a JSON-list summary at ``path``, replacing any existing
    entry with the same ``exp`` so a multi-variant shell loop accumulates one
    row per variant. Atomic (tmp + replace)."""
    import json
    import os

    rows: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [r for r in loaded if not (isinstance(r, dict) and r.get("exp") == row.get("exp"))]
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _fmt_metric(v: Any) -> str:
    """Compact metric value rendering: round floats, pass through the rest."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
