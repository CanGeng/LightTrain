"""Checkpoint fork.

``fork()`` copies (or symlinks) an existing checkpoint into a fresh run
directory wired to a new config, then records a ``fork_of`` lineage edge
in the parent run's SQLite store.

Typical workflow::

    from lighttrain.lab.fork import fork
    from lighttrain.config import load_config

    cfg = load_config("recipes/pretrain_causal.yaml", ["++optim.lr=1e-4"])
    report = fork(Path("runs/my_exp/run_001/checkpoints/step_1000"), cfg)
    # → runs/my_fork_exp/20250524-…/
    #   fork_meta.json  ← parent provenance
    #   checkpoints/step_1000/  ← copied weights

The new run can then be resumed with::

    lighttrain resume --run <report.new_run_dir>
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ForkReport:
    new_run_dir: Path
    parent_checkpoint: Path
    parent_run_dir: Path | None
    lineage_edge_recorded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_parent_run_dir(checkpoint_path: Path) -> Path | None:
    """Heuristically locate the run dir that owns a checkpoint.

    Checkpoint layout::

        <run_dir>/checkpoints/step_<N>/model.safetensors …
        <run_dir>/checkpoints/last/    (symlink or dir)
    """
    p = checkpoint_path.resolve()
    # Walk up to find a directory that contains both 'checkpoints/' and 'env.json'
    for candidate in [p, p.parent, p.parent.parent, p.parent.parent.parent]:
        if (candidate / "checkpoints").is_dir() and (candidate / "env.json").exists():
            return candidate
        if candidate / "lineage.sqlite" in candidate.iterdir() if candidate.is_dir() else []:
            return candidate
    # Fallback: assume structure is <run_dir>/checkpoints/<step>/
    if p.parent.name == "checkpoints" or p.parent.parent.name == "checkpoints":
        return p.parent.parent if p.parent.name != "checkpoints" else p.parent.parent
    return None


def _try_detect_parent_run_dir(checkpoint_path: Path) -> Path | None:
    p = checkpoint_path.resolve()
    for candidate in [p.parent.parent, p.parent.parent.parent]:
        if not candidate.is_dir():
            continue
        children = {c.name for c in candidate.iterdir()}
        if "checkpoints" in children and "env.json" in children:
            return candidate
        if "checkpoints" in children and "lineage.sqlite" in children:
            return candidate
    return None


def _write_fork_meta(
    new_run_dir: Path,
    parent_checkpoint: Path,
    parent_run_dir: Path | None,
) -> None:
    meta = {
        "fork_of_checkpoint": str(parent_checkpoint),
        "fork_of_run_dir": str(parent_run_dir) if parent_run_dir else None,
        "forked_at_ts": time.time(),
    }
    (new_run_dir / "fork_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _copy_checkpoint(
    src: Path,
    new_run_dir: Path,
    *,
    symlink: bool = False,
) -> Path:
    """Copy (or symlink) *src* checkpoint dir into ``new_run_dir/checkpoints/``."""
    ckpt_root = new_run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    dst = ckpt_root / src.name
    if dst.exists() or dst.is_symlink():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copytree(src, dst)
    # Mirror last.json pointer
    last_json = ckpt_root / "last.json"
    last_json.write_text(
        json.dumps({"target": src.name}), encoding="utf-8"
    )
    return dst


def _record_lineage_edge(
    parent_run_dir: Path,
    parent_checkpoint: Path,
    new_run_dir: Path,
    forked_at_step: int | None,
) -> bool:
    """Add a ``fork_of`` edge in the parent run's lineage store.

    Returns *True* if the edge was recorded, *False* if the store was
    unavailable (non-fatal — lineage is a soft dependency).
    """
    sqlite_path = parent_run_dir / "lineage.sqlite"
    if not sqlite_path.exists():
        return False
    try:
        from ..lineage.store import LineageStore

        with LineageStore(sqlite_path) as store:
            # Register the parent checkpoint node (upsert so we don't duplicate)
            parent_node_id = store.upsert_node(
                kind="checkpoint",
                name=str(parent_checkpoint),
                version=None,
                run_id=parent_run_dir.name,
                step=forked_at_step,
            )
            # Register the child run node
            child_node_id = store.upsert_node(
                kind="run",
                name=str(new_run_dir),
                version=None,
                run_id=new_run_dir.name,
            )
            store.add_edge(
                child_node_id,
                parent_node_id,
                "fork_of",
                payload={
                    "parent_run_dir": str(parent_run_dir),
                    "parent_checkpoint": str(parent_checkpoint),
                    "new_run_dir": str(new_run_dir),
                    "forked_at_step": forked_at_step,
                },
            )
        return True
    except Exception:  # noqa: BLE001
        _log.warning(
            "lab.fork: failed to record fork_of lineage edge in %s; "
            "reporting edge as not recorded",
            sqlite_path,
            exc_info=True,
        )
        return False


def _parse_step_from_ckpt(checkpoint_path: Path) -> int | None:
    import re

    m = re.search(r"step[_-](\d+)", checkpoint_path.name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fork(
    from_checkpoint: Path,
    new_config: Mapping[str, Any],
    *,
    symlink: bool = False,
    run_dir: Path | None = None,
) -> ForkReport:
    """Fork *from_checkpoint* into a new run directory.

    Args:
        from_checkpoint: Path to the checkpoint directory to fork
            (e.g. ``runs/exp/run_001/checkpoints/step_1000``).
        new_config: Mapping (e.g. a loaded :class:`~lighttrain.config.RootConfig`
            or plain dict) describing the new run's configuration.
        symlink: If *True*, symlink the checkpoint instead of copying.
            Saves disk space but the parent run must not be deleted.
        run_dir: Override the destination run directory.  If *None*, a new
            directory is created via :func:`~lighttrain.utils.run_dir.make_run_dir`
            using the config's ``run_root`` / ``exp`` fields.

    Returns:
        :class:`ForkReport` with the new run directory and lineage status.
    """
    from_checkpoint = Path(from_checkpoint).resolve()
    if not from_checkpoint.exists():
        raise FileNotFoundError(f"fork: checkpoint not found: {from_checkpoint}")

    # Determine run_dir for the fork
    if run_dir is not None:
        new_run_dir = Path(run_dir)
        new_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        cfg_dict: dict[str, Any]
        if hasattr(new_config, "model_dump"):
            cfg_dict = new_config.model_dump()
        elif isinstance(new_config, Mapping):
            cfg_dict = dict(new_config)
        else:
            cfg_dict = {}

        try:
            import yaml as _yaml

            from ..utils.run_dir import make_run_dir

            resolved_yaml = _yaml.safe_dump(cfg_dict) if cfg_dict else ""
            new_run_dir = make_run_dir(
                root=cfg_dict.get("run_root", "runs"),
                exp=cfg_dict.get("exp", "fork"),
                slug="fork",
                resolved_yaml=resolved_yaml,
            )
        except Exception:  # pragma: no cover — graceful fallback  # noqa: BLE001
            _log.warning(
                "lab.fork: make_run_dir failed; falling back to a temp directory "
                "for the forked run",
                exc_info=True,
            )
            import tempfile

            new_run_dir = Path(tempfile.mkdtemp(prefix="lighttrain_fork_"))

    parent_run_dir = _try_detect_parent_run_dir(from_checkpoint)
    _copy_checkpoint(from_checkpoint, new_run_dir, symlink=symlink)
    _write_fork_meta(new_run_dir, from_checkpoint, parent_run_dir)

    # Persist the new config
    try:
        import yaml as _yaml

        cfg_dict = (
            new_config.model_dump()
            if hasattr(new_config, "model_dump")
            else dict(new_config)
        )
        (new_run_dir / "config.yaml").write_text(
            _yaml.safe_dump(cfg_dict), encoding="utf-8"
        )
    except Exception:  # pragma: no cover  # noqa: BLE001
        _log.warning(
            "lab.fork: failed to persist config.yaml into %s; "
            "the forked run will lack a config snapshot",
            new_run_dir,
            exc_info=True,
        )

    forked_at_step = _parse_step_from_ckpt(from_checkpoint)
    lineage_ok = False
    if parent_run_dir is not None:
        lineage_ok = _record_lineage_edge(
            parent_run_dir, from_checkpoint, new_run_dir, forked_at_step
        )

    return ForkReport(
        new_run_dir=new_run_dir,
        parent_checkpoint=from_checkpoint,
        parent_run_dir=parent_run_dir,
        lineage_edge_recorded=lineage_ok,
    )


__all__ = ["fork", "ForkReport"]
