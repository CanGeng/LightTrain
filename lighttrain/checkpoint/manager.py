"""CheckpointManager — atomic write protocol.

Layout::

    <run_dir>/checkpoints/
        step_<n>/
            model.safetensors
            optimizer.pt
            scheduler.pt
            rng.pt
            manifest.json     (last write — presence-marker)
        last.json             { "target": "step_<n>" }
        best.json             { "target": "step_<n>", "metric": ..., "value": ... }

On Linux/macOS we additionally try to ``os.symlink`` ``last/`` and ``best/``
pointing at ``step_<n>``; on Windows the JSON file is the source of truth
(``symlink_to`` requires admin or developer mode).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch


_STEP_RE = re.compile(r"^step_(\d+)$")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _torch_save_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def _save_safetensors(state_dict: Mapping[str, torch.Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file

        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file({k: v.detach().cpu().clone() for k, v in state_dict.items()},
                  str(tmp))
        os.replace(tmp, path)
    except ImportError:  # pragma: no cover — safetensors is a hard dep
        _torch_save_atomic(dict(state_dict), path.with_suffix(".pt"))


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if path.exists():
        from safetensors.torch import load_file

        return load_file(str(path))
    pt = path.with_suffix(".pt")
    if pt.exists():
        return torch.load(str(pt), map_location="cpu")
    raise FileNotFoundError(f"No model weights at {path} or {pt}")


class CheckpointManager:
    """Atomic step / last / best management."""

    def __init__(self, run_dir: str | Path, *, keep_last_n: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = int(keep_last_n)

    # ---- write paths -------------------------------------------------------

    def save(
        self,
        step: int,
        state: Mapping[str, Any],
        *,
        kind: str = "step",
        extras: Mapping[str, Any] | None = None,
        parallel_ctx: Any | None = None,
    ) -> Path | None:
        """Write a checkpoint atomically.

        ``state`` may include keys ``model`` (state_dict), ``optimizer``,
        ``scheduler``, ``rng``, ``trainer``. Manifest is the LAST file
        written, so a partial dir without ``manifest.json`` is recognized
        as incomplete and skipped on resume.

        In distributed runs, pass ``parallel_ctx`` so that only rank-0
        writes to disk.  Non-rank-0 processes return ``None`` immediately.
        """
        if parallel_ctx is not None and not parallel_ctx.is_main_process:
            return None

        target = self.ckpt_dir / f"step_{int(step)}"
        target.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "step": int(step),
            "kind": kind,
            "files": {},
            "extras": dict(extras or {}),
        }

        if "model" in state and state["model"] is not None:
            path = target / "model.safetensors"
            _save_safetensors(state["model"], path)
            manifest["files"]["model"] = path.name

        for key, name in (
            ("optimizer", "optimizer.pt"),
            ("scheduler", "scheduler.pt"),
            ("rng", "rng.pt"),
            ("trainer", "trainer.pt"),
            # data_module state (sampler position, etc.) must land on disk
            # so functional-resume can restore it.
            ("data_module", "data_module.pt"),
        ):
            if key in state and state[key] is not None:
                path = target / name
                _torch_save_atomic(state[key], path)
                manifest["files"][key] = name

        _atomic_write_text(target / "manifest.json", json.dumps(manifest, indent=2))

        # Rotate / pointers.
        if kind == "step":
            self._update_pointer("last", target)
            self._prune()
        elif kind == "best":
            self._update_pointer("best", target, metric_extras=extras)

        return target

    # ---- read paths --------------------------------------------------------

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint dir into a state dict (CPU tensors)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint {p} does not exist.")
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Incomplete checkpoint at {p}: missing manifest.json."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        out: dict[str, Any] = {"step": int(manifest.get("step", 0)),
                               "manifest": manifest}
        if "model" in manifest["files"]:
            out["model"] = _load_safetensors(p / manifest["files"]["model"])
        for key, name in (
            ("optimizer", "optimizer.pt"),
            ("scheduler", "scheduler.pt"),
            ("rng", "rng.pt"),
            ("trainer", "trainer.pt"),
            ("data_module", "data_module.pt"),
        ):
            f = p / name
            if f.exists():
                out[key] = torch.load(str(f), map_location="cpu", weights_only=False)
        return out

    def list_steps(self) -> list[Path]:
        if not self.ckpt_dir.exists():
            return []
        out: list[tuple[int, Path]] = []
        for child in self.ckpt_dir.iterdir():
            if not child.is_dir():
                continue
            m = _STEP_RE.match(child.name)
            if m and (child / "manifest.json").exists():
                out.append((int(m.group(1)), child))
        out.sort(key=lambda kv: kv[0])
        return [p for _, p in out]

    def latest(self) -> Path | None:
        steps = self.list_steps()
        if not steps:
            return self._read_pointer("last")
        return steps[-1]

    def best(self) -> Path | None:
        return self._read_pointer("best")

    # ---- pointer helpers ---------------------------------------------------

    def _pointer_json(self, kind: str) -> Path:
        return self.ckpt_dir / f"{kind}.json"

    def _pointer_link(self, kind: str) -> Path:
        return self.ckpt_dir / kind

    def _update_pointer(
        self,
        kind: str,
        target: Path,
        *,
        metric_extras: Mapping[str, Any] | None = None,
    ) -> None:
        info: dict[str, Any] = {"target": target.name}
        if metric_extras:
            info["extras"] = dict(metric_extras)
        _atomic_write_text(self._pointer_json(kind), json.dumps(info, indent=2))

        link = self._pointer_link(kind)
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink() or link.is_dir():
                    try:
                        if link.is_dir() and not link.is_symlink():
                            shutil.rmtree(link)
                        else:
                            link.unlink()
                    except OSError:
                        return
            os.symlink(target.name, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows w/o developer mode — JSON pointer is enough.
            return

    def _read_pointer(self, kind: str) -> Path | None:
        link = self._pointer_link(kind)
        if link.is_symlink() or (link.exists() and link.is_dir()):
            return link.resolve()
        j = self._pointer_json(kind)
        if not j.exists():
            return None
        info = json.loads(j.read_text(encoding="utf-8"))
        target_name = info.get("target")
        if not target_name:
            return None
        target = self.ckpt_dir / target_name
        return target if target.exists() else None

    def _prune(self) -> None:
        if self.keep_last_n <= 0:
            return
        steps = self.list_steps()
        excess = len(steps) - self.keep_last_n
        if excess <= 0:
            return
        # Don't delete a step that is the current 'best' or 'last' pointer target.
        # Use _read_pointer so we protect what the on-disk pointer actually points
        # to (which may disagree with list_steps()[-1] if the latest manifest is
        # corrupt or the pointer was updated out-of-sync with manifest writes).
        best = self._read_pointer("best")
        last = self._read_pointer("last")
        for path in steps[:excess]:
            if best is not None and path.resolve() == best.resolve():
                continue
            if last is not None and path.resolve() == last.resolve():
                continue
            shutil.rmtree(path, ignore_errors=True)


__all__ = ["CheckpointManager"]
