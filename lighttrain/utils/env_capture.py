"""Capture environment metadata for run-dir provenance."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any


def _safe(fn, default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def capture_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": _safe(socket.gethostname, "unknown"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git_sha": _git_sha(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None

    try:
        import accelerate

        info["accelerate"] = accelerate.__version__
    except ImportError:
        info["accelerate"] = None

    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None

    return info


__all__ = ["capture_env"]
