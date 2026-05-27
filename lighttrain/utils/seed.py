"""Deterministic seeding."""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy (if available), torch, and CUDA RNGs.

    Returns the (clamped) seed used. ``PYTHONHASHSEED`` is left untouched on
    purpose: setting it after interpreter startup has no effect.
    """
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def rng_state() -> dict[str, Any]:
    """Capture RNG state for checkpointing (functional resume)."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "torch" in state:
        try:
            import torch

            torch.set_rng_state(state["torch"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except ImportError:
            pass


__all__ = ["restore_rng_state", "rng_state", "seed_everything"]
