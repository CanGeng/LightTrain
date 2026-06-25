"""Run-directory builder.

Layout::

    runs/<exp>/<ts>-<slug>-<short_hash>/
        config.snapshot.yaml   # exact YAML the user passed (pre-resolution)
        config.resolved.yaml   # post-merge, post-overrides, post-interpolation
        env.json               # capture_env() output
        logs/                  # metrics.jsonl + tensorboard events
        checkpoints/           # step_<n>/, last/, best/
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .env_capture import capture_env
from .hashing import short_hash

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(text: str, max_len: int = 32) -> str:
    s = text.strip().lower()
    s = _SLUG_RE.sub("-", s)
    s = s.strip("-_")
    return (s or "run")[:max_len]


def make_run_dir(
    root: str | Path,
    exp: str,
    *,
    slug: str | None = None,
    snapshot_yaml: str = "",
    resolved_yaml: str = "",
    extra_env: dict | None = None,
) -> Path:
    """Create a fresh ``runs/<exp>/<ts>-<slug>-<hash>/`` and seed its files."""
    root = Path(root)
    exp_slug = slugify(exp) or "run"
    slug = slugify(slug) if slug else exp_slug
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    hash_input = (snapshot_yaml or resolved_yaml or ts) + slug
    h = short_hash(hash_input, n=8)

    run_dir = root / exp_slug / f"{ts}-{slug}-{h}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if snapshot_yaml:
        (run_dir / "config.snapshot.yaml").write_text(snapshot_yaml, encoding="utf-8")
    if resolved_yaml:
        (run_dir / "config.resolved.yaml").write_text(resolved_yaml, encoding="utf-8")

    env = capture_env()
    if extra_env:
        env.update(extra_env)
    (run_dir / "env.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8"
    )

    return run_dir


def broadcast_run_dir(
    factory: Callable[[], Path],
    *,
    world_size: int,
    is_main: bool,
    device: Any,
) -> Path:
    """Make all ranks agree on one run dir: rank 0 creates it, then broadcasts.

    ``make_run_dir`` timestamps the path with ``datetime.now()``, which each
    rank evaluates independently — a multi-rank launch straddling a one-second
    boundary would otherwise split ranks across sibling run dirs. Here only the
    main process calls ``factory`` (the dir-creating side effect); the resulting
    path string is broadcast so every rank uses it.

    Single-process (or pre-dist) callers just call ``factory`` directly.

    ``broadcast_object_list`` is a collective and thus a sync point: when it
    returns, rank 0 has finished ``factory()`` (dir created + seeded), so the
    path is safe to use on every rank. ``device`` must be the rank's local
    device (cuda:local_rank for nccl, cpu for gloo/force_cpu).
    """
    import torch.distributed as dist

    if world_size <= 1 or not dist.is_initialized():
        return factory()
    payload: list[str | None] = [None]
    if is_main:
        payload[0] = str(factory())
    dist.broadcast_object_list(payload, src=0, device=device)
    assert payload[0] is not None  # populated by src=0 broadcast
    return Path(payload[0])


__all__ = ["broadcast_run_dir", "make_run_dir", "slugify"]
