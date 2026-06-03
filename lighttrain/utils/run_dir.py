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
from datetime import datetime
from pathlib import Path

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


__all__ = ["make_run_dir", "slugify"]
