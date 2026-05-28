"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from lighttrain.registry import get_registry


@pytest.fixture
def clean_registry():
    """Snapshot the global registry, yield, then restore.

    Tests can register dummies freely; this fixture isolates them.
    """
    reg = get_registry()
    snap = reg.snapshot()
    try:
        yield reg
    finally:
        reg.restore(snap)


@pytest.fixture
def tmp_yaml(tmp_path: Path):
    """Factory: write a YAML file at tmp_path/<name> and return its path."""

    def _write(content: str, name: str = "cfg.yaml") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return _write


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """A scratch directory for multi-file YAML composition."""
    d = tmp_path / "configs"
    d.mkdir()
    return d
