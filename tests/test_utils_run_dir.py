"""Run-dir layout + env.json + slugify behavior."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lighttrain.utils.run_dir import make_run_dir, slugify


def test_slugify_lowercases_and_dasherizes():
    assert slugify("Tiny Pretrain!!") == "tiny-pretrain"
    assert slugify("hello___WORLD") == "hello___world"


def test_slugify_empty_falls_back():
    assert slugify("") == "run"


def test_make_run_dir_creates_layout(tmp_path: Path):
    out = make_run_dir(
        tmp_path / "runs",
        "tiny_pretrain",
        slug="smoke",
        snapshot_yaml="seed: 1",
        resolved_yaml="seed: 1\n",
        extra_env={"k": "v"},
    )
    assert out.exists() and out.is_dir()
    assert (out / "logs").is_dir()
    assert (out / "checkpoints").is_dir()
    assert (out / "config.snapshot.yaml").read_text(encoding="utf-8") == "seed: 1"
    assert (out / "config.resolved.yaml").exists()

    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["k"] == "v"
    assert "python" in env or "python_version" in env

    name = out.name
    # ts-slug-hash pattern.
    assert re.match(r"^\d{8}-\d{6}-smoke-[0-9a-f]{8}$", name)


def test_make_run_dir_unique_per_call(tmp_path: Path):
    a = make_run_dir(tmp_path / "r", "exp", snapshot_yaml="a")
    b = make_run_dir(tmp_path / "r", "exp", snapshot_yaml="b")
    assert a != b
