"""Tests for lighttrain.lab.fork — DESIGN §26.10 (M8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.lab.fork import ForkReport, fork
from tests._diagnostics import expect_exists

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint_dir(run_dir: Path, step: int = 100) -> Path:
    """Create a minimal checkpoint directory inside *run_dir*."""
    ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "model.safetensors").write_bytes(b"\x00" * 16)
    (ckpt_dir / "manifest.json").write_text(
        json.dumps({"step": step}), encoding="utf-8"
    )
    (run_dir / "env.json").write_text("{}", encoding="utf-8")
    return ckpt_dir


# ---------------------------------------------------------------------------
# ForkReport structure
# ---------------------------------------------------------------------------


def test_fork_creates_new_run_dir(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "forked"})

    assert isinstance(report, ForkReport)
    expect_exists(report.new_run_dir, tmp_path, what="forked run dir")
    assert report.parent_checkpoint == ckpt.resolve()


def test_fork_copies_checkpoint_files(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "forked"})

    copied_ckpt = report.new_run_dir / "checkpoints" / ckpt.name
    expect_exists(copied_ckpt, report.new_run_dir, what="copied checkpoint dir")
    expect_exists(copied_ckpt / "model.safetensors", copied_ckpt, what="model.safetensors")


def test_fork_writes_fork_meta(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "forked"})

    meta_path = report.new_run_dir / "fork_meta.json"
    expect_exists(meta_path, report.new_run_dir, what="fork_meta.json")
    meta = json.loads(meta_path.read_text())
    assert "fork_of_checkpoint" in meta
    assert str(ckpt.resolve()) in meta["fork_of_checkpoint"]


def test_fork_symlink_mode(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "forked_sym"}, symlink=True)

    copied_ckpt = report.new_run_dir / "checkpoints" / ckpt.name
    assert copied_ckpt.is_symlink()


def test_fork_explicit_run_dir(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)
    explicit_dir = tmp_path / "explicit_new_run"

    report = fork(ckpt, {}, run_dir=explicit_dir)

    assert report.new_run_dir == explicit_dir
    expect_exists(explicit_dir, tmp_path, what="explicit run dir")


def test_fork_nonexistent_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        fork(tmp_path / "nonexistent" / "step_0", {})


def test_fork_writes_config_yaml(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    cfg = {"run_root": str(tmp_path), "exp": "forked", "optim": {"lr": 1e-4}}
    report = fork(ckpt, cfg)

    config_path = report.new_run_dir / "config.yaml"
    expect_exists(config_path, report.new_run_dir, what="config.yaml")
    import yaml

    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["optim"]["lr"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Lineage recording
# ---------------------------------------------------------------------------


def test_fork_records_lineage_edge_when_store_exists(tmp_path: Path):
    from lighttrain.observability.lineage.store import LineageStore

    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent, step=200)

    # Seed a lineage store in the parent run dir
    with LineageStore(parent / "lineage.sqlite") as store:
        store.upsert_node(kind="run", name="parent", run_id="parent_run")

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "fork_lineage"})

    assert report.lineage_edge_recorded is True

    # Verify the edge was written
    with LineageStore(parent / "lineage.sqlite") as store:
        edges = list(store.iter_edges(kind="fork_of"))
    assert len(edges) == 1


def test_fork_lineage_soft_failure_no_sqlite(tmp_path: Path):
    parent = tmp_path / "parent_run"
    parent.mkdir()
    ckpt = _make_checkpoint_dir(parent)

    report = fork(ckpt, {"run_root": str(tmp_path), "exp": "no_lineage"})

    # No lineage.sqlite — should not crash, just return False
    assert report.lineage_edge_recorded is False


# ---------------------------------------------------------------------------
# Three-generation lineage (R16 pattern)
# ---------------------------------------------------------------------------


def test_three_generation_fork_chain(tmp_path: Path):
    from lighttrain.observability.lineage.store import LineageStore

    # Gen 1 — initial run
    gen1 = tmp_path / "gen1"
    gen1.mkdir()
    ckpt1 = _make_checkpoint_dir(gen1, step=50)
    with LineageStore(gen1 / "lineage.sqlite") as _:
        pass  # just create the store

    # Gen 2 — fork from gen1
    r2 = fork(ckpt1, {"run_root": str(tmp_path), "exp": "gen2"})
    ckpt2 = _make_checkpoint_dir(r2.new_run_dir, step=50)
    with LineageStore(r2.new_run_dir / "lineage.sqlite") as _:
        pass

    # Gen 3 — fork from gen2
    r3 = fork(ckpt2, {"run_root": str(tmp_path), "exp": "gen3"})

    # Both gen2 and gen3 have fork_meta.json
    expect_exists(r2.new_run_dir / "fork_meta.json", r2.new_run_dir, what="gen2 fork_meta.json")
    expect_exists(r3.new_run_dir / "fork_meta.json", r3.new_run_dir, what="gen3 fork_meta.json")

    # Gen1 lineage store has a fork_of edge pointing to gen2's run
    with LineageStore(gen1 / "lineage.sqlite") as store:
        gen1_edges = list(store.iter_edges(kind="fork_of"))
    assert len(gen1_edges) == 1

    # Gen2 lineage store has a fork_of edge pointing to gen3's run
    with LineageStore(r2.new_run_dir / "lineage.sqlite") as store:
        gen2_edges = list(store.iter_edges(kind="fork_of"))
    assert len(gen2_edges) == 1
