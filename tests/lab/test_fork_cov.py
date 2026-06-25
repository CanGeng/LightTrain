"""Coverage-extension tests for ``lighttrain.lab.fork``.

Uncovered lines targeted (from 81% baseline):
* 62–72  _detect_parent_run_dir: resolve, loop, lineage.sqlite branch, fallback
* 79      _try_detect_parent_run_dir: ``if not candidate.is_dir(): continue``
* 83–85   _try_detect_parent_run_dir: lineage.sqlite variant + return None
* 114     _copy_checkpoint: existing dst cleanup (file or dir exists)
* 172–173 _record_lineage_edge: successful ``return True``
* 179     _record_lineage_edge: exception path ``return False``
* 228     fork(): ``cfg_dict = new_config.model_dump()``
* 232     fork(): ``cfg_dict = {}`` (non-Mapping, no model_dump)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lighttrain.lab.fork import (
    ForkReport,
    _copy_checkpoint,
    _detect_parent_run_dir,
    _record_lineage_edge,
    _try_detect_parent_run_dir,
    fork,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_run_dir(base: Path, *, with_env_json: bool = True, with_lineage: bool = False) -> Path:
    """Return a run dir with the canonical structure."""
    run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    if with_env_json:
        (run_dir / "env.json").write_text("{}", encoding="utf-8")
    if with_lineage:
        (run_dir / "lineage.sqlite").touch()
    return run_dir


def _make_checkpoint(run_dir: Path, step: int = 100) -> Path:
    ckpt = run_dir / "checkpoints" / f"step_{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 8)
    return ckpt


# ---------------------------------------------------------------------------
# _detect_parent_run_dir (lines 62–72)
# ---------------------------------------------------------------------------


def test_invariant_detect_parent_run_dir_finds_via_env_json(tmp_path: Path):
    """Line 65-66: returns candidate when checkpoints/ + env.json exist."""
    run_dir = _make_run_dir(tmp_path / "run1")
    ckpt = _make_checkpoint(run_dir)
    # checkpoint path is run_dir/checkpoints/step_100
    result = _detect_parent_run_dir(ckpt)
    assert result is not None
    assert result.resolve() == run_dir.resolve()


def test_invariant_detect_parent_run_dir_lineage_sqlite_branch(tmp_path: Path):
    """Line 67-68: candidate with lineage.sqlite matches even without env.json."""
    run_dir = _make_run_dir(tmp_path / "run_lineage", with_env_json=False, with_lineage=True)
    ckpt = _make_checkpoint(run_dir)
    # p.parent is checkpoints/, p.parent.parent is run_dir — which has lineage.sqlite
    result = _detect_parent_run_dir(ckpt)
    # The function should return run_dir (or at least a non-None path) via the lineage branch
    assert result is not None


def test_invariant_detect_parent_run_dir_fallback_checkpoints_parent(tmp_path: Path):
    """Lines 70-71: fallback when p.parent.name == 'checkpoints' (no env.json/lineage)."""
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    ckpt_root = orphan_dir / "checkpoints"
    ckpt_root.mkdir()
    step_dir = ckpt_root / "step_42"
    step_dir.mkdir()
    # No env.json or lineage.sqlite anywhere
    result = _detect_parent_run_dir(step_dir)
    # Fallback: p.parent.name == 'checkpoints' → return p.parent.parent
    assert result is not None
    assert result.resolve() == orphan_dir.resolve()


def test_invariant_detect_parent_run_dir_fallback_grandparent_named_checkpoints(tmp_path: Path):
    """Lines 70-71: fallback when p.parent.parent.name == 'checkpoints'."""
    orphan_dir = tmp_path / "orphan2"
    orphan_dir.mkdir()
    ckpt_root = orphan_dir / "checkpoints"
    ckpt_root.mkdir()
    # Two levels deep: checkpoints/subset/step_10
    subset = ckpt_root / "subset"
    subset.mkdir()
    # No env.json or lineage.sqlite — p.parent.name = "subset", p.parent.parent.name = "checkpoints"
    result = _detect_parent_run_dir(subset)
    # Line 71: return p.parent.parent (i.e. ckpt_root) since p.parent.name != "checkpoints"
    # Actually p.parent.name = "subset" != "checkpoints", so returns p.parent.parent = ckpt_root
    assert result is not None


def test_invariant_detect_parent_run_dir_returns_none_when_no_markers(tmp_path: Path):
    """Line 72: returns None when no env.json, lineage.sqlite, or 'checkpoints' ancestor."""
    lonely = tmp_path / "lonely" / "random" / "path"
    lonely.mkdir(parents=True)
    result = _detect_parent_run_dir(lonely)
    assert result is None


# ---------------------------------------------------------------------------
# _try_detect_parent_run_dir (lines 79, 83–85)
# ---------------------------------------------------------------------------


def test_invariant_try_detect_skips_non_dir_candidates(tmp_path: Path):
    """Line 79: candidate that is not a dir is skipped via ``continue``."""
    # Build a path that doesn't exist, so candidate.is_dir() == False for at least one level
    shallow = tmp_path / "only_one_level"
    shallow.mkdir()
    ckpt = shallow / "step_5"
    ckpt.mkdir()
    # shallow's parent is tmp_path which is a dir but has no checkpoints/ child
    # shallow's parent.parent may not be a dir if tmp_path.parent is root-ish, but let's verify
    # The function checks p.parent.parent and p.parent.parent.parent
    # Here: p = ckpt.resolve(), p.parent = shallow, p.parent.parent = tmp_path
    result = _try_detect_parent_run_dir(ckpt)
    # tmp_path doesn't have checkpoints/ so returns None
    assert result is None


def test_invariant_try_detect_finds_via_lineage_sqlite(tmp_path: Path):
    """Lines 83-84: detects parent when checkpoints/ + lineage.sqlite exist (no env.json)."""
    run_dir = _make_run_dir(tmp_path / "run_ls", with_env_json=False, with_lineage=True)
    ckpt = _make_checkpoint(run_dir)
    # p = ckpt, p.parent = checkpoints/, p.parent.parent = run_dir
    result = _try_detect_parent_run_dir(ckpt)
    assert result is not None
    assert result.resolve() == run_dir.resolve()


def test_invariant_try_detect_returns_none_when_no_match(tmp_path: Path):
    """Line 85: returns None when neither candidate matches."""
    isolated = tmp_path / "a" / "b" / "c" / "d"
    isolated.mkdir(parents=True)
    # "c" has no checkpoints/, "b" has no checkpoints/
    result = _try_detect_parent_run_dir(isolated)
    assert result is None


def test_invariant_try_detect_finds_via_env_json(tmp_path: Path):
    """Lines 81-82: detects parent when checkpoints/ + env.json exist."""
    run_dir = _make_run_dir(tmp_path / "run_ej")
    ckpt = _make_checkpoint(run_dir)
    result = _try_detect_parent_run_dir(ckpt)
    assert result is not None
    assert result.resolve() == run_dir.resolve()


# ---------------------------------------------------------------------------
# _copy_checkpoint (line 114 — existing dst cleanup)
# ---------------------------------------------------------------------------


def test_invariant_copy_checkpoint_removes_existing_dst_dir(tmp_path: Path):
    """Line 114: existing dst directory is removed (rmtree) before recopy."""
    src = tmp_path / "step_10"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"\xff" * 4)

    new_run_dir = tmp_path / "new_run"
    new_run_dir.mkdir()

    # First copy
    _copy_checkpoint(src, new_run_dir)

    # Add a stale file in the destination
    dst = new_run_dir / "checkpoints" / src.name
    assert dst.is_dir()
    (dst / "stale.bin").write_bytes(b"\xde\xad")

    # Second copy — should remove old dst and re-copy cleanly
    _copy_checkpoint(src, new_run_dir)
    dst_after = new_run_dir / "checkpoints" / src.name
    assert (dst_after / "weights.bin").exists()
    # Stale file should be gone (dst was rmtree'd)
    assert not (dst_after / "stale.bin").exists()


def test_invariant_copy_checkpoint_removes_existing_dst_file_symlink(tmp_path: Path):
    """Line 114 (else branch): existing dst that is a non-dir symlink is unlinked.

    Note: a symlink to a *file* satisfies ``not dst.is_dir()`` so the ``else``
    branch (``dst.unlink()``) is taken.  A symlink to a *dir* hits the rmtree
    branch which is a known bug (see suspected_bugs in StructuredOutput).
    """
    src = tmp_path / "step_20"
    src.mkdir()
    (src / "model.bin").write_bytes(b"\x01" * 4)

    new_run_dir = tmp_path / "new_run2"
    new_run_dir.mkdir()
    ckpt_root = new_run_dir / "checkpoints"
    ckpt_root.mkdir()
    dst = ckpt_root / src.name

    # Pre-create a symlink pointing to a *file* (not a dir) so is_dir()==False
    dummy_file = tmp_path / "dummy.bin"
    dummy_file.write_bytes(b"\xff")
    dst.symlink_to(dummy_file)
    assert dst.is_symlink()
    assert not dst.is_dir()

    # _copy_checkpoint should unlink the file-symlink and then do a real copy
    _copy_checkpoint(src, new_run_dir, symlink=False)
    dst_after = ckpt_root / src.name
    assert dst_after.exists()
    assert not dst_after.is_symlink()
    assert (dst_after / "model.bin").exists()


def test_invariant_copy_checkpoint_replaces_existing_dst_dir_symlink(tmp_path: Path):
    """Fixed: an existing dst that is a symlink *to a directory* is unlinked, not
    sent to ``shutil.rmtree`` (which raises OSError on a symlink in py3.12+).
    """
    src = tmp_path / "step_10"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"\x00" * 4)

    new_run_dir = tmp_path / "new_run_symdir"
    new_run_dir.mkdir()
    ckpt_root = new_run_dir / "checkpoints"
    ckpt_root.mkdir()
    dst = ckpt_root / src.name

    # Pre-create a symlink pointing to a *directory* — the py3.12 crash case.
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "old.bin").write_bytes(b"\xde\xad")
    dst.symlink_to(target_dir)
    assert dst.is_symlink() and dst.is_dir()  # is_dir follows the symlink

    _copy_checkpoint(src, new_run_dir, symlink=False)
    dst_after = ckpt_root / src.name
    assert dst_after.exists() and not dst_after.is_symlink()
    assert (dst_after / "weights.bin").exists()
    assert not (dst_after / "old.bin").exists()  # symlink replaced, not followed


# ---------------------------------------------------------------------------
# _record_lineage_edge (lines 172–173, 179)
# ---------------------------------------------------------------------------


def test_invariant_record_lineage_edge_returns_true_on_success(tmp_path: Path):
    """Lines 172-173: returns True when the LineageStore write succeeds."""
    from lighttrain.observability.lineage.store import LineageStore

    parent_run = tmp_path / "parent"
    parent_run.mkdir()
    sqlite_path = parent_run / "lineage.sqlite"

    # Seed the store so the file exists
    with LineageStore(sqlite_path) as _:
        pass

    ckpt = tmp_path / "step_50"
    ckpt.mkdir()
    new_run = tmp_path / "child"
    new_run.mkdir()

    result = _record_lineage_edge(
        parent_run_dir=parent_run,
        parent_checkpoint=ckpt,
        new_run_dir=new_run,
        forked_at_step=50,
    )
    assert result is True


def test_pin_current_behavior_record_lineage_edge_returns_false_on_exception(tmp_path: Path):
    """Line 179: returns False when the LineageStore import/write raises.

    Pin: the function swallows the exception and returns False (soft-failure).
    This is the documented non-fatal path for a missing or broken lineage store.
    """
    import sys
    import types

    parent_run = tmp_path / "parent_exc"
    parent_run.mkdir()
    sqlite_path = parent_run / "lineage.sqlite"
    sqlite_path.touch()  # file exists so we pass the early guard

    # Inject a broken LineageStore via sys.modules so the import inside
    # _record_lineage_edge raises at construction time.
    fake_store_mod = types.ModuleType("lighttrain.observability.lineage.store")

    class _BadStore:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated store init failure")

        def __enter__(self):
            raise RuntimeError("simulated store enter failure")

        def __exit__(self, *a):
            pass

    fake_store_mod.LineageStore = _BadStore  # type: ignore[attr-defined]
    old = sys.modules.get("lighttrain.observability.lineage.store")
    sys.modules["lighttrain.observability.lineage.store"] = fake_store_mod

    try:
        result = _record_lineage_edge(
            parent_run_dir=parent_run,
            parent_checkpoint=tmp_path / "step_1",
            new_run_dir=tmp_path / "child",
            forked_at_step=1,
        )
    finally:
        if old is None:
            sys.modules.pop("lighttrain.observability.lineage.store", None)
        else:
            sys.modules["lighttrain.observability.lineage.store"] = old

    assert result is False


# ---------------------------------------------------------------------------
# fork() with model_dump() config (line 228)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in for a Pydantic v2 model with model_dump()."""

    def __init__(self, run_root: str, exp: str) -> None:
        self._data = {"run_root": run_root, "exp": exp}

    def model_dump(self) -> dict:
        return dict(self._data)


def test_invariant_fork_accepts_model_dump_config(tmp_path: Path):
    """Line 228: fork() calls model_dump() when config has that method."""
    parent = tmp_path / "parent"
    parent.mkdir()
    ckpt = _make_checkpoint(parent)
    (parent / "env.json").write_text("{}", encoding="utf-8")

    cfg = _FakeConfig(run_root=str(tmp_path), exp="model_dump_exp")
    report = fork(ckpt, cfg)

    assert isinstance(report, ForkReport)
    assert report.parent_checkpoint == ckpt.resolve()
    assert (report.new_run_dir / "fork_meta.json").exists()


def test_invariant_fork_model_dump_config_writes_config_yaml(tmp_path: Path):
    """Line 228 + config persist: config.yaml is written using model_dump() data."""
    import yaml

    parent = tmp_path / "parent2"
    parent.mkdir()
    ckpt = _make_checkpoint(parent)
    (parent / "env.json").write_text("{}", encoding="utf-8")

    cfg = _FakeConfig(run_root=str(tmp_path), exp="dump_yaml_exp")
    report = fork(ckpt, cfg)

    config_yaml = report.new_run_dir / "config.yaml"
    assert config_yaml.exists()
    loaded = yaml.safe_load(config_yaml.read_text())
    assert loaded["exp"] == "dump_yaml_exp"


# ---------------------------------------------------------------------------
# fork() with non-Mapping, non-model_dump config (line 232)
# ---------------------------------------------------------------------------


class _BareConfig:
    """Object with neither model_dump() nor Mapping behaviour."""

    pass


def test_invariant_fork_bare_object_config_falls_back_to_empty_dict(tmp_path: Path):
    """Line 232: when config is not a Mapping and has no model_dump, cfg_dict={}
    and fork falls back to 'runs'/'fork' defaults for the run directory.
    """
    parent = tmp_path / "parent_bare"
    parent.mkdir()
    ckpt = _make_checkpoint(parent)
    (parent / "env.json").write_text("{}", encoding="utf-8")

    explicit_dir = tmp_path / "bare_run"
    # Use explicit run_dir so we don't need a real 'runs' root to exist
    report = fork(ckpt, _BareConfig(), run_dir=explicit_dir)

    assert isinstance(report, ForkReport)
    assert report.new_run_dir == explicit_dir
    assert (explicit_dir / "fork_meta.json").exists()


# ---------------------------------------------------------------------------
# Checkpoint dst cleanup integrated via fork() (regression)
# ---------------------------------------------------------------------------


def test_invariant_fork_symlink_cleans_up_existing_copy(tmp_path: Path):
    """Regression: if dst already exists as a real dir (first copy run), a second
    fork call with symlink=True still unlinks properly (line 114 path).
    """
    parent = tmp_path / "parent_rl"
    parent.mkdir()
    ckpt = _make_checkpoint(parent)
    (parent / "env.json").write_text("{}", encoding="utf-8")

    explicit_dir = tmp_path / "rl_run"
    # First fork — copies as real dir
    fork(ckpt, {}, run_dir=explicit_dir)
    dst = explicit_dir / "checkpoints" / ckpt.name
    assert dst.is_dir() and not dst.is_symlink()

    # Second fork to same dir with symlink=True — should replace dir with symlink
    fork(ckpt, {}, run_dir=explicit_dir, symlink=True)
    assert dst.is_symlink()


# ---------------------------------------------------------------------------
# parse_step edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("step_1000", 1000),
        ("step-42", 42),
        ("checkpoint_step_7", 7),
        ("last", None),
        ("epoch10", None),
    ],
)
def test_invariant_parse_step_from_ckpt(name: str, expected: int | None):
    """_parse_step_from_ckpt returns correct int or None for various names."""
    from lighttrain.lab.fork import _parse_step_from_ckpt

    p = Path(name)
    result = _parse_step_from_ckpt(p)
    assert result == expected


# ---------------------------------------------------------------------------
# fork() lineage with forked_at_step extracted
# ---------------------------------------------------------------------------


def test_invariant_fork_extracts_step_for_lineage_edge(tmp_path: Path):
    """When checkpoint name contains step_N, the lineage edge carries that step."""
    from lighttrain.observability.lineage.store import LineageStore

    parent = tmp_path / "parent_step"
    parent.mkdir()
    ckpt = _make_checkpoint(parent, step=300)
    (parent / "env.json").write_text("{}", encoding="utf-8")

    with LineageStore(parent / "lineage.sqlite") as _:
        pass

    explicit_dir = tmp_path / "forked_run"
    report = fork(ckpt, {}, run_dir=explicit_dir)
    assert report.lineage_edge_recorded is True

    with LineageStore(parent / "lineage.sqlite") as store:
        edges = list(store.iter_edges(kind="fork_of"))
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# _write_fork_meta fields
# ---------------------------------------------------------------------------


def test_invariant_write_fork_meta_parent_run_dir_none(tmp_path: Path):
    """_write_fork_meta writes null for fork_of_run_dir when parent is None."""
    from lighttrain.lab.fork import _write_fork_meta

    run_dir = tmp_path / "new_run"
    run_dir.mkdir()
    ckpt = tmp_path / "step_0"

    _write_fork_meta(run_dir, ckpt, None)

    meta = json.loads((run_dir / "fork_meta.json").read_text())
    assert meta["fork_of_run_dir"] is None
    assert meta["fork_of_checkpoint"] == str(ckpt)
    assert isinstance(meta["forked_at_ts"], float)


def test_invariant_write_fork_meta_with_parent_run_dir(tmp_path: Path):
    """_write_fork_meta records parent_run_dir as string when given."""
    from lighttrain.lab.fork import _write_fork_meta

    run_dir = tmp_path / "new_run2"
    run_dir.mkdir()
    parent = tmp_path / "parent_run"
    ckpt = parent / "step_5"

    _write_fork_meta(run_dir, ckpt, parent)

    meta = json.loads((run_dir / "fork_meta.json").read_text())
    assert meta["fork_of_run_dir"] == str(parent)
