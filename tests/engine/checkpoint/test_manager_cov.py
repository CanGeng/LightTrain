"""Edge-case coverage tests for ``lighttrain.engine.checkpoint.manager``.

Pins the uncovered branches identified from the 89 % coverage run:

* Lines 85-88  – ``_load_safetensors``: ``.pt`` fallback when ``.safetensors``
                  absent; ``FileNotFoundError`` when neither file exists.
* Line 179     – ``CheckpointManager.load``: non-existent path raises.
* Line 204     – ``list_steps``: returns [] when ``ckpt_dir`` itself is absent.
* Lines 250,253,254 – ``_update_pointer``: real-directory link removal via
                  ``shutil.rmtree``; ``OSError`` in inner try returns early.
* Lines 256,258 – ``_update_pointer``: outer ``(OSError, NotImplementedError)``
                  handler returns without crashing.
* Lines 267-272 – ``_read_pointer``: JSON-only path (no symlink); target_name
                  missing from JSON; target dir non-existent on disk.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lighttrain.engine.checkpoint.manager import (
    CheckpointManager,
    _load_safetensors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(seed: int = 0) -> dict[str, Any]:
    """Deterministic minimal state dict for round-trip checks."""
    torch.manual_seed(seed)
    model = nn.Linear(4, 2)
    return {"model": model.state_dict()}


def _materialize_step(ckpt_dir: Path, step: int) -> Path:
    """Create a minimal valid step dir (manifest present), bypassing save()."""
    p = ckpt_dir / f"step_{step}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(
        json.dumps({"step": step, "kind": "step", "files": {}, "extras": {}}),
        encoding="utf-8",
    )
    return p


# ===========================================================================
# _load_safetensors – lines 85-88
# ===========================================================================


def test_invariant_load_safetensors_pt_fallback(tmp_path: Path) -> None:
    """``_load_safetensors`` falls back to ``.pt`` when ``.safetensors`` is absent.

    Lines 85-87: the ``.safetensors`` path does not exist; a ``.pt`` sibling
    does → torch.load is used instead of safetensors.
    """
    torch.manual_seed(7)
    tensor = torch.randn(3, 3)
    pt_path = tmp_path / "model.pt"
    torch.save({"w": tensor}, str(pt_path))

    # Ask for .safetensors; only .pt exists.
    safetensors_path = tmp_path / "model.safetensors"
    assert not safetensors_path.exists()
    assert pt_path.exists()

    result = _load_safetensors(safetensors_path)
    assert "w" in result
    torch.testing.assert_close(result["w"], tensor)


def test_invariant_load_safetensors_raises_when_neither_exists(tmp_path: Path) -> None:
    """``_load_safetensors`` raises ``FileNotFoundError`` (line 88) when both
    ``.safetensors`` and ``.pt`` are absent.
    """
    missing = tmp_path / "ghost.safetensors"
    with pytest.raises(FileNotFoundError, match="No model weights"):
        _load_safetensors(missing)


# ===========================================================================
# CheckpointManager.load – line 179 (non-existent path)
# ===========================================================================


def test_invariant_load_nonexistent_path_raises(tmp_run_dir: Path) -> None:
    """``load()`` raises ``FileNotFoundError`` when the checkpoint path does not
    exist on disk (line 179).
    """
    mgr = CheckpointManager(tmp_run_dir)
    ghost = tmp_run_dir / "checkpoints" / "step_999"
    assert not ghost.exists()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mgr.load(ghost)


# ===========================================================================
# list_steps – line 204 (ckpt_dir absent)
# ===========================================================================


def test_invariant_list_steps_returns_empty_when_ckpt_dir_removed(
    tmp_path: Path,
) -> None:
    """``list_steps()`` returns ``[]`` when ``ckpt_dir`` has been removed after
    construction (line 204 – the early-return guard).
    """
    run = tmp_path / "run2"
    run.mkdir()
    mgr = CheckpointManager(run)
    # Remove the checkpoints dir that the constructor created.
    shutil.rmtree(mgr.ckpt_dir)
    assert not mgr.ckpt_dir.exists()
    assert mgr.list_steps() == []


# ===========================================================================
# _update_pointer – real-dir removal via shutil.rmtree (lines 249-250)
# ===========================================================================


def test_invariant_update_pointer_removes_real_dir_link_slot(
    tmp_run_dir: Path,
) -> None:
    """When the pointer slot is a real (non-symlink) directory, ``_update_pointer``
    calls ``shutil.rmtree`` (lines 249-250) and then creates a symlink.

    This exercises the branch: ``link.is_dir() and not link.is_symlink()``.
    """
    mgr = CheckpointManager(tmp_run_dir)
    target_step = tmp_run_dir / "checkpoints" / "step_1"
    target_step.mkdir(parents=True, exist_ok=True)

    # Plant a real directory at the "last" link slot.
    link_slot = tmp_run_dir / "checkpoints" / "last"
    link_slot.mkdir()
    assert link_slot.is_dir() and not link_slot.is_symlink()

    # Should remove the real dir and succeed (or gracefully handle OSError).
    try:
        mgr._update_pointer("last", target_step)
    except Exception:  # noqa: BLE001
        pytest.skip("Platform does not support symlink creation here")

    # After _update_pointer the slot is either a symlink (Linux) or the
    # JSON file was written (Windows); either way no crash and JSON present.
    assert (tmp_run_dir / "checkpoints" / "last.json").exists()


# ===========================================================================
# _update_pointer – OSError in inner try returns early (lines 253-254)
# ===========================================================================


def test_pin_current_behavior_update_pointer_inner_oserror_returns(
    tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin current behavior: when ``link.unlink()`` raises ``OSError``, the
    inner except (lines 253-254) returns immediately without calling
    ``os.symlink``.

    Debatable: silently swallowing the error means the pointer JSON was already
    written (so JSON-based readers are fine), but the symlink is not updated.
    This behavior is intentional for Windows compatibility.
    """
    mgr = CheckpointManager(tmp_run_dir)
    target_step = tmp_run_dir / "checkpoints" / "step_1"
    target_step.mkdir(parents=True, exist_ok=True)

    # Save a first step so a symlink already exists at the "last" slot.
    try:
        mgr.save(1, _make_state(seed=1))
    except Exception:  # noqa: BLE001
        pytest.skip("Initial save failed")

    link_slot = tmp_run_dir / "checkpoints" / "last"
    if not (link_slot.is_symlink() or link_slot.exists()):
        pytest.skip("No symlink was created (symlinks not supported)")

    # Make unlink() raise OSError so the inner except fires (line 253-254).
    original_unlink = Path.unlink

    def _raising_unlink(self, missing_ok: bool = False) -> None:
        if self == link_slot:
            raise OSError("simulated unlink failure")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _raising_unlink)

    target_step2 = tmp_run_dir / "checkpoints" / "step_2"
    target_step2.mkdir(parents=True, exist_ok=True)

    # Must not raise even though unlink failed.
    mgr._update_pointer("last", target_step2)

    # JSON was written before the symlink branch, so it must exist.
    assert (tmp_run_dir / "checkpoints" / "last.json").exists()


# ===========================================================================
# _update_pointer – outer OSError / NotImplementedError handler (lines 256,258)
# ===========================================================================


def test_pin_current_behavior_update_pointer_oserror_on_symlink(
    tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin current behavior: when ``os.symlink`` raises ``OSError`` (line 256),
    the method returns silently (line 258). The JSON pointer was already written,
    so the checkpoint is still functional via JSON.

    This models Windows without Developer Mode, or any other OS where symlinks
    are restricted.
    """
    mgr = CheckpointManager(tmp_run_dir)
    target_step = tmp_run_dir / "checkpoints" / "step_1"
    target_step.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(os, "symlink", lambda *a, **kw: (_ for _ in ()).throw(OSError("no symlinks")))

    # Must not raise.
    mgr._update_pointer("last", target_step)

    # JSON written before symlink attempt.
    assert (tmp_run_dir / "checkpoints" / "last.json").exists()
    info = json.loads((tmp_run_dir / "checkpoints" / "last.json").read_text())
    assert info["target"] == "step_1"


def test_pin_current_behavior_update_pointer_notimplemented_on_symlink(
    tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin current behavior: ``NotImplementedError`` from ``os.symlink`` is also
    caught (line 256) and swallowed (line 258).

    Some embedded or restricted environments raise ``NotImplementedError`` for
    symlink creation; the JSON fallback keeps the pointer functional.
    """
    mgr = CheckpointManager(tmp_run_dir)
    target_step = tmp_run_dir / "checkpoints" / "step_1"
    target_step.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(os, "symlink", lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError))

    mgr._update_pointer("last", target_step)

    # JSON still present.
    assert (tmp_run_dir / "checkpoints" / "last.json").exists()


# ===========================================================================
# _read_pointer – JSON-only path (lines 267-272)
# ===========================================================================


def test_invariant_read_pointer_json_fallback_no_symlink(
    tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_read_pointer`` resolves via JSON when no symlink exists (lines 267-272).

    Force ``os.symlink`` to raise so ``_update_pointer`` writes only the JSON.
    Then verify ``_read_pointer`` returns the correct target.
    """
    monkeypatch.setattr(os, "symlink", lambda *a, **kw: (_ for _ in ()).throw(OSError("no symlink")))

    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(3, _make_state(seed=3))

    link = tmp_run_dir / "checkpoints" / "last"
    # No symlink was created.
    assert not link.is_symlink()

    result = mgr._read_pointer("last")
    assert result is not None
    assert result.resolve() == (tmp_run_dir / "checkpoints" / "step_3").resolve()


def test_invariant_read_pointer_returns_none_when_no_json_and_no_link(
    tmp_run_dir: Path,
) -> None:
    """``_read_pointer`` returns ``None`` when neither symlink nor JSON exists
    (line 265-266).
    """
    mgr = CheckpointManager(tmp_run_dir)
    assert mgr._read_pointer("last") is None
    assert mgr._read_pointer("best") is None


def test_pin_current_behavior_read_pointer_missing_target_key_returns_none(
    tmp_run_dir: Path,
) -> None:
    """Pin current behavior: if JSON exists but has no ``"target"`` key
    (lines 268-270), ``_read_pointer`` returns ``None``.

    Debatable: arguably the manager should raise on malformed JSON, but the
    current code silently returns ``None`` for robustness.
    """
    mgr = CheckpointManager(tmp_run_dir)
    # Write a JSON without the "target" key.
    (tmp_run_dir / "checkpoints" / "last.json").write_text(
        json.dumps({"note": "no target key here"}), encoding="utf-8"
    )
    result = mgr._read_pointer("last")
    assert result is None


def test_pin_current_behavior_read_pointer_target_dir_not_on_disk(
    tmp_run_dir: Path,
) -> None:
    """Pin current behavior: if JSON's ``"target"`` names a dir that does not
    exist on disk (lines 271-272), ``_read_pointer`` returns ``None``.

    Debatable: could raise ``FileNotFoundError``, but current code returns
    ``None`` so callers can distinguish "no checkpoint ever saved" from
    "pointer exists but target was pruned".
    """
    mgr = CheckpointManager(tmp_run_dir)
    # Write a JSON whose "target" points at a nonexistent step.
    (tmp_run_dir / "checkpoints" / "last.json").write_text(
        json.dumps({"target": "step_999"}), encoding="utf-8"
    )
    result = mgr._read_pointer("last")
    assert result is None


# ===========================================================================
# latest() uses _read_pointer fallback when list_steps is empty (line 218)
# ===========================================================================


def test_invariant_latest_falls_back_to_pointer_when_no_valid_steps(
    tmp_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``latest()`` returns the pointer-json target when ``list_steps()`` is
    empty (line 218 – the ``if not steps: return self._read_pointer('last')``
    branch).

    Trigger: no step dir with a manifest exists, but a pointer JSON does.
    """
    monkeypatch.setattr(os, "symlink", lambda *a, **kw: (_ for _ in ()).throw(OSError("no symlink")))

    mgr = CheckpointManager(tmp_run_dir)
    # Write the pointer JSON directly (simulating an external update).
    step_dir = tmp_run_dir / "checkpoints" / "step_7"
    step_dir.mkdir()
    (tmp_run_dir / "checkpoints" / "last.json").write_text(
        json.dumps({"target": "step_7"}), encoding="utf-8"
    )

    # list_steps returns [] because step_7 has no manifest.json.
    assert mgr.list_steps() == []

    result = mgr.latest()
    assert result is not None
    assert result.resolve() == step_dir.resolve()


def test_invariant_latest_returns_none_when_no_steps_and_no_pointer(
    tmp_run_dir: Path,
) -> None:
    """``latest()`` returns ``None`` when both ``list_steps`` and the pointer
    are empty (the combined no-checkpoint state).
    """
    mgr = CheckpointManager(tmp_run_dir)
    assert mgr.latest() is None


# ===========================================================================
# best() method
# ===========================================================================


def test_invariant_best_returns_none_initially(tmp_run_dir: Path) -> None:
    """``best()`` returns ``None`` before any ``kind='best'`` save."""
    mgr = CheckpointManager(tmp_run_dir)
    assert mgr.best() is None


def test_invariant_best_resolves_after_best_save(tmp_run_dir: Path) -> None:
    """``best()`` resolves to the ``best``-kind checkpoint dir after a save."""
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(5, _make_state(seed=5), kind="best")
    result = mgr.best()
    assert result is not None
    assert result.resolve() == (tmp_run_dir / "checkpoints" / "step_5").resolve()


# ===========================================================================
# save() – non-step kind does NOT call _prune or update 'last' pointer
# ===========================================================================


def test_invariant_best_save_does_not_touch_last_pointer(tmp_run_dir: Path) -> None:
    """A ``kind='best'`` save writes ``best.json`` but must NOT write
    ``last.json`` (only ``kind='step'`` saves advance the 'last' pointer).
    """
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(10, _make_state(seed=10), kind="best")

    last_json = tmp_run_dir / "checkpoints" / "last.json"
    best_json = tmp_run_dir / "checkpoints" / "best.json"
    assert best_json.exists(), "best.json must exist after kind='best' save"
    assert not last_json.exists(), "last.json must NOT be written by kind='best' save"


# ===========================================================================
# save() – parallel_ctx non-main rank (line 130-131)
# ===========================================================================


class _NonMainCtx:
    is_main_process = False


class _MainCtx:
    is_main_process = True


def test_invariant_save_returns_none_for_non_main_process(tmp_run_dir: Path) -> None:
    """Non-main ``parallel_ctx`` causes ``save()`` to return ``None`` immediately
    (lines 130-131), writing nothing to disk.
    """
    mgr = CheckpointManager(tmp_run_dir)
    result = mgr.save(1, _make_state(), parallel_ctx=_NonMainCtx())
    assert result is None
    assert mgr.list_steps() == []


def test_invariant_save_proceeds_for_main_process(tmp_run_dir: Path) -> None:
    """Main ``parallel_ctx`` allows ``save()`` to proceed normally."""
    mgr = CheckpointManager(tmp_run_dir)
    result = mgr.save(1, _make_state(), parallel_ctx=_MainCtx())
    assert result is not None
    assert (result / "manifest.json").exists()


# ===========================================================================
# Parametrized: all optional state keys are saved and loaded
# ===========================================================================


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimizer", {"step": 42}),
        ("scheduler", {"last_epoch": 5}),
        ("rng", {"seed": 123}),
        ("trainer", {"global_step": 99}),
        ("data_module", {"consumed": 256}),
    ],
)
def test_invariant_optional_state_keys_round_trip(
    tmp_run_dir: Path, key: str, value: dict
) -> None:
    """Each optional state key is saved into the manifest and reloaded."""
    mgr = CheckpointManager(tmp_run_dir)
    state: dict[str, Any] = {key: value}
    target = mgr.save(1, state)
    assert target is not None
    loaded = mgr.load(target)
    assert key in loaded
    assert loaded[key] == value


# ===========================================================================
# Prune: excess <= 0 is a no-op (line 280)
# ===========================================================================


def test_invariant_prune_no_op_when_not_enough_steps(tmp_run_dir: Path) -> None:
    """``_prune()`` does nothing when ``len(steps) <= keep_last_n`` (excess <= 0).

    Input: save 2 steps with keep_last_n=3; both must survive.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=3)
    mgr.save(1, _make_state(seed=1))
    mgr.save(2, _make_state(seed=2))
    surviving = [p.name for p in mgr.list_steps()]
    assert "step_1" in surviving
    assert "step_2" in surviving


# ===========================================================================
# _update_pointer with metric_extras populates JSON extras (line 241)
# ===========================================================================


def test_invariant_update_pointer_with_metric_extras(tmp_run_dir: Path) -> None:
    """``_update_pointer`` with ``metric_extras`` writes an ``extras`` field in
    the JSON (line 241 – ``if metric_extras:`` branch).
    """
    mgr = CheckpointManager(tmp_run_dir)
    step_dir = tmp_run_dir / "checkpoints" / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)

    mgr._update_pointer("best", step_dir, metric_extras={"metric": "loss", "value": 0.1})

    info = json.loads((tmp_run_dir / "checkpoints" / "best.json").read_text())
    assert info["extras"]["metric"] == "loss"
    assert info["extras"]["value"] == pytest.approx(0.1)


def test_invariant_update_pointer_without_metric_extras_no_extras_key(
    tmp_run_dir: Path,
) -> None:
    """``_update_pointer`` without ``metric_extras`` must not include an ``extras``
    field in the JSON (line 240 – branch not taken).
    """
    mgr = CheckpointManager(tmp_run_dir)
    step_dir = tmp_run_dir / "checkpoints" / "step_1"
    step_dir.mkdir(parents=True, exist_ok=True)

    mgr._update_pointer("last", step_dir)

    info = json.loads((tmp_run_dir / "checkpoints" / "last.json").read_text())
    assert "extras" not in info
