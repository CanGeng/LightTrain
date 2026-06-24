"""Adversarial tests for ``lighttrain.engine.checkpoint.CheckpointManager``.

The legacy ``tests/test_checkpoint_atomic.py`` is shape-only — it checks
that ``manifest.json`` exists, that ``list_steps()`` returns the right
*count*, etc. The tests here pin:

  * write-order: model files land before ``manifest.json`` (CKPT-ATOMIC)
  * atomicity: every write uses the ``.tmp + os.replace`` pattern
  * tmp-leftover handling: a partial dir is excluded from ``list_steps``
  * load round-trip: tensor values match via ``torch.testing.assert_close``
  * the historical CKPT_PRUNE_02 fix from docs/changelog/v0.1.4
  * sibling invariant: ``best`` pointer target is similarly protected
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from lighttrain.distributed import ParallelContext
from lighttrain.engine.checkpoint import CheckpointManager
from lighttrain.engine.checkpoint import (
    manager as ckpt_manager,  # for monkeypatching internals
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_state(seed: int = 0) -> dict[str, Any]:
    """Return a deterministic state dict suitable for round-trip assertions."""
    torch.manual_seed(seed)
    model = nn.Linear(4, 2)
    return {
        "model": model.state_dict(),
        "rng": {"torch": torch.get_rng_state()},
    }


def _record_os_replace_order(monkeypatch) -> list[tuple[str, str]]:
    """Patch ``os.replace`` to record (src, dst) pairs in call order."""
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _wrapped(src, dst, *a, **kw):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr("os.replace", _wrapped)
    return calls


# --------------------------------------------------------------------------- #
# Atomic-write invariants                                                     #
# --------------------------------------------------------------------------- #


def test_invariant_save_order_manifest_after_model(tmp_run_dir, monkeypatch) -> None:
    """``manifest.json`` is the LAST file to be ``os.replace``-d into the step dir.

    Invariant: the manifest is the presence-marker — readers test for it. If
    it landed before the model file, a concurrent reader could observe a
    "complete" checkpoint that has no weights.

    Input: one save with model+optimizer+rng. Record every ``os.replace`` call;
    find the indices of the model and the manifest. Assert manifest index >
    every other file index.
    """
    calls = _record_os_replace_order(monkeypatch)
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(1, _make_state())

    # Index of manifest replace + index of model replace.
    manifest_idxs = [i for i, (_, dst) in enumerate(calls) if dst.endswith("manifest.json")]
    model_idxs = [i for i, (_, dst) in enumerate(calls) if dst.endswith("model.safetensors")]
    assert manifest_idxs, "no manifest.json os.replace observed"
    assert model_idxs, "no model.safetensors os.replace observed"
    # The step-dir manifest must replace AFTER the step-dir model.
    assert max(model_idxs) < min(manifest_idxs), (
        f"manifest landed before model: replace order = {calls}"
    )


def test_invariant_atomic_write_via_tmp_replace(tmp_run_dir, monkeypatch) -> None:
    """Every write into the step dir uses a ``.tmp`` source and ``os.replace``.

    Invariant: no file is written directly to its final path; that would
    expose half-written bytes to a concurrent reader.
    """
    calls = _record_os_replace_order(monkeypatch)
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(7, _make_state())

    # Filter to replaces whose destination lives inside step_7 (the manifest
    # file we care about) — pointer JSONs may live in checkpoints/ at the
    # parent level, also via .tmp.
    inside = [(s, d) for s, d in calls if "step_7" in d]
    assert inside, "no replaces landed inside step_7"
    for src, dst in inside:
        # tmp names carry a unique ``.tmp.<pid>.<token>`` suffix (so concurrent
        # writers don't collide); match the ``.tmp.`` marker, not a literal suffix.
        assert ".tmp." in src, f"non-atomic write {src} → {dst}"


def test_io_error_during_optimizer_save_leaves_no_manifest(
    tmp_run_dir, monkeypatch
) -> None:
    """If an intermediate write raises, no ``manifest.json`` is produced.

    Invariant (CKPT-ATOMIC): the manifest only exists when the checkpoint is
    complete. A mid-save crash must leave the step dir in a state where
    ``list_steps()`` rejects it.
    """
    # Save model (safetensors) successfully; raise on the FIRST torch.save
    # for the optimizer. Model has already been replaced — but no manifest.

    def _torch_save_raises(*_a, **_kw):
        raise OSError("simulated disk failure during optimizer save")

    mgr = CheckpointManager(tmp_run_dir)
    monkeypatch.setattr(ckpt_manager, "_torch_save_atomic",
                        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))

    state = _make_state()
    state["optimizer"] = {"foo": torch.tensor([1.0])}
    with pytest.raises(IOError):
        mgr.save(3, state)

    step_dir = tmp_run_dir / "checkpoints" / "step_3"
    if step_dir.exists():
        assert not (step_dir / "manifest.json").exists(), (
            "manifest was written despite mid-save failure"
        )
    assert mgr.list_steps() == [], (
        "list_steps must exclude incomplete (no-manifest) step dirs"
    )


def test_partial_write_missing_manifest_skipped_by_list_steps(
    tmp_run_dir,
) -> None:
    """A step dir without ``manifest.json`` is invisible to ``list_steps`` and
    raises ``FileNotFoundError`` on explicit ``load``.

    Input: hand-construct ``step_5/model.safetensors`` without manifest.
    Contract: presence-marker semantics.
    """
    mgr = CheckpointManager(tmp_run_dir)
    step = tmp_run_dir / "checkpoints" / "step_5"
    step.mkdir()
    (step / "model.safetensors").write_bytes(b"\x00\x01\x02")
    assert mgr.list_steps() == []
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        mgr.load(step)


def test_partial_write_tmp_leftover_not_a_step(tmp_run_dir) -> None:
    """A leftover ``.tmp`` from a torn write does not appear as a step.

    Input: write ``step_5/model.safetensors.tmp`` directly (no rename done).
    Pin: ``_STEP_RE`` only matches ``step_<int>`` directories AND requires
    ``manifest.json`` to exist.
    """
    mgr = CheckpointManager(tmp_run_dir)
    step = tmp_run_dir / "checkpoints" / "step_5"
    step.mkdir()
    (step / "model.safetensors.tmp").write_bytes(b"\x00")
    assert mgr.list_steps() == []


def test_load_with_corrupted_manifest_raises_json_decode(tmp_run_dir) -> None:
    """A truncated/garbage ``manifest.json`` surfaces as ``JSONDecodeError``.

    Invariant: load() must fail loud rather than silently return empty.
    """
    mgr = CheckpointManager(tmp_run_dir)
    step = tmp_run_dir / "checkpoints" / "step_4"
    step.mkdir()
    (step / "manifest.json").write_text("{not valid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mgr.load(step)


# --------------------------------------------------------------------------- #
# Prune: CKPT_PRUNE_02 regression + best pointer invariant                    #
# --------------------------------------------------------------------------- #


def _materialize_step(ckpt_dir: Path, step: int) -> Path:
    """Create a minimal valid step dir on disk, bypassing CheckpointManager.save.

    Writes both ``model.safetensors`` (dummy bytes) and ``manifest.json`` so
    ``list_steps`` will pick it up.
    """
    p = ckpt_dir / f"step_{step}"
    p.mkdir()
    (p / "model.safetensors").write_bytes(b"\x00")
    (p / "manifest.json").write_text(
        json.dumps({"step": step, "kind": "step", "files": {}, "extras": {}}),
        encoding="utf-8",
    )
    return p


def test_regression_CKPT_PRUNE_02_protects_stale_last_pointer(tmp_run_dir) -> None:
    """Pre-fix bug: ``_prune`` did not consult ``_read_pointer('last')``, so a
    step targeted by the on-disk ``last`` pointer could be deleted when the
    pointer had not advanced to ``list_steps()[-1]`` (e.g. mid-failure during
    pointer rotation), leaving ``last`` as a dangling soft-link
    (see docs/changelog/v0.1.4: '_prune 不保护 last 指针').

    Input setup (bypasses ``save()`` so retention does not run during setup):
        - manually materialize step_1..step_5 in checkpoints/
        - manually rewrite last.json AND symlink so ``last`` → step_2
          (simulates: pointer never advanced past step_2 due to a crash
          between save-of-step_2 and save-of-step_3's pointer update)
        - mgr = CheckpointManager(run_dir, keep_last_n=2)
        - mgr._prune()  →  excess = 5 - 2 = 3, steps[:3] = [step_1, step_2, step_3]

    Analytical solution:
        Post-fix: the loop at manager.py:265-270 skips step_2 because
        ``path.resolve() == last.resolve()``. step_1 and step_3 are deleted,
        step_2 survives.
        Pre-fix (no last-pointer check): step_2 is rmtree-d → ``last`` is
        now a dangling symlink, and the manager's on-disk invariants break.

    Asserts: step_2 survives; last.resolve() still resolves to step_2; at
    least one other older step (step_1) was actually deleted (proving prune
    ran and was not a no-op).
    """
    ckpt_dir = tmp_run_dir / "checkpoints"

    for step in range(1, 6):
        _materialize_step(ckpt_dir, step)

    # Build a CheckpointManager AFTER the steps already exist on disk, so
    # the constructor does no pruning.
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=2)

    # Manually rewrite the ``last`` pointer to step_2 (both JSON and symlink).
    target = ckpt_dir / "step_2"
    (ckpt_dir / "last.json").write_text(
        json.dumps({"target": target.name}), encoding="utf-8"
    )
    link = ckpt_dir / "last"
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        os.symlink(target.name, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported in this environment")

    # Sanity: pointer resolves to step_2 BEFORE prune.
    assert mgr._read_pointer("last").resolve() == target.resolve()
    assert (ckpt_dir / "step_1").exists()

    mgr._prune()

    # Post-fix expectations:
    assert target.exists(), "step_2 (last pointer target) was wrongly pruned"
    assert mgr._read_pointer("last").resolve() == target.resolve(), (
        "last pointer became dangling after prune"
    )
    # Prove prune actually ran (not a no-op): step_1 is older and unprotected.
    assert not (ckpt_dir / "step_1").exists(), (
        "_prune did not actually delete anything — test scenario broken"
    )


def test_invariant_prune_protects_best_pointer_target(tmp_run_dir) -> None:
    """The on-disk ``best`` pointer target is protected from retention pruning.

    Invariant: the model-of-record (``best``) is never wiped, even when its
    step number falls outside the ``keep_last_n`` window.

    Input: save step_3 with ``kind='best'`` (sets best pointer but does not
    prune), then save step_4, 5, 6 with ``kind='step'`` and keep_last_n=2.
    By the time step_6 is saved, list_steps == [3,4,5,6], excess=2,
    steps[:2] = [step_3, step_4]. step_3 must survive because best→step_3.

    Sanity-check: step_4 (which is NOT the best target) is actually pruned,
    proving the test isn't accidentally a no-op.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=2)
    mgr.save(3, _make_state(seed=3), kind="best")
    mgr.save(4, _make_state(seed=4), kind="step")
    mgr.save(5, _make_state(seed=5), kind="step")
    mgr.save(6, _make_state(seed=6), kind="step")

    ckpt_dir = tmp_run_dir / "checkpoints"
    assert (ckpt_dir / "step_3").exists(), "best pointer target was pruned"
    assert (ckpt_dir / "step_5").exists()
    assert (ckpt_dir / "step_6").exists()
    # step_4 should have been pruned (oldest non-best, falls outside keep_last_n=2).
    assert not (ckpt_dir / "step_4").exists(), (
        "step_4 (not protected) should have been pruned"
    )
    # Best pointer still resolves correctly.
    assert mgr._read_pointer("best").resolve() == (ckpt_dir / "step_3").resolve()


def test_prune_keep_last_zero_is_noop(tmp_run_dir) -> None:
    """``keep_last_n <= 0`` disables pruning entirely.

    Input: save 5 steps with keep_last_n=0; all must survive.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=0)
    for s in range(1, 6):
        mgr.save(s, _make_state(seed=s))
    surviving = sorted(p.name for p in mgr.list_steps())
    assert surviving == [f"step_{s}" for s in range(1, 6)]


def test_prune_keeps_last_n_in_step_order(tmp_run_dir) -> None:
    """With keep_last_n=2, only the two newest steps survive and remain loadable.

    Input: save steps 1..5. Assertion goes beyond name check: each surviving
    manifest must parse to a valid checkpoint loadable by ``mgr.load``.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=2)
    for s in range(1, 6):
        mgr.save(s, _make_state(seed=s))
    survivors = mgr.list_steps()
    survivor_names = sorted(p.name for p in survivors)
    assert survivor_names == ["step_4", "step_5"]
    for p in survivors:
        loaded = mgr.load(p)
        assert loaded["step"] in (4, 5)
        # Model state dict round-trips.
        assert "model" in loaded
        for _k, v in loaded["model"].items():
            assert isinstance(v, torch.Tensor)


# --------------------------------------------------------------------------- #
# Pointer atomicity + symlink resolution                                      #
# --------------------------------------------------------------------------- #


def test_invariant_pointer_json_written_atomic_via_tmp(
    tmp_run_dir, monkeypatch
) -> None:
    """``last.json`` is written via ``.tmp + os.replace``, not a direct truncating
    open.

    Invariant: prevents a partial pointer file from being observed.
    """
    calls = _record_os_replace_order(monkeypatch)
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(1, _make_state())
    pointer_replaces = [c for c in calls if c[1].endswith("last.json")]
    assert pointer_replaces, "no last.json os.replace recorded"
    for src, _dst in pointer_replaces:
        assert ".tmp." in src  # unique ``.tmp.<pid>.<token>`` marker


def test_symlink_resolves_to_step_dir(tmp_run_dir) -> None:
    """``checkpoints/last`` symlink (if creatable) resolves to the latest step dir.

    On platforms where ``os.symlink`` raises (Windows without dev mode), the
    test skips rather than asserting nothing.
    """
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(2, _make_state())
    link = tmp_run_dir / "checkpoints" / "last"
    if not link.is_symlink():
        pytest.skip("symlink not created (likely Windows without dev mode)")
    target = tmp_run_dir / "checkpoints" / "step_2"
    assert link.resolve() == target.resolve()


def test_last_pointer_resolves_via_json_fallback(tmp_run_dir) -> None:
    """``_read_pointer('last')`` resolves through ``last.json`` even when no
    symlink is usable (the Windows / no-dev-mode fallback path).

    Invariant: ``last.json`` records ``{"target": <step_dir_name>}`` and
    ``_read_pointer`` must resolve it to the most-recent step dir regardless
    of symlink availability.
    """
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(1, _make_state())
    target = mgr.save(2, _make_state())

    last_json = tmp_run_dir / "checkpoints" / "last.json"
    assert last_json.exists()
    info = json.loads(last_json.read_text(encoding="utf-8"))
    assert info["target"] == target.name

    resolved = mgr._read_pointer("last")
    assert resolved is not None
    assert resolved.resolve() == target.resolve()


def test_best_pointer_records_metric_extras(tmp_run_dir) -> None:
    """A ``kind='best'`` save writes ``best.json`` carrying the metric extras
    (``metric`` name + ``value``) so the model-of-record is self-describing.
    """
    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(
        3,
        _make_state(seed=3),
        kind="best",
        extras={"metric": "loss", "value": 0.42},
    )
    info = json.loads(
        (tmp_run_dir / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )
    assert info["target"] == "step_3"
    assert info["extras"]["metric"] == "loss"
    assert info["extras"]["value"] == 0.42


def test_save_load_tied_weights(tmp_run_dir) -> None:
    """CheckpointManager must not crash on tied weights (safetensors shared
    storage) and must preserve both values and the tied-weight invariant.
    """
    from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM

    model = TinyCausalLM(
        vocab_size=64, d_model=32, n_layers=1, n_heads=2,
        max_seq_len=32, tie_weights=True,
    )
    assert model.lm_head.weight is model.tok_emb.weight, (
        "pre-condition: weights are tied"
    )

    mgr = CheckpointManager(tmp_run_dir)
    # Must not raise safetensors shared-storage error.
    saved = mgr.save(0, {"model": model.state_dict()})
    assert (saved / "manifest.json").exists()

    # Load into a fresh tied model and verify values + tied-weight semantics.
    fresh = TinyCausalLM(
        vocab_size=64, d_model=32, n_layers=1, n_heads=2,
        max_seq_len=32, tie_weights=True,
    )
    ckpt = mgr.load(saved)
    fresh.load_state_dict(ckpt["model"])

    assert torch.equal(fresh.lm_head.weight, model.tok_emb.weight), (
        "loaded lm_head.weight must equal original tok_emb.weight"
    )
    assert fresh.lm_head.weight is fresh.tok_emb.weight, (
        "tied-weight invariant must survive round-trip into a tied model"
    )


# --------------------------------------------------------------------------- #
# Full-state persistence: data_module + full RNG, manifest-driven load        #
# --------------------------------------------------------------------------- #


def test_save_round_trips_data_module_and_full_rng(tmp_run_dir) -> None:
    """A full save persists ``data_module`` + python/numpy/torch RNG, and the
    manifest references both ``data_module.pt`` and ``rng.pt``.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=3)

    state = {
        "model": {"w": torch.randn(2, 2)},
        "optimizer": {"step": 1, "param_groups": []},
        "scheduler": {"base_lrs": [1e-3]},
        "rng": {
            "python": (3, tuple(range(625)), None),
            "numpy": ("MT19937", [0] * 624, 624, 0, 0.0),
            "torch": torch.get_rng_state(),
        },
        "trainer": {"step": 5},
        "data_module": {"sampler": {"epoch": 1, "consumed": 8}},
    }

    target = mgr.save(5, state, kind="step")

    assert (target / "data_module.pt").exists()
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["data_module"] == "data_module.pt"
    assert manifest["files"]["rng"] == "rng.pt"

    loaded = mgr.load(target)
    assert "data_module" in loaded
    assert loaded["data_module"]["sampler"]["consumed"] == 8
    rng = loaded["rng"]
    assert set(rng.keys()) >= {"python", "numpy", "torch"}


def test_load_is_manifest_driven_not_file_existence(tmp_run_dir) -> None:
    """``load()`` reads components strictly by ``manifest["files"]``; a stale
    on-disk ``optimizer.pt`` from a prior save of the same step must not leak
    into the loaded state when the new manifest no longer lists it.
    """
    mgr = CheckpointManager(tmp_run_dir, keep_last_n=3)

    mgr.save(
        5,
        {"model": {"w": torch.zeros(2)}, "optimizer": {"OLD": 1}},
        kind="step",
    )
    target = mgr.ckpt_dir / "step_5"
    assert (target / "optimizer.pt").exists()

    # Re-save the SAME step WITHOUT optimizer.
    mgr.save(5, {"model": {"w": torch.ones(2)}}, kind="step")

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert "optimizer" not in manifest["files"]

    loaded = mgr.load(target)
    # manifest-driven load ignores the stale optimizer.pt still on disk.
    assert "optimizer" not in loaded


def test_save_docstring_pins_not_crash_atomic_contract() -> None:
    """Pin the documented same-step-overwrite crash-atomicity limitation so a
    change to overwrite semantics consciously updates the docstring.
    """
    save_doc = CheckpointManager.save.__doc__ or ""
    assert "not crash-atomic" in save_doc
    assert "same-step overwrite" in save_doc


# --------------------------------------------------------------------------- #
# Rank-aware save                                                             #
# --------------------------------------------------------------------------- #


def test_save_skips_non_main_rank(tmp_run_dir) -> None:
    """Non-main rank (rank > 0) must produce no files on disk.

    Contract: rank-0 owns disk I/O for checkpoints; other ranks early-return.
    """
    mgr = CheckpointManager(tmp_run_dir)
    ctx = ParallelContext(rank=1, world_size=2)
    result = mgr.save(7, _make_state(), parallel_ctx=ctx)
    assert result is None
    assert mgr.list_steps() == []
    assert not (tmp_run_dir / "checkpoints" / "step_7").exists()


# --------------------------------------------------------------------------- #
# Load round-trip preserves tensor values exactly                             #
# --------------------------------------------------------------------------- #


def test_load_round_trip_preserves_state_dict_values(tmp_run_dir) -> None:
    """Saved model tensors round-trip to identical (within tolerance) values.

    Input: a random Linear(4,2). Save, then load, then assert per-tensor
    closeness with ``atol=1e-5, rtol=1e-4`` — pin numeric exactness, not
    just shape/key presence.
    """
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()}

    mgr = CheckpointManager(tmp_run_dir)
    mgr.save(1, {"model": model.state_dict()})
    loaded = mgr.load(tmp_run_dir / "checkpoints" / "step_1")

    assert set(loaded["model"].keys()) == set(orig.keys())
    for k in orig:
        torch.testing.assert_close(
            loaded["model"][k], orig[k], atol=1e-5, rtol=1e-4
        )


# --------------------------------------------------------------------------- #
# Concurrent writers cannot corrupt manifests                                 #
# --------------------------------------------------------------------------- #


def _parallel_writer_proc(run_dir: str, step: int, seed: int) -> None:
    """Worker that saves a checkpoint at ``step`` into its own run dir."""
    torch.manual_seed(seed)
    model = nn.Linear(4, 2)
    mgr = CheckpointManager(run_dir, keep_last_n=99)
    mgr.save(step, {"model": model.state_dict()})


def test_invariant_atomic_writes_no_partial_bytes_under_concurrency(tmp_path) -> None:
    """The ``.tmp + os.replace`` primitive yields valid JSON under real parallel
    scheduling — even when two writers run truly concurrently at the OS level.

    Invariant: byte-level atomicity of the write primitive. No reader observes
    half-written manifest bytes; ``json.loads`` succeeds on every manifest.

    Setup: spawn two processes, **each writing into its own run dir** (not
    the same dir). This is NOT a test of CheckpointManager's race protection
    — the manager is designed for single-rank-0 use. Concurrent writers
    sharing one dir no longer crash (temp files now carry a unique
    ``.tmp.<pid>.<token>`` suffix), but last/best *ordering* is still
    undefined under multiple writers; see
    ``test_concurrent_writers_same_run_dir_do_not_crash``.
    What this test pins is that the OS-level atomic-rename primitive is
    safe under genuine parallel scheduling stress.
    """
    procs = []
    ctx = mp.get_context("spawn")
    run_dirs: list[Path] = []
    for i, (step, seed) in enumerate(((5, 1), (6, 2))):
        rd = tmp_path / f"run{i}"
        rd.mkdir()
        run_dirs.append(rd)
        p = ctx.Process(
            target=_parallel_writer_proc, args=(str(rd), step, seed)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"writer exited with {p.exitcode}"

    for rd, (step, _) in zip(run_dirs, ((5, 1), (6, 2)), strict=False):
        manifest = rd / "checkpoints" / f"step_{step}" / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))  # would raise on partial bytes
        assert data["step"] == step


def test_concurrent_writers_same_run_dir_do_not_crash(tmp_path) -> None:
    """Two writers sharing ONE run dir (different steps) must not crash.

    Regression for the L3 race: previously both writers raced on the shared
    ``last.json.tmp`` and the loser's ``os.replace`` raised ``FileNotFoundError``
    (subprocess exitcode 1). Unique per-write temp names remove that crash.

    This pins the *no-crash* property only — last/best ordering remains
    undefined under multiple writers (single-writer contract). We use distinct
    steps + a large ``keep_last_n`` so neither step dir is pruned.
    """
    shared = tmp_path / "shared_run"
    shared.mkdir()
    ctx = mp.get_context("spawn")
    procs = []
    for step, seed in ((5, 1), (6, 2)):
        p = ctx.Process(
            target=_parallel_writer_proc, args=(str(shared), step, seed)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"writer exited with {p.exitcode} (FileNotFoundError race?)"

    ckpt_dir = shared / "checkpoints"
    for step in (5, 6):
        assert (ckpt_dir / f"step_{step}" / "manifest.json").exists()
    # last.json was written and points at an existing step dir with a manifest.
    last = json.loads((ckpt_dir / "last.json").read_text(encoding="utf-8"))
    target = ckpt_dir / last["target"]
    assert (target / "manifest.json").exists()


# --------------------------------------------------------------------------- #
# Single-writer contract — make the implicit assumption explicit              #
# --------------------------------------------------------------------------- #


def test_pin_checkpoint_manager_is_single_writer() -> None:
    """Pin the implicit single-writer contract: ``CheckpointManager.save``
    expects only rank-0 to write to disk.

    This contract lives as a comment in ``save()``'s docstring ("In
    distributed runs, pass ``parallel_ctx`` so that only rank-0 writes to
    disk."). Two writers sharing one run dir no longer *crash* (temp files
    carry a unique ``.tmp.<pid>.<token>`` suffix), but last/best ordering is
    still undefined — the single-writer contract stands. This test makes that
    contract a hard requirement so anyone adding real multi-writer support has
    to consciously update the docstring at the same time.

    If this behavior is intentionally changed (e.g. adding fcntl-based
    locking for multi-writer support), update this test AND document the
    new concurrency contract in CheckpointManager docstring.
    """
    save_doc = CheckpointManager.save.__doc__ or ""
    assert "only rank-0" in save_doc, (
        "CheckpointManager.save docstring must contain the literal "
        "'only rank-0' to pin the single-writer contract. If you removed "
        "that phrase intentionally, update this test AND document the new "
        "concurrency contract."
    )
