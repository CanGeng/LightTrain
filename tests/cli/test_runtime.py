"""Tests for ``lighttrain.cli._runtime`` helpers.

Covers v0.1.7 fixes:

* Issue #9 — ``_import_user_modules`` must be idempotent within a process so
  resume / multi-stage scripts don't re-run @register decorators and raise
  ``RegistryConflictError``.
* Issue #2 + #10 — ``setup_run_from_config`` accepts either a config path or
  an already-parsed ``RootConfig``; combining ``overrides`` with a parsed
  RootConfig is rejected.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lighttrain.cli._runtime import (
    _IMPORTED_USER_MODULES,
    _auto_attach_m4_callbacks,
    _import_user_modules,
    setup_run_from_config,
)
from lighttrain.config import ConfigError, load_config
from lighttrain.registry import get_registry

# ===========================================================================
# Issue #9 — _import_user_modules idempotency
# ===========================================================================


@pytest.fixture
def _clean_imported_cache():
    """Snapshot the module-level cache so tests don't poison each other."""
    snap = set(_IMPORTED_USER_MODULES)
    try:
        yield
    finally:
        _IMPORTED_USER_MODULES.clear()
        _IMPORTED_USER_MODULES.update(snap)


def test_import_user_modules_idempotent(
    tmp_path: Path, clean_registry, _clean_imported_cache
):
    """Goal (Issue #9): a second call for the same file path must be a no-op,
    not a second ``spec.loader.exec_module`` that re-runs @register and
    raises ``RegistryConflictError``.

    Setup: write a temp ``.py`` that registers a new optimizer (force=False,
    so a duplicate registration WOULD raise). Call _import_user_modules
    twice. The second call must not raise.
    """
    plugin = tmp_path / "v017_idempotency_plugin.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from lighttrain.registry import register

            @register("optimizer", "v017_idempotent_dummy")
            class DummyOpt:
                def __init__(self, **kw): pass
            """
        ),
        encoding="utf-8",
    )

    _import_user_modules([str(plugin)])
    # Pre-fix, this second call would re-execute the decorator and raise.
    _import_user_modules([str(plugin)])

    reg = get_registry()
    assert reg.get("optimizer", "v017_idempotent_dummy") is not None


def test_import_user_modules_dedupes_via_resolved_path(
    tmp_path: Path, clean_registry, _clean_imported_cache
):
    """Goal (Issue #9): the cache key is the *resolved* absolute path, so
    relative + absolute references to the same file collapse to one import.
    """
    plugin = tmp_path / "v017_resolved_dedup_plugin.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from lighttrain.registry import register

            @register("optimizer", "v017_resolved_dedup_dummy")
            class DummyOpt:
                def __init__(self, **kw): pass
            """
        ),
        encoding="utf-8",
    )

    abs_path = str(plugin.resolve())
    _import_user_modules([abs_path])
    _import_user_modules([abs_path])  # exact same key — covered by primary test
    _import_user_modules([str(plugin)])  # same file, possibly non-canonical form

    reg = get_registry()
    assert reg.get("optimizer", "v017_resolved_dedup_dummy") is not None


# ===========================================================================
# Issue #2 + #10 — setup_run_from_config dispatch on Path vs RootConfig
# ===========================================================================


def test_setup_run_from_config_rejects_overrides_with_rootconfig(tmp_path: Path):
    """Goal (Issue #2 + #10): an already-parsed RootConfig already had its
    overrides baked in at load_config time; passing fresh overrides
    alongside it is semantically ambiguous and must raise ``ValueError``
    rather than be silently ignored.
    """
    cfg_path = tmp_path / "recipe.yaml"
    cfg_path.write_text("mode: lab\nseed: 7\n", encoding="utf-8")
    cfg = load_config(cfg_path)

    with pytest.raises(ValueError, match="overrides"):
        setup_run_from_config(cfg, overrides=["++seed=42"])  # type: ignore[arg-type]


def test_setup_run_from_config_rejects_wrong_type(tmp_path: Path):
    """Goal (Issue #2): passing something that's neither a path nor a
    RootConfig raises a clear TypeError naming the bad type — not the
    pre-fix opaque ``argument should be a str or an os.PathLike object``.
    """
    with pytest.raises(TypeError, match="RootConfig"):
        setup_run_from_config(42)  # type: ignore[arg-type]


# ===========================================================================
# Tensor-parallel selection — fail loud instead of silently no-op'ing
# (a requested ``parallel.tp > 1`` that can't be applied must raise).
# ===========================================================================


def _cfg(tmp_path: Path, body: str):
    p = tmp_path / "recipe.yaml"
    p.write_text(body, encoding="utf-8")
    return load_config(p)


# ===========================================================================
# Auto-attached diagnostics — construction failures must not be swallowed:
# critical InvariantsCallback fails loud; non-critical ones warn & skip.
# ===========================================================================


class _FakeBus:
    def __init__(self):
        self.callbacks: list = []

    def add(self, cb):
        self.callbacks.append(cb)


class _FakeTrainer:
    def __init__(self):
        self.bus = _FakeBus()
        self.callbacks: list = []


def test_auto_attach_default_config_constructs(tmp_path: Path):
    """A default/empty config must construct the default InvariantsCallback
    without failing — empty specs fall back to the built-in invariant set
    (not a no-op). Guards against the fail-loud change regressing normal runs.
    """
    cfg = _cfg(tmp_path, "mode: lab\n")
    trainer = _FakeTrainer()
    _auto_attach_m4_callbacks(cfg, trainer, [])
    names = {type(cb).__name__ for cb in trainer.callbacks}
    assert "InvariantsCallback" in names


def test_auto_attach_invariants_failure_fails_loud(tmp_path: Path, monkeypatch):
    """The critical InvariantsCallback failing to construct must raise, not
    silently leave the run without invariants."""
    import lighttrain.builtin_plugins.callbacks.invariants as inv_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(inv_mod, "InvariantsCallback", _Boom)
    with pytest.raises(ConfigError, match="InvariantsCallback"):
        _auto_attach_m4_callbacks(_cfg(tmp_path, "mode: lab\n"), _FakeTrainer(), [])


def test_auto_attach_noncritical_failure_warns(tmp_path: Path, monkeypatch):
    """A non-critical diagnostic (FrozenStepCallback) failing to construct
    must warn and continue, not raise and not silently pass."""
    import lighttrain.builtin_plugins.callbacks.builtins.frozen_step as fs_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(fs_mod, "FrozenStepCallback", _Boom)
    with pytest.warns(UserWarning, match="FrozenStepCallback"):
        _auto_attach_m4_callbacks(_cfg(tmp_path, "mode: lab\n"), _FakeTrainer(), [])


# ---------------------------------------------------------------------------
# Return-contract guardrail (P2 extraction): exact key set for BOTH return
# paths, so splitting setup_run_from_config into stages can't silently drop or
# add a programmatic-API key.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent.parent
_RECIPE = _REPO / "examples" / "references" / "recipes" / "pretrain_causal.yaml"

_BUNDLE_KEYS = {
    "cfg", "resolved_yaml", "run_dir", "model", "data", "optimizer", "scheduler",
    "loss_fn", "callbacks", "logger", "ckpt_manager", "engine", "accelerator",
    "trainer", "device", "lineage_store", "parallel_ctx", "grad_sync",
}


@pytest.mark.skipif(not _RECIPE.exists(), reason="pretrain_causal.yaml missing")
def test_setup_run_from_config_print_only_keyset(tmp_path: Path):
    """``print_config_only`` returns ONLY {cfg, resolved_yaml} (early-return)."""
    out = setup_run_from_config(
        _RECIPE,
        overrides=[f"++run_root={tmp_path.as_posix()}"],
        print_config_only=True,
    )
    assert set(out) == {"cfg", "resolved_yaml"}


@pytest.mark.skipif(not _RECIPE.exists(), reason="pretrain_causal.yaml missing")
def test_setup_run_from_config_bundle_keyset(tmp_path: Path):
    """The full bundle returns exactly the documented programmatic-API key set."""
    bundle = setup_run_from_config(
        _RECIPE,
        overrides=[
            f"++run_root={tmp_path.as_posix()}",
            "++trainer.max_steps=1",
            "++trainer.val_every=0",
            "++trainer.ckpt_every=0",
            "++logger=[{name: jsonl}]",
        ],
    )
    try:
        assert set(bundle) == _BUNDLE_KEYS
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()
