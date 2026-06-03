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

import lighttrain.cli._runtime as _runtime
from lighttrain.cli._runtime import (
    _IMPORTED_USER_MODULES,
    _auto_attach_m4_callbacks,
    _build_model_parallel_strategy,
    _build_pipeline_schedule,
    _import_user_modules,
    setup_run_from_config,
)
from lighttrain.config import ConfigError, load_config
from lighttrain.registry import RegistryConflictError, get_registry


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
        setup_run_from_config(cfg, overrides=["++seed=42"])


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


def test_tp_disabled_returns_none(tmp_path: Path):
    """No parallel section, or ``tp <= 1``, is the normal single-GPU path —
    returns None, never raises."""
    assert _build_model_parallel_strategy(_cfg(tmp_path, "mode: lab\n")) is None
    cfg = _cfg(tmp_path, "mode: lab\nparallel: {tp: 1}\n")
    assert _build_model_parallel_strategy(cfg) is None


def test_tp_requested_without_block_raises(tmp_path: Path):
    """``parallel.tp > 1`` but no ``tensor_parallel:`` block previously
    returned None silently (user thinks they're parallel, but aren't).
    It must now fail loud."""
    cfg = _cfg(tmp_path, "mode: lab\nparallel: {tp: 4}\n")
    with pytest.raises(ConfigError, match="tensor_parallel"):
        _build_model_parallel_strategy(cfg)


def test_tp_strategy_unregistered_raises(tmp_path: Path, monkeypatch):
    """``tp > 1`` with a ``tensor_parallel:`` block but no registered strategy
    (plugins not loaded) must raise, not return None."""
    cfg = _cfg(tmp_path, "mode: lab\nparallel:\n  tp: 4\n  tensor_parallel: {}\n")

    def _boom(*_a, **_k):
        raise KeyError("model_parallel_strategy/tensor_parallel")

    monkeypatch.setattr(_runtime, "_registry_get", _boom)
    with pytest.raises(ConfigError, match="not registered"):
        _build_model_parallel_strategy(cfg)


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
    import lighttrain.callbacks.invariants as inv_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(inv_mod, "InvariantsCallback", _Boom)
    with pytest.raises(ConfigError, match="InvariantsCallback"):
        _auto_attach_m4_callbacks(_cfg(tmp_path, "mode: lab\n"), _FakeTrainer(), [])


def test_auto_attach_noncritical_failure_warns(tmp_path: Path, monkeypatch):
    """A non-critical diagnostic (FrozenStepCallback) failing to construct
    must warn and continue, not raise and not silently pass."""
    import lighttrain.callbacks.builtins.frozen_step as fs_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(fs_mod, "FrozenStepCallback", _Boom)
    with pytest.warns(UserWarning, match="FrozenStepCallback"):
        _auto_attach_m4_callbacks(_cfg(tmp_path, "mode: lab\n"), _FakeTrainer(), [])


# ===========================================================================
# SP / EP — registered but not wired into the runtime: must fail loud.
# ===========================================================================


def test_sp_requested_fails_loud(tmp_path: Path):
    """`parallel.sp: true` is registered but not wired into the selector — it
    must fail loud instead of silently no-op'ing."""
    cfg = _cfg(tmp_path, "mode: lab\nparallel: {sp: true}\n")
    with pytest.raises(ConfigError, match="sequence parallelism"):
        _build_model_parallel_strategy(cfg)


def test_ep_requested_fails_loud(tmp_path: Path):
    """`parallel.ep > 1` is a skeleton (no real all-to-all) — must fail loud."""
    cfg = _cfg(tmp_path, "mode: lab\nparallel: {ep: 2}\n")
    with pytest.raises(ConfigError, match="expert parallelism"):
        _build_model_parallel_strategy(cfg)


# ===========================================================================
# Pipeline schedule selection — honors `parallel.pipeline.schedule`, fails loud
# on an unknown schedule, and drops the `schedule` selector key from ctor kwargs.
# ===========================================================================


def test_pipeline_schedule_disabled_returns_none(tmp_path: Path):
    """No parallel section, or `pp <= 1`, is the normal path — None, no raise."""
    assert _build_pipeline_schedule(_cfg(tmp_path, "mode: lab\n")) is None
    cfg = _cfg(tmp_path, "mode: lab\nparallel: {pp: 1}\n")
    assert _build_pipeline_schedule(cfg) is None


def test_pipeline_schedule_gpipe_selected(tmp_path: Path):
    """`schedule: gpipe` selects GPipeSchedule, and the `schedule` selector key
    is dropped from ctor kwargs (GPipeSchedule.__init__ takes no `schedule=`)."""
    from lighttrain.config._components import import_all_components

    import_all_components()
    cfg = _cfg(tmp_path, "mode: lab\nparallel:\n  pp: 2\n  pipeline: {schedule: gpipe}\n")
    sched = _build_pipeline_schedule(cfg)
    assert type(sched).__name__ == "GPipeSchedule"


def test_pipeline_unknown_schedule_fails_loud(tmp_path: Path):
    """An unknown `schedule:` must raise ConfigError, not silently fall back."""
    cfg = _cfg(tmp_path, "mode: lab\nparallel:\n  pp: 2\n  pipeline: {schedule: nope}\n")
    with pytest.raises(ConfigError, match="not registered"):
        _build_pipeline_schedule(cfg)


def test_setup_run_tp_misconfig_raises_configerror_not_rank(tmp_path: Path):
    """Integration: a non-torchrun run with `parallel.tp > 1` but no
    `tensor_parallel:` block must surface the precise ConfigError from the
    parallel-config preflight — not a generic 'RANK expected' from process-group
    init. The preflight runs before the run dir is created, so a misconfig must
    also leave no polluting run dir under ``runs/``."""
    run_root = tmp_path / "runs"
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        f"mode: lab\nrun_root: {run_root}\nparallel: {{tp: 4}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="tensor_parallel"):
        setup_run_from_config(cfg)
    assert not run_root.exists(), "invalid parallel config must not create a run dir"
