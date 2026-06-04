"""CLI runtime helpers — config → run dir → components → trainer wiring.

Keeps ``cli/_app.py`` thin: every train-style command boils down to
``setup_run_from_config(...)`` + ``trainer.fit()``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args

import torch
from omegaconf import OmegaConf

from lighttrain.builtin_plugins.engine.standard import StandardEngine
from lighttrain.builtin_plugins.update_rules.standard import StandardUpdateRule

from .. import __version__
from ..architectures.profile import ArchitectureProfile, LossOnlyObjective
from ..checkpoint.manager import CheckpointManager
from ..config import ConfigError, RootConfig, load_config
from ..config._components import import_all_components
from ..config._models import (
    build_primary_model,
    normalize_model_set,
    optim_spec_for,
    primary_trainable,
)
from ..config._resolver import _as_plain_dict as _to_dict
from ..config._resolver import resolve as _resolve

# ``user_modules`` import now lives in ``config/`` and is invoked at the
# ``load_config`` chokepoint (so every recipe-eating command gets it for free).
# Re-exported here under the old private name for backward compatibility with
# callers/tests that import it from ``cli._runtime``; both names share the one
# process-wide dedup set.
from ..config._user_modules import _IMPORTED_USER_MODULES  # noqa: F401
from ..config._user_modules import import_user_modules as _import_user_modules
from ..distributed._context import ParallelContext
from ..engine._context import StepContext
from ..logging._bus import LoggerBus
from ..registry import get as _registry_get
from ..utils.accelerate import build_accelerator
from ..utils.run_dir import make_run_dir, slugify
from ..utils.seed import seed_everything

# Keystone migration (step 2): trainer names removed in favour of the single
# ``preference`` trainer + the ``loss:`` seam. Resolved to a clear error below.
_REMOVED_PREFERENCE_TRAINERS = frozenset({"dpo", "ipo", "simpo", "orpo", "kto"})


# Eager registry population now lives in ``config/_components.py`` (auto-discovery
# over a curated package list, invoked at the ``load_config`` chokepoint). Kept
# under the old private name for the non-load_config entry point (this module's
# ``setup_run_from_config`` RootConfig branch) and for callers/tests that import it.
_eager_import_components = import_all_components

# ``_to_dict`` (the pydantic/Mapping → dict coercion) is imported from
# ``config._resolver`` at the top of this file — its many internal callers below
# and ``cli/_produce.py`` keep using the name unchanged.


def _build_model(cfg: RootConfig) -> Any:
    """Build the PRIMARY trainable model from any declaration form
    (``model:``+``model_profiles:`` or a ``models:`` set). Thin wrapper over the
    single source of truth ``build_primary_model`` — used by ``dry-run --build``,
    ``export`` and ``produce-artifact`` (which only need the one model). For the
    full multi-model set use ``normalize_model_set`` directly."""
    return build_primary_model(cfg)[0]


def _build_judge(cfg: RootConfig) -> Any | None:
    spec = _to_dict(getattr(cfg, "judge", None))
    return _resolve(spec, category="judge") if spec else None


def _inject_allow_stale_artifact(spec: dict[str, Any]) -> None:
    """Recursively set ``allow_stale_artifact=True`` on artifact_joined dataset
    specs and on each entry of their ``join`` list, but only where the user
    did not already make an explicit choice."""
    if not isinstance(spec, dict):
        return
    dataset = spec.get("dataset")
    if isinstance(dataset, dict):
        if dataset.get("name") == "artifact_joined":
            dataset.setdefault("allow_stale_artifact", True)
            joins = dataset.get("join")
            if isinstance(joins, list):
                for j in joins:
                    if isinstance(j, dict):
                        j.setdefault("allow_stale_artifact", True)
        # Nested base (e.g. artifact_joined wrapping artifact_joined) — recurse.
        _inject_allow_stale_artifact(dataset)


def _build_data(
    cfg: RootConfig,
    *,
    run_dir: Path | None = None,
    console: Any | None = None,
    allow_stale_artifact: bool = False,
) -> Any:
    spec = _to_dict(cfg.data)
    if not spec:
        raise RuntimeError("recipe is missing `data:` section")
    if allow_stale_artifact:
        _inject_allow_stale_artifact(spec)
    prep_spec = _to_dict(cfg.prep_graph) if getattr(cfg, "prep_graph", None) else None

    # Auto-route to prep_graph data_module when:
    #   * cfg.prep_graph is set, AND
    #   * cfg.data.source looks like "prep_graph:<terminal>".
    source = spec.get("source")
    is_prep_ref = isinstance(source, str) and source.startswith("prep_graph:")
    if prep_spec and (is_prep_ref or spec.get("name") == "prep_graph"):
        merged = dict(spec)
        merged.pop("source", None)
        if is_prep_ref and "train" not in merged:
            merged["train"] = source[len("prep_graph:") :]
        merged.setdefault("name", "prep_graph")
        merged["prep_graph"] = prep_spec
        if run_dir is not None:
            merged.setdefault("store_root", str(Path(run_dir) / "prep"))
        merged.setdefault("console", console)
        return _resolve(merged, category="data_module")

    if "name" not in spec and "_target_" not in spec:
        spec = {"name": "simple", **spec}
    return _resolve(spec, category="data_module")


def _build_optimizer(cfg: RootConfig, model: torch.nn.Module) -> Any:
    spec = _to_dict(cfg.optim)
    if not spec:
        raise RuntimeError("recipe is missing `optim:` section")
    wrapper = _resolve(spec, category="optimizer")
    wrapper.build(model)
    return wrapper


def _build_optimizer_for(optim_spec: Any, model: torch.nn.Module) -> Any:
    """Build + bind one optimizer from a resolved spec against ``model``.

    The per-model sibling of ``_build_optimizer`` for the ``models:``/
    ``optimizers:`` set — each trainable entry gets its OWN optimizer bound to
    its OWN parameters (the step-4 pairing watch-point), not a single build.
    """
    spec = _to_dict(optim_spec)
    if not spec:
        raise RuntimeError("optimizer spec is empty")
    wrapper = _resolve(spec, category="optimizer")
    wrapper.build(model)
    return wrapper


# Model/optimizer declaration normalisation (``normalize_model_set`` +
# ``_resolve_entry_spec``) moved to ``config/_models.py`` — the single source of
# truth shared by the CLI runtime, ``lab.estimate`` and ``export``. Imported at
# the top of this file.


def _load_state_dict_into(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load weights from a .safetensors or torch checkpoint into ``model``
    (used for frozen aux models like an OPD teacher)."""
    p = Path(ckpt_path).expanduser()
    if p.is_dir():
        st = p / "model.safetensors"
        pt = p / "model.pt"
        p = st if st.exists() else pt
    if str(p).endswith(".safetensors"):
        from safetensors.torch import load_file

        state: Any = load_file(str(p))
    else:
        state = torch.load(str(p), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    model.load_state_dict(state, strict=False)


def _build_scheduler(cfg: RootConfig, optimizer: Any) -> Any:
    spec = _to_dict(cfg.scheduler)
    if not spec:
        return None
    sched = _resolve(spec, category="scheduler")
    inner = getattr(optimizer, "optimizer", optimizer)
    if hasattr(sched, "attach"):
        sched.attach(inner)
    return sched


def _build_objective(cfg: RootConfig) -> tuple[Any, str]:
    """Build the single canonical training objective.

    Returns ``(objective_or_None, source)`` where ``source`` is:
      - ``"objective"`` — a real ``ObjectiveProfile`` from ``cfg.objective``
        (owns ``prepare_batch`` + loss);
      - ``"loss"`` — a ``LossOnlyObjective`` wrapping ``cfg.loss`` (identity
        prepare; the user wrote a plain loss);
      - ``"none"`` — neither given; the trainer supplies its own default
        (``Trainer.default_objective``). The runtime deliberately does **not**
        inject a universal cross-entropy here — the default belongs to the
        trainer (so RL/preference don't have to type-sniff it back).
    """
    obj_spec = _to_dict(getattr(cfg, "objective", None))
    if obj_spec:
        return _resolve(obj_spec, category="objective"), "objective"
    loss_spec = _to_dict(cfg.loss)
    if loss_spec:
        loss_fn = _resolve(loss_spec, category="loss")
        family = getattr(loss_fn, "loss_family", "generic")
        return LossOnlyObjective(loss_fn, loss_family=family), "loss"
    return None, "none"


def _wire_objective(
    trainer: Any, engine: Any, recipe_objective: Any, source: str, trainer_name: str
) -> Any:
    """Bind the canonical objective to a freshly-built trainer + enforce contract.

    Done *after* construction so ``trainer.default_objective()`` can read fields
    a subclass set in ``__init__`` (e.g. a built RL surrogate loss). Returns the
    final ``loss_fn`` to publish in the bundle (``None`` for inline trainers).
    """
    consumes = bool(getattr(trainer, "consumes_objective", True))
    consumes_prepare = bool(getattr(trainer, "consumes_objective_prepare", True))
    requires = bool(getattr(trainer, "requires_objective", False))

    # Author-declaration sanity (a trainer-class bug, not a recipe error).
    if not consumes and requires:
        raise TypeError(
            f"Trainer class {type(trainer).__name__} declares consumes_objective=False "
            f"with requires_objective=True — illegal combination "
            f"(requires_objective only applies when consumes_objective=True)."
        )

    # Direction ①: recipe provided loss/objective but the trainer is inline.
    if recipe_objective is not None and not consumes:
        if source == "loss":
            raise ConfigError(
                f"trainer `{trainer_name}` has a trainer-owned inline loss; remove `loss:`."
            )
        raise ConfigError(
            f"trainer `{trainer_name}` does not consume the objective seam; remove `objective:`."
        )
    # Direction ②: a real objective (with prepare_batch) given to a trainer that
    # never runs the prepare path → would only half-apply, so reject loudly.
    if source == "objective" and not consumes_prepare:
        raise ConfigError(
            f"trainer `{trainer_name}` does not run objective.prepare_batch; "
            f"use `loss:` for a plain loss, or a trainer that supports the prepare path."
        )

    if not consumes:
        return None  # inline trainer: leave trainer.objective = None, no backfill.

    if recipe_objective is not None:
        trainer.objective = recipe_objective
    elif requires:
        raise ConfigError(
            f"trainer `{trainer_name}` requires an explicit `loss:`/`objective:` "
            f"(no sensible default)."
        )
    else:
        trainer.objective = trainer.default_objective()

    trainer.ctx.loss_fn = trainer.objective
    if engine is not None and hasattr(engine, "loss_fn"):
        engine.loss_fn = trainer.objective
    return trainer.objective


def _build_arch_profile(cfg: RootConfig) -> Any | None:
    """Resolve ``trainer.arch_profile`` to an ``ArchitectureProfile`` object.

    A string (e.g. ``rwkv``) is looked up in the ``architecture`` registry and
    the factory is called; an already-built profile passes through; ``None``
    stays ``None``. An unknown name is a clear ``ConfigError`` (vs. the old
    silent no-op where a bare string never activated the stateful reset path).
    """
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return None
    ap = getattr(trainer_cfg, "arch_profile", None)
    if ap is None or isinstance(ap, ArchitectureProfile):
        return ap
    if isinstance(ap, str):
        try:
            factory = _registry_get("architecture", ap)
        except Exception as exc:  # registry miss
            raise ConfigError(
                f"unknown arch_profile {ap!r} — not registered under 'architecture'. "
                f"Built-ins: transformer, rwkv."
            ) from exc
        return factory()
    raise ConfigError(
        f"trainer.arch_profile must be a registered name or an ArchitectureProfile, "
        f"got {type(ap).__name__}."
    )


def _build_callbacks(cfg: RootConfig) -> list[Any]:
    raw = cfg.callbacks if cfg.callbacks is not None else []
    if isinstance(raw, Mapping):
        raw = [raw]
    out: list[Any] = []
    for entry in raw:
        spec = _to_dict(entry)
        if not spec:
            continue
        out.append(_resolve(spec, category="callback"))
    return out


def _build_logger(cfg: RootConfig, run_dir: Path) -> LoggerBus:
    raw = cfg.logger if cfg.logger is not None else []
    if isinstance(raw, Mapping):
        raw = [raw]
    backends: list[Any] = []
    for entry in raw:
        spec = _to_dict(entry)
        if not spec:
            continue
        # Only inject run_dir for backends that file-log; the registry
        # short-name tells us cheaply.
        name = spec.get("name")
        if name in ("jsonl", "tensorboard", "tb"):
            spec.setdefault("run_dir", str(run_dir))
        backends.append(_resolve(spec, category="logger"))
    return LoggerBus(backends)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _init_parallel(cfg: RootConfig) -> ParallelContext:
    """Return ParallelContext from config.

    When ``cfg.parallel`` is absent or all degrees==1, returns a plain
    single_gpu() context without touching torch.distributed.
    When ``cfg.parallel`` is present with dp/tp/pp > 1, initializes the
    process group and DeviceMesh from the torchrun environment variables.
    """
    par = getattr(cfg, "parallel", None)
    if par is None:
        return ParallelContext.single_gpu()
    # Presence of parallel section doesn't mean we need dist — check degrees.
    dp = int(getattr(par, "dp", 1))
    tp = int(getattr(par, "tp", 1))
    pp = int(getattr(par, "pp", 1))
    ep = int(getattr(par, "ep", 1))
    if dp * tp * pp * ep == 1:
        return ParallelContext.single_gpu()
    return ParallelContext.from_env(par)


def _build_grad_sync_strategy(cfg: RootConfig) -> Any | None:
    """Build a GradSyncStrategy from cfg.parallel.grad_sync, or None for noop.

    Returns None (not a NoopGradSyncStrategy instance) for the single-GPU
    path so existing update-rule code keeps its fast path unchanged.
    """
    par = getattr(cfg, "parallel", None)
    if par is None:
        return None
    grad_sync_cfg = getattr(par, "grad_sync", None)
    if grad_sync_cfg is None:
        return None
    name = str(getattr(grad_sync_cfg, "name", "noop") or "noop")
    if name == "noop":
        return None
    # Import and construct the strategy via the registry.
    strategy_cls = _registry_get("grad_sync_strategy", name)
    kwargs: dict[str, Any] = {}
    if hasattr(grad_sync_cfg, "model_dump"):
        raw = grad_sync_cfg.model_dump()
    elif isinstance(grad_sync_cfg, Mapping):
        raw = dict(grad_sync_cfg)
    else:
        raw = {}
    for k, v in raw.items():
        if k != "name":
            kwargs[k] = v
    return strategy_cls(**kwargs)


def _build_model_parallel_strategy(cfg: RootConfig) -> Any | None:
    """Build a ModelParallelStrategy from cfg.parallel.tensor_parallel, or None.

    Fails loud (``ConfigError``) when parallelism is requested but cannot be
    applied — a missing ``tensor_parallel:`` block, an unregistered strategy, or
    the not-yet-wired ``sp`` / ``ep`` degrees would otherwise silently fall back
    to single-GPU (the user would think they were parallel when they weren't).
    """
    par = getattr(cfg, "parallel", None)
    if par is None:
        return None
    # SP / EP are registered but not wired into the runtime selector (only
    # `tensor_parallel` is applied) — fail loud rather than silently no-op.
    if bool(getattr(par, "sp", False)):
        raise ConfigError(
            "parallel.sp (sequence parallelism) is registered but not yet wired "
            "into the train runtime; remove it (see operations/distributed)."
        )
    if int(getattr(par, "ep", 1)) > 1:
        raise ConfigError(
            "parallel.ep (expert parallelism) is a skeleton not yet wired into "
            "the train runtime; set ep=1 (see operations/distributed)."
        )
    tp = int(getattr(par, "tp", 1))
    if tp <= 1:
        return None
    tp_cfg = getattr(par, "tensor_parallel", None)
    if tp_cfg is None:
        raise ConfigError(
            f"parallel.tp={tp} requests tensor parallelism but no "
            "`parallel.tensor_parallel:` block is configured."
        )
    try:
        strategy_cls = _registry_get("model_parallel_strategy", "tensor_parallel")
    except Exception as exc:  # strategy not registered (plugins not loaded)
        raise ConfigError(
            f"parallel.tp={tp} requested but the `tensor_parallel` "
            "model_parallel_strategy is not registered "
            "(distributed plugins not loaded?)."
        ) from exc
    kwargs: dict[str, Any] = {}
    if hasattr(tp_cfg, "model_dump"):
        kwargs = {k: v for k, v in tp_cfg.model_dump().items() if v is not None}
    return strategy_cls(**kwargs)


def _build_pipeline_schedule(cfg: RootConfig) -> Any | None:
    """Build a PipelineSchedule from cfg.parallel.pipeline, or None when pp<=1.

    Mirrors ``_build_model_parallel_strategy``: fails loud (``ConfigError``) when
    ``parallel.pp > 1`` but the configured schedule can't be resolved or
    constructed. The ``schedule`` key selects the implementation and is dropped
    from the constructor kwargs (it is not a ctor arg for every schedule, e.g.
    ``gpipe``).
    """
    par = getattr(cfg, "parallel", None)
    if par is None or int(getattr(par, "pp", 1)) <= 1:
        return None
    pipeline_cfg = getattr(par, "pipeline", None)
    schedule = str(getattr(pipeline_cfg, "schedule", "1f1b") or "1f1b")
    try:
        ps_cls = _registry_get("pipeline_schedule", schedule)
    except Exception as exc:  # unknown schedule / plugins not loaded
        raise ConfigError(
            f"pipeline schedule {schedule!r} is not registered "
            "(unknown schedule, or distributed plugins not loaded)."
        ) from exc
    ps_kwargs: dict[str, Any] = {}
    if pipeline_cfg is not None and hasattr(pipeline_cfg, "model_dump"):
        ps_kwargs = {k: v for k, v in pipeline_cfg.model_dump().items() if v is not None}
    ps_kwargs.pop("schedule", None)  # selector key, not a ctor arg
    try:
        return ps_cls(**ps_kwargs)
    except Exception as exc:
        raise ConfigError(
            f"pipeline schedule {schedule!r} failed to construct: {exc}"
        ) from exc


def _build_optimizer_factory(cfg: RootConfig):
    """Return a Callable[[nn.Module], optimizer] that grad_sync.prepare can call.

    FSDP requires the optimizer to be built AFTER model wrapping, so the
    factory is lazy — it resolves the spec fresh each time it is called.
    """
    def factory(model: torch.nn.Module) -> Any:
        spec = _to_dict(cfg.optim)
        if not spec:
            raise RuntimeError("recipe is missing `optim:` section")
        wrapper = _resolve(spec, category="optimizer")
        wrapper.build(model)
        return wrapper
    return factory


def _diag_field(cfg: Any, key: str, default: Any) -> Any:
    """Read a nested field from optional ``cfg.diagnostics`` block."""
    diag = getattr(cfg, "diagnostics", None)
    if diag is None:
        return default
    if hasattr(diag, key):
        v = getattr(diag, key)
        return v if v is not None else default
    if isinstance(diag, Mapping):
        return diag.get(key, default)
    return default


def _auto_attach_m4_callbacks(cfg: Any, trainer: Any, existing: list[Any]) -> None:
    """Attach default Failure-first callbacks to ``trainer.bus`` based on
    ``cfg.mode`` and ``cfg.diagnostics`` / ``cfg.invariants`` /
    ``cfg.realtime_control``. Each callback class is added only when the same
    *class name* isn't already on the existing callback list — so a user who
    declared ``- {name: frozen_step}`` keeps full control over its config.

    All auto-attached callbacks import lazily and no-op when their config block
    is empty/missing. Construction failures are surfaced, not swallowed: the
    critical ``InvariantsCallback`` fails loud (re-raises); non-critical
    diagnostics emit a warning and are skipped.
    """
    mode = str(getattr(cfg, "mode", "lab") or "lab")
    bus = getattr(trainer, "bus", None)
    if bus is None:
        return

    have: set[str] = {type(cb).__name__ for cb in existing}

    # InvariantsCallback — always-on; reads cfg.invariants for user-declared
    # invariants and falls back to the default set.
    if "InvariantsCallback" not in have:
        try:
            from ..builtin_plugins.callbacks.invariants import InvariantsCallback

            specs = getattr(cfg, "invariants", None)
            if isinstance(specs, Mapping):
                specs = [specs]
            cb = InvariantsCallback(specs=list(specs) if specs else None)
            bus.add(cb)
            trainer.callbacks.append(cb)
        except Exception as exc:  # critical diagnostic — fail loud
            raise ConfigError(
                f"failed to construct the default InvariantsCallback: {exc}"
            ) from exc

    # FrozenStepCallback — scheduled snapshots in lab mode.
    every = int(_diag_field(cfg, "frozen_step_every", 1000 if mode == "lab" else 0))
    if every > 0 and "FrozenStepCallback" not in have:
        try:
            from ..builtin_plugins.callbacks.builtins.frozen_step import (
                FrozenStepCallback,
            )

            fs_cb = FrozenStepCallback(every=every)
            bus.add(fs_cb)
            trainer.callbacks.append(fs_cb)
        except Exception as exc:  # noqa: BLE001 — non-critical, warn & skip
            warnings.warn(
                f"auto-attach FrozenStepCallback failed, skipping: {exc}",
                stacklevel=2,
            )

    # FileSignalsCallback — file-based runtime knobs in lab.
    rt = getattr(cfg, "realtime_control", None)
    rt_enabled = True if mode == "lab" else False
    if isinstance(rt, Mapping) and "enabled" in rt:
        rt_enabled = bool(rt["enabled"])
    elif rt is not None and hasattr(rt, "enabled"):
        rt_enabled = bool(rt.enabled)
    if rt_enabled and "FileSignalsCallback" not in have:
        try:
            from ..builtin_plugins.realtime_control.file_signals import (
                FileSignalsCallback,
            )

            poll_every = 10
            if isinstance(rt, Mapping):
                poll_every = int(rt.get("poll_every", poll_every))
            elif rt is not None and hasattr(rt, "poll_every"):
                poll_every = int(rt.poll_every or poll_every)
            bus.add(FileSignalsCallback(poll_every=poll_every))
            trainer.callbacks.append(bus.callbacks[-1])
        except Exception as exc:  # noqa: BLE001 — non-critical, warn & skip
            warnings.warn(
                f"auto-attach FileSignalsCallback failed, skipping: {exc}",
                stacklevel=2,
            )

    # CallbackIsolationSink — writes callback_failures.jsonl.
    if "CallbackIsolationSink" not in have:
        try:
            from ..diagnostics.callback_isolation import CallbackIsolationSink

            bus.add(CallbackIsolationSink())
            trainer.callbacks.append(bus.callbacks[-1])
        except Exception as exc:  # noqa: BLE001 — non-critical, warn & skip
            warnings.warn(
                f"auto-attach CallbackIsolationSink failed, skipping: {exc}",
                stacklevel=2,
            )


def _validate_mode_override(mode: str) -> str:
    """Validate a CLI ``--mode`` override against ``RootConfig.mode``'s allowed
    values, deriving them from the schema so the two never drift.

    The ``--mode`` flag assigns ``cfg.mode`` directly on an already-validated
    RootConfig, which Pydantic does NOT re-check (``validate_assignment`` is
    off). Without this guard an invalid mode (e.g. ``--mode bogus``) is silently
    accepted and written into the run's config snapshot, which then fails its
    own ``Literal`` schema on reload/resume.
    """
    allowed = get_args(RootConfig.model_fields["mode"].annotation)
    if mode not in allowed:
        raise ConfigError(
            f"--mode {mode!r} is not a valid mode; choose one of "
            f"{list(allowed)}."
        )
    return mode


# ---------------------------------------------------------------------------
# setup_run_from_config stages — extracted from the former ~430-line function.
# Kept module-level (not nested) so tests that ``monkeypatch.setattr(_runtime,
# <name>, ...)`` still bind, and so each phase is independently testable.
# ---------------------------------------------------------------------------


def _resolve_config(
    config: str | Path | RootConfig,
    overrides: list[str] | None,
    mode: str | None,
) -> tuple[RootConfig, Path | None, str, str]:
    """Parse config (path or RootConfig), apply mode, seed, snapshot YAMLs."""
    # Unconditional: covers the RootConfig branch that bypasses load_config; the
    # path branch's load_config also does it, but the _DONE guard makes the
    # second call ~free. Order: built-ins here, user_modules below.
    import_all_components()
    if isinstance(config, (str, Path)):
        path = Path(config)
        snapshot_yaml = path.read_text(encoding="utf-8")
        # load_config is the chokepoint — it already imports cfg.user_modules.
        loaded = load_config(path, overrides=overrides or [])
        config_path: Path | None = path
    else:
        if not isinstance(config, RootConfig):
            raise TypeError(
                "config must be a str/Path to a YAML file or a parsed "
                f"RootConfig; got {type(config).__name__}."
            )
        if overrides:
            raise ValueError(
                "Cannot apply `overrides` to an already-parsed RootConfig. "
                "Pass a config path instead, or apply overrides via "
                "load_config() before calling setup_run_from_config()."
            )
        loaded = config
        config_path = None
        snapshot_yaml = OmegaConf.to_yaml(OmegaConf.create(config.model_dump()))
        # This branch bypasses load_config: a RootConfig may have been hand-built
        # without going through the chokepoint, so import its user_modules here
        # (idempotent — free if load_config already ran).
        _import_user_modules(list(getattr(config, "user_modules", None) or []))
    # setup_run_from_config always validates (load_config default validate=True),
    # so the result is a RootConfig; narrow it for the typed _build_* helpers.
    assert isinstance(loaded, RootConfig)
    cfg = loaded
    if mode is not None:
        # _validate_mode_override returns a validated "lab"/"prod" (typed str);
        # cfg.mode is a Literal and validate_assignment is off (see that helper).
        cfg.mode = _validate_mode_override(mode)  # type: ignore[assignment]
    seed_everything(int(cfg.seed))
    resolved_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump()))
    return cfg, config_path, snapshot_yaml, resolved_yaml


def _preflight_parallel(cfg: RootConfig) -> tuple[Any | None, Any | None]:
    """Validate TP/PP config *before* any run-dir side effects (fail clean).

    Resolving here (rather than at apply time) means the precise ConfigError
    beats _init_parallel's generic "RANK expected" when running without a
    launcher, and a pure-config error never pollutes ``runs/``.
    """
    return _build_model_parallel_strategy(cfg), _build_pipeline_schedule(cfg)


def _prepare_run_dir(
    cfg: RootConfig,
    *,
    config_path: Path | None,
    snapshot_yaml: str,
    resolved_yaml: str,
    existing_run_dir: Path | None,
) -> Path:
    """Create (or reuse) the run dir and capture a best-effort code snapshot."""
    if existing_run_dir is not None:
        run_dir = Path(existing_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"existing_run_dir {run_dir} does not exist")
        # Don't rewrite snapshot/resolved/env — original run owns them. Resume
        # is supposed to be additive.
        return run_dir
    run_dir = make_run_dir(
        cfg.run_root,
        cfg.exp,
        slug=slugify(cfg.exp),
        snapshot_yaml=snapshot_yaml,
        resolved_yaml=resolved_yaml,
        extra_env={
            "lighttrain_version": __version__,
            "config_path": (
                str(config_path) if config_path is not None
                else "<in-memory RootConfig>"
            ),
        },
    )
    # Code snapshot: best-effort; failures degrade to writing a plain
    # ``code_snapshot_pointer.txt`` pointing to run_dir.
    try:
        from ..utils.code_snapshot import capture_code_snapshot

        user_mods = getattr(cfg, "user_modules", None) or []
        capture_code_snapshot(run_dir, user_modules=list(user_mods))
    except Exception:  # noqa: BLE001 — must never block a training start
        import warnings

        warnings.warn(
            "code snapshot failed (see logs); continuing without it", stacklevel=2
        )
    return run_dir


def _open_lineage_store(run_dir: Path) -> Any:
    """Per-run SQLite lineage store (soft dependency; None on failure)."""
    try:
        from ..lineage.store import LineageStore as _LineageStore

        return _LineageStore(run_dir / "lineage.sqlite")
    except Exception:
        return None


def _require_optim_spec(entry: dict[str, Any], optimizers_cfg: Any) -> Any:
    """Resolve the optimizer spec for a trainable entry; raise if missing."""
    spec = optim_spec_for(entry, optimizers_cfg)
    if spec is None:
        raise RuntimeError(
            "recipe is missing an optimizer for a trainable model "
            "(declare `optim:` or `optimizers:` and reference it via the "
            "entry's `optimizer:` field)."
        )
    return spec


def _build_primary_model(
    cfg: RootConfig,
    primary_entry: dict[str, Any],
    *,
    mp_strategy: Any | None,
    pipeline_schedule: Any | None,
    parallel_ctx: ParallelContext,
) -> Any:
    """Phase B/C: build the primary model, TP/SP/EP surgery, then PP split."""
    # Phase B: model surgery (TP/SP/EP) — must run on bare model before FSDP/DDP.
    model = _resolve(primary_entry["spec"], category="model")
    if mp_strategy is not None:
        try:
            model = mp_strategy.apply(model, parallel_ctx)
        except ConfigError:
            raise
        except Exception as exc:  # apply failed (e.g. missing device_mesh)
            raise ConfigError(f"tensor-parallel apply failed: {exc}") from exc
    # Phase C: pipeline splitting (PP) — after TP surgery, before DP wrap. PP is
    # fail-loud: a requested pp>1 that cannot be applied raises ConfigError.
    if pipeline_schedule is not None:
        try:
            model = pipeline_schedule.prepare(model, parallel_ctx)
        except ConfigError:
            raise
        except Exception as exc:  # prepare failed
            raise ConfigError(
                f"pipeline parallel (pp={int(getattr(cfg.parallel, 'pp', 1))}) "
                f"failed to prepare: {exc}"
            ) from exc
    return model


def _build_trainable_core(
    cfg: RootConfig,
    model: Any,
    *,
    primary_optim_spec: Any,
    n_trainable: int,
    parallel_ctx: ParallelContext,
    device: Any,
    run_dir: Path,
    allow_stale_artifact: bool,
) -> tuple[Any, Any, Any, Any, Any | None]:
    """Phase D: grad-sync wrap (DDP/FSDP/ZeRO) + data + optimizer + scheduler."""
    grad_sync = _build_grad_sync_strategy(cfg)
    if grad_sync is not None and n_trainable > 1:
        raise ConfigError(
            "multiple trainable models + a gradient-sync strategy (distributed "
            "Axis-B) is not supported yet; multi-optimizer training is single-GPU "
            "for now."
        )
    if grad_sync is not None:
        def optimizer_factory(m: torch.nn.Module) -> Any:
            return _build_optimizer_for(primary_optim_spec, m)

        data_module = _build_data(
            cfg, run_dir=run_dir, allow_stale_artifact=allow_stale_artifact
        )
        _raw_loader = data_module.train_loader()
        model, optimizer, _loader = grad_sync.prepare(
            model, optimizer_factory, _raw_loader, parallel_ctx, device=device
        )
        # scheduler is built against the (possibly-wrapped) optimizer
        scheduler = _build_scheduler(cfg, optimizer)
    else:
        model = model.to(device)
        data_module = _build_data(
            cfg, run_dir=run_dir, allow_stale_artifact=allow_stale_artifact
        )
        optimizer = _build_optimizer_for(primary_optim_spec, model)
        scheduler = _build_scheduler(cfg, optimizer)
    return model, optimizer, scheduler, data_module, grad_sync


def _build_aux_models(
    models_cfg: dict[str, Any],
    optimizers_cfg: Any,
    *,
    primary_name: str,
    primary_model: Any,
    primary_optimizer: Any,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build frozen aux models (Axis A) + extra trainable models (Axis B)."""
    models: dict[str, Any] = {primary_name: primary_model}
    optimizers: dict[str, Any] = {primary_name: primary_optimizer}
    for _name, _entry in models_cfg.items():
        if _name == primary_name:
            continue
        _aux = _resolve(_entry["spec"], category="model").to(device)
        if _entry["checkpoint"]:
            _load_state_dict_into(_aux, str(_entry["checkpoint"]))
        if _entry["trainable"]:
            optimizers[_name] = _build_optimizer_for(
                _require_optim_spec(_entry, optimizers_cfg), _aux
            )
        else:
            for _p in _aux.parameters():
                _p.requires_grad_(False)
            _aux.eval()
        models[_name] = _aux
    return models, optimizers


def _build_update_rule(cfg: RootConfig) -> Any:
    """Build the update rule (StandardUpdateRule fast-path or registry)."""
    update_rule_spec = (
        cfg.engine.update_rule if cfg.engine is not None else {"name": "standard"}
    )
    if isinstance(update_rule_spec, Mapping) and update_rule_spec.get("name") == "standard":
        accumulate = cfg.trainer.accumulate if cfg.trainer is not None else 1
        grad_clip = cfg.trainer.grad_clip if cfg.trainer is not None else 1.0
        return StandardUpdateRule(
            grad_clip=float(grad_clip),
            accumulate_grad_batches=int(accumulate),
        )
    return _resolve(dict(update_rule_spec), category="update_rule")


def _build_accelerator_from_cfg(cfg: RootConfig) -> Any:
    """AMP / Accelerator. ``mixed_precision: 'no'`` keeps the no-AMP raw path."""
    mp = cfg.engine.mixed_precision if cfg.engine is not None else "no"
    accumulate = cfg.trainer.accumulate if cfg.trainer is not None else 1
    return build_accelerator(str(mp), gradient_accumulation_steps=int(accumulate))


def _build_engine(
    cfg: RootConfig,
    *,
    update_rule: Any,
    loss_fn: Any,
    accelerator: Any,
) -> Any:
    """Build the engine. ``standard`` is direct; others dispatch via registry."""
    engine_name = cfg.engine.name if cfg.engine is not None else "standard"
    if engine_name == "standard":
        return StandardEngine(
            update_rule=update_rule,
            loss_fn=loss_fn,
            accelerator=accelerator,
        )
    engine_cls = _registry_get("engine", engine_name)
    engine_kwargs: dict[str, Any] = {
        "update_rule": update_rule,
        "loss_fn": loss_fn,
        "accelerator": accelerator,
    }
    # Forward every non-name field of cfg.engine as a kwarg (e.g. resident_layers
    # / prefetch / storage / pin_memory). Unknown kwargs raise TypeError.
    if cfg.engine is not None:
        engine_raw = cfg.engine.model_dump() if hasattr(cfg.engine, "model_dump") else dict(cfg.engine)
        for k, v in engine_raw.items():
            if k in ("name", "mixed_precision", "update_rule"):
                continue
            engine_kwargs.setdefault(k, v)
    return engine_cls(**engine_kwargs)


def _build_trainer(
    cfg: RootConfig,
    *,
    engine: Any,
    data_module: Any,
    optimizer: Any,
    scheduler: Any,
    callbacks: list[Any],
    logger: LoggerBus,
    ckpt_manager: Any,
    model: Any,
    models: dict[str, Any],
    optimizers: dict[str, Any],
    device: Any,
) -> tuple[Any, str]:
    """Resolve + construct the trainer (kwarg forwarding, reward adapter, sig filter)."""
    trainer_name = cfg.trainer.name if cfg.trainer is not None else "pretrain"
    # Keystone migration: the five offline-preference trainers collapsed into one
    # ``preference`` trainer; the algorithm is now the ``loss:`` seam.
    if trainer_name in _REMOVED_PREFERENCE_TRAINERS:
        raise ConfigError(
            f"trainer `{trainer_name}` was removed. The offline-preference trainers "
            "(dpo/ipo/simpo/orpo/kto) are now one `preference` trainer; select the "
            f"algorithm via the loss seam:\n"
            f"    trainer: {{ name: preference, ... }}\n"
            f"    loss:    {{ name: {trainer_name}, ... }}   # move beta/gamma/lam here"
        )
    trainer_cls = _registry_get("trainer", trainer_name)

    trainer_kwargs: dict[str, Any] = {
        "engine": engine,
        "data_module": data_module,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "callbacks": callbacks,
        "logger": logger,
        "ckpt_manager": ckpt_manager,
        "max_steps": int(cfg.trainer.max_steps) if cfg.trainer else 1000,
        "val_every": int(cfg.trainer.val_every) if cfg.trainer else 0,
        "ckpt_every": int(cfg.trainer.ckpt_every) if cfg.trainer else 500,
        "log_every": int(cfg.trainer.log_every) if cfg.trainer else 50,
        "model": model,
        "models": models,
        "optimizers": optimizers,
        "device": device,
    }

    # Forward trainer-specific recipe fields (rollout_steps, ppo_epochs, beta, …).
    # ``grad_clip`` is forwarded; the signature filter drops it for trainers that
    # don't declare it. ``accumulate`` stays runtime-only (engine-layer field).
    _RUNTIME_ONLY = {
        "name", "max_steps", "val_every", "ckpt_every", "log_every", "accumulate",
    }
    if cfg.trainer is not None:
        trainer_raw = (
            cfg.trainer.model_dump()
            if hasattr(cfg.trainer, "model_dump")
            else dict(cfg.trainer)
        )
        for k, v in trainer_raw.items():
            if k not in _RUNTIME_ONLY:
                trainer_kwargs.setdefault(k, v)
        # Resolve trainer.arch_profile (str → ArchitectureProfile).
        _arch_profile = _build_arch_profile(cfg)
        if _arch_profile is not None:
            trainer_kwargs["arch_profile"] = _arch_profile

    # Build judge and wrap as a tensor-aware reward_fn for RL trainers, via a
    # registrable judge->reward adapter (F2).
    judge = _build_judge(cfg)
    if judge is not None and trainer_name in ("ppo", "grpo"):
        from ..config._exceptions import ConfigResolveError as _ConfigResolveError

        reward_kind = getattr(judge, "reward_kind", "pointwise")
        adapter_spec = _to_dict(getattr(cfg, "reward_adapter", None)) or {"name": reward_kind}
        try:
            adapter = _resolve(
                {**adapter_spec, "judge": judge, "tokenizer": data_module.tokenizer},
                category="reward_adapter",
            )
        except Exception as exc:  # registry miss / construction error
            raise _ConfigResolveError(
                f"no usable reward_adapter for judge {type(judge).__name__!r} "
                f"(reward_kind={reward_kind!r}): {exc}. Register a "
                f"'{reward_kind}' reward_adapter, or set `reward_adapter:` in the "
                f"recipe. (A 'pairwise' adapter is a deferred feature — see "
                f"lighttrain/builtin_plugins/rl/reward_adapters.py.)"
            ) from exc
        trainer_kwargs["reward_fn"] = adapter

    # VAR_KEYWORD detection: trainers with **kwargs (DPO/ORPO/…) accept all
    # remaining trainer_kwargs; trainers without (PPO/GRPO) are signature-filtered.
    _sig = inspect.signature(trainer_cls.__init__)
    _has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in _sig.parameters.values()
    )
    if _has_var_kw:
        trainer = trainer_cls(**trainer_kwargs)
    else:
        _accepted = set(_sig.parameters) - {"self"}
        trainer = trainer_cls(**{k: v for k, v in trainer_kwargs.items()
                                 if k in _accepted})
    return trainer, trainer_name


def _wire_trainer_context(
    trainer: Any,
    *,
    model: Any,
    engine: Any,
    recipe_objective: Any,
    obj_source: str,
    trainer_name: str,
    accelerator: Any,
    lineage_store: Any,
    run_dir: Path,
    cfg: RootConfig,
    parallel_ctx: ParallelContext,
    grad_sync: Any | None,
    callbacks: list[Any],
) -> Any:
    """Wire ctx components the trainer didn't take in __init__; returns loss_fn."""
    ctx: StepContext = trainer.ctx
    ctx.model = model
    # Bind the canonical objective to the trainer (resolving its default
    # post-construction) and enforce the consume/require contract both ways.
    loss_fn = _wire_objective(trainer, engine, recipe_objective, obj_source, trainer_name)
    ctx.accelerator = accelerator
    ctx.lineage_store = lineage_store
    ctx.run_id = run_dir.name
    # Diagnostics callbacks (nan_hunter, frozen_step, crash_bundle, …) read these.
    ctx.run_dir = run_dir
    ctx.mode = str(getattr(cfg, "mode", "lab") or "lab")
    # Distributed fields — always set; single-GPU uses the defaults.
    ctx.parallel_ctx = parallel_ctx
    ctx.grad_sync = grad_sync
    # Stash run_dir on the trainer so LineageRecorderCallback can read it.
    try:
        trainer._run_dir = run_dir  # type: ignore[attr-defined]
    except Exception:
        pass
    # Auto-attach default lab-mode callbacks (land at the *end* of the list so
    # user-declared ones still see their events first).
    _auto_attach_m4_callbacks(cfg, trainer, callbacks)
    return loss_fn


def setup_run_from_config(
    config: str | Path | RootConfig,
    *,
    overrides: list[str] | None = None,
    mode: str | None = None,
    print_config_only: bool = False,
    existing_run_dir: Path | None = None,
    allow_stale_artifact: bool = False,
) -> dict[str, Any]:
    """Load+validate config, build run dir, instantiate components.

    ``config`` may be either a config path (``str``/``Path``) or an
    already-parsed ``RootConfig``. The path form is the common case (CLI);
    the RootConfig form lets the programmatic API skip the redundant
    ``load_config`` step (Issues #2, #10). ``overrides`` is only valid with
    the path form — pass them via ``load_config`` first if you already have
    a RootConfig.

    When ``existing_run_dir`` is given (used by ``lighttrain resume``), no new
    run dir is created — all I/O (logs / checkpoints / lineage.sqlite) targets
    that directory so the resumed run remains a single self-consistent unit.

    ``allow_stale_artifact`` propagates the CLI flag down to any
    ``artifact_joined`` dataset spec / join entries that didn't make an explicit
    choice.

    Returns a dict with keys: cfg, run_dir, model, data, optimizer,
    scheduler, loss_fn, callbacks, logger, ckpt_manager, engine, accelerator,
    trainer.  Caller decides whether to ``trainer.fit()``.
    """
    cfg, config_path, snapshot_yaml, resolved_yaml = _resolve_config(config, overrides, mode)
    if print_config_only:
        return {"cfg": cfg, "resolved_yaml": resolved_yaml}

    # Parallel-config preflight runs *before* any run-dir side effects so a
    # pure-config error never pollutes ``runs/``.
    mp_strategy, pipeline_schedule = _preflight_parallel(cfg)
    run_dir = _prepare_run_dir(
        cfg,
        config_path=config_path,
        snapshot_yaml=snapshot_yaml,
        resolved_yaml=resolved_yaml,
        existing_run_dir=existing_run_dir,
    )

    # Phase A: distributed topology (single_gpu() when cfg.parallel is absent).
    parallel_ctx = _init_parallel(cfg)
    device = parallel_ctx.local_device
    lineage_store = _open_lineage_store(run_dir)

    # Normalise model/optimizer declaration into the internal set form (a lone
    # model:/optim: desugars to a one-entry set).
    models_cfg, optimizers_cfg = normalize_model_set(cfg)
    _primary_name, _primary_entry = primary_trainable(models_cfg)
    _n_trainable = sum(1 for e in models_cfg.values() if e["trainable"])

    model = _build_primary_model(
        cfg,
        _primary_entry,
        mp_strategy=mp_strategy,
        pipeline_schedule=pipeline_schedule,
        parallel_ctx=parallel_ctx,
    )
    _primary_optim = _require_optim_spec(_primary_entry, optimizers_cfg)
    model, optimizer, scheduler, data_module, grad_sync = _build_trainable_core(
        cfg,
        model,
        primary_optim_spec=_primary_optim,
        n_trainable=_n_trainable,
        parallel_ctx=parallel_ctx,
        device=device,
        run_dir=run_dir,
        allow_stale_artifact=allow_stale_artifact,
    )
    models, optimizers = _build_aux_models(
        models_cfg,
        optimizers_cfg,
        primary_name=_primary_name,
        primary_model=model,
        primary_optimizer=optimizer,
        device=device,
    )

    # The single canonical seam. ``recipe_objective`` may be None (neither
    # loss: nor objective: given) — the trainer's ``default_objective()`` then
    # supplies it, resolved post-construction in ``_wire_objective``. The engine
    # is built with this (possibly None) loss_fn and re-wired after the trainer.
    recipe_objective, _obj_source = _build_objective(cfg)
    loss_fn = recipe_objective
    callbacks = _build_callbacks(cfg)
    logger = _build_logger(cfg, run_dir)
    ckpt_manager = CheckpointManager(run_dir)

    update_rule = _build_update_rule(cfg)
    accelerator = _build_accelerator_from_cfg(cfg)

    engine = _build_engine(
        cfg, update_rule=update_rule, loss_fn=loss_fn, accelerator=accelerator
    )

    trainer, trainer_name = _build_trainer(
        cfg,
        engine=engine,
        data_module=data_module,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        logger=logger,
        ckpt_manager=ckpt_manager,
        model=model,
        models=models,
        optimizers=optimizers,
        device=device,
    )

    loss_fn = _wire_trainer_context(
        trainer,
        model=model,
        engine=engine,
        recipe_objective=recipe_objective,
        obj_source=_obj_source,
        trainer_name=trainer_name,
        accelerator=accelerator,
        lineage_store=lineage_store,
        run_dir=run_dir,
        cfg=cfg,
        parallel_ctx=parallel_ctx,
        grad_sync=grad_sync,
        callbacks=callbacks,
    )

    return {
        "cfg": cfg,
        "resolved_yaml": resolved_yaml,
        "run_dir": run_dir,
        "model": model,
        "data": data_module,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "callbacks": callbacks,
        "logger": logger,
        "ckpt_manager": ckpt_manager,
        "engine": engine,
        "accelerator": accelerator,
        "trainer": trainer,
        "device": device,
        "lineage_store": lineage_store,
        "parallel_ctx": parallel_ctx,
        "grad_sync": grad_sync,
    }


__all__ = ["build_prep_runner", "setup_run_from_config"]


def build_prep_runner(
    config_path: Path,
    *,
    overrides: list[str] | None = None,
    store_root: str | Path | None = None,
    workers: int = 1,
    console: Any | None = None,
    pool_kind: str = "thread",
) -> dict[str, Any]:
    """Build a :class:`PrepRunner` from a recipe's ``prep_graph:`` block.

    ``pool_kind`` chooses between ``thread`` and ``process`` executors for
    in-layer parallelism; defaults to ``thread`` (IO-bound work). ``process``
    is the right choice for CPU-bound nodes (heavy
    tokenization, large-format encoding) provided the nodes are pickle-safe.

    Returns a dict ``{cfg, runner, graph, store_root}``. Used by the
    ``prep`` family of CLI commands.
    """
    from ..prepgraph.dag import PrepGraph
    from ..prepgraph.runner import PrepRunner

    # load_config below populates the registry (register_components default True).
    cfg = load_config(config_path, overrides=overrides or [])
    prep_spec = _to_dict(cfg.prep_graph)
    if not prep_spec:
        raise RuntimeError(
            f"recipe {config_path} has no `prep_graph:` section"
        )
    graph = PrepGraph.from_config(prep_spec)
    if store_root is None:
        store_root = Path(cfg.run_root) / cfg.exp / "prep"
    # Recipe-level pool_kind takes precedence over the CLI default if set.
    pool_kind = str(prep_spec.get("pool_kind") or pool_kind)
    runner = PrepRunner(
        graph,
        store_root=store_root,
        workers=workers,
        console=console,
        pool_kind=pool_kind,  # type: ignore[arg-type]
    )
    return {
        "cfg": cfg,
        "graph": graph,
        "runner": runner,
        "store_root": Path(store_root),
    }
