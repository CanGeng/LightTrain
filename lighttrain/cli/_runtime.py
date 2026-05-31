"""CLI runtime helpers — config → run dir → components → trainer wiring.

Keeps ``cli/_app.py`` thin: every train-style command boils down to
``setup_run_from_config(...)`` + ``trainer.fit()``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, get_args

import torch
from omegaconf import OmegaConf

from .. import __version__
from ..checkpoint.manager import CheckpointManager
from ..config import ConfigError, RootConfig, load_config
from ..config._resolver import resolve as _resolve
from ..config._resolver import select_model_spec
from ..distributed._context import ParallelContext
from ..engine._context import StepContext
from ..engine.standard import StandardEngine
from ..logging._bus import LoggerBus
from ..registry import get as _registry_get
from ..update_rules.standard import StandardUpdateRule
from ..utils.accelerate import build_accelerator
from ..utils.run_dir import make_run_dir, slugify
from ..utils.seed import seed_everything


# ``user_modules`` import now lives in ``config/`` and is invoked at the
# ``load_config`` chokepoint (so every recipe-eating command gets it for free).
# Re-exported here under the old private name for backward compatibility with
# callers/tests that import it from ``cli._runtime``; both names share the one
# process-wide dedup set.
from ..config._user_modules import _IMPORTED_USER_MODULES  # noqa: F401
from ..config._user_modules import import_user_modules as _import_user_modules

# Keystone migration (step 2): trainer names removed in favour of the single
# ``preference`` trainer + the ``loss:`` seam. Resolved to a clear error below.
_REMOVED_PREFERENCE_TRAINERS = frozenset({"dpo", "ipo", "simpo", "orpo", "kto"})


def _eager_import_components() -> None:
    """Force-import every module that uses ``@register`` so the registry
    is fully populated before we resolve recipe specs."""
    # Models
    from ..models import adapters as _models_adapters  # noqa: F401
    from ..models.adapters import tiny_lm as _tiny_lm  # noqa: F401
    # Data core
    from ..data.core import _module as _dm  # noqa: F401
    from ..data.core import _prep_module as _pdm  # noqa: F401
    from ..data.core import collators as _coll  # noqa: F401
    from ..data.core import datasets as _ds  # noqa: F401
    from ..data.core import samplers as _samp  # noqa: F401
    from ..data.core import tokenizers as _tok  # noqa: F401
    # Multimodal data
    from ..data import collators as _mm_coll  # noqa: F401
    from ..data import processors as _proc  # noqa: F401
    from ..data import samplers as _samp2  # noqa: F401
    # PrepGraph nodes
    from ..prepgraph import nodes as _nodes  # noqa: F401
    # Optimizers / schedulers / losses
    from ..losses import core as _loss_core  # noqa: F401
    from ..losses import distill as _loss_distill  # noqa: F401
    # Preference losses (dpo/ipo/simpo/orpo/kto/bradley_terry) — the ``loss:``
    # seam for the collapsed preference trainer. Previously registered via the
    # now-deleted per-algorithm trainer shells; register them explicitly.
    from ..losses import preference as _loss_pref  # noqa: F401
    # RL value heads + judge->reward adapters (registry categories value_head /
    # reward_adapter) — resolved from recipes by the RL trainers / runtime.
    from ..rl import reward_adapters as _reward_adapters  # noqa: F401
    from ..rl import value_heads as _value_heads  # noqa: F401
    from ..optim import schedulers as _sched  # noqa: F401
    from ..optim import wrappers as _opt  # noqa: F401
    # Logging backends
    from ..logging.backends import console as _con  # noqa: F401
    from ..logging.backends import jsonl as _jsonl  # noqa: F401
    try:
        from ..logging.backends import tb as _tb  # noqa: F401
    except Exception:  # pragma: no cover — tensorboard optional
        pass
    # Callbacks
    from ..callbacks import builtins as _cbs  # noqa: F401
    # Trainers
    from ..trainers import pretrain as _ptr  # noqa: F401
    # Eval (judges)
    from ..eval import judge as _judge_mod  # noqa: F401
    # Artifacts + lineage
    from ..artifacts import producer as _ap  # noqa: F401
    from ..artifacts import store as _as  # noqa: F401
    from ..artifacts import joined_dataset as _ajd  # noqa: F401
    from ..artifacts import dynamic_producer as _adp  # noqa: F401
    # Failure-first callbacks (invariants / diagnostics / realtime_control).
    # Each module is allowed to fail import silently; the eager import
    # ensures their @register decorators run when available.
    try:
        from .. import invariants as _inv  # noqa: F401
        from ..callbacks import invariants as _cb_inv  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..callbacks.builtins import frozen_step as _fs_cb  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..diagnostics import nan_hunter as _nh  # noqa: F401
        from ..diagnostics import dead_neuron as _dn  # noqa: F401
        from ..diagnostics import grad_flow as _gf  # noqa: F401
        from ..diagnostics import sample_preview as _sp  # noqa: F401
        from ..diagnostics import loss_attribution as _la  # noqa: F401
        from ..diagnostics import callback_isolation as _ci  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..realtime_control import file_signals as _rfs  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    # PEFT adapters. peft is an opt-in extra; import lazily so a recipe
    # that doesn't use LoRA / IA3 / QLoRA still works without the package.
    try:
        from ..models import peft as _peft  # noqa: F401
    except ImportError:
        pass
    # Frontier plugins (layer_offload + quant + distributed). Same opt-in pattern.
    try:
        import plugins.layer_offload as _lo_plugin  # noqa: F401
    except ImportError:
        pass
    try:
        import plugins.quant as _quant_plugin  # noqa: F401
    except ImportError:
        pass
    try:
        import plugins.distributed as _dist_plugin  # noqa: F401
    except ImportError:
        pass


def _to_dict(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    if isinstance(spec, Mapping):
        return dict(spec)
    return spec  # already concrete


def _build_model(cfg: RootConfig) -> Any:
    # v0.1.8: ``model`` is a string selector into ``model_profiles``; a bare
    # dict ``model:`` is rejected by select_model_spec with a migration hint.
    spec = select_model_spec(getattr(cfg, "model", None), getattr(cfg, "model_profiles", None))
    return _resolve(spec, category="model")


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


def _resolve_entry_spec(spec: Any, model_profiles: Any) -> dict[str, Any]:
    """Resolve a ``models:`` entry's ``spec`` — either an inline component spec
    or ``{profile: <name>}`` selecting from the top-level ``model_profiles:``
    catalogue (the variant-selection axis is orthogonal to the model-set axis)."""
    spec = _to_dict(spec)
    if "profile" in spec:
        return select_model_spec(spec["profile"], model_profiles)
    return spec


def _desugar_models(cfg: RootConfig) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Normalise the model/optimizer declaration into the internal set form.

    Returns ``(models_cfg, optimizers_cfg)`` where each models entry is
    ``{spec, trainable, optimizer, checkpoint}`` (``spec`` already resolved to a
    component-spec dict) and ``optimizers_cfg`` maps name -> optimizer spec.

    Single entry point (no double code path downstream): a lone ``model:`` +
    ``model_profiles:`` + ``optim:`` desugars to ``{main: {...}}``. Declaring
    both ``model:`` and ``models:`` is a conflict error.
    """
    models = getattr(cfg, "models", None)
    optimizers = getattr(cfg, "optimizers", None)
    has_lone_model = getattr(cfg, "model", None) is not None

    if models is not None and has_lone_model:
        raise ConfigError(
            "recipe sets both `model:` and `models:`. Use `models:` (the model "
            "set) and drop the lone `model:`; variant selection still works via "
            "each entry's `spec: {profile: <name>}` into `model_profiles:`."
        )

    if models is None:
        # Sugar: lone model:/model_profiles:/optim: → a one-entry set.
        spec = select_model_spec(
            getattr(cfg, "model", None), getattr(cfg, "model_profiles", None)
        )
        models_cfg = {
            "main": {
                "spec": dict(spec),
                "trainable": True,
                "optimizer": "main",
                "checkpoint": None,
            }
        }
        optim_spec = _to_dict(cfg.optim)
        optimizers_cfg = {"main": optim_spec} if optim_spec else {}
        return models_cfg, optimizers_cfg

    # Explicit models: set.
    mp = getattr(cfg, "model_profiles", None)
    models_cfg = {}
    for name, raw in _to_dict(models).items():
        entry = _to_dict(raw)
        models_cfg[name] = {
            "spec": _resolve_entry_spec(entry.get("spec"), mp),
            "trainable": bool(entry.get("trainable", True)),
            "optimizer": entry.get("optimizer"),
            "checkpoint": entry.get("checkpoint"),
        }
    if optimizers is not None:
        optimizers_cfg = {k: _to_dict(v) for k, v in _to_dict(optimizers).items()}
    else:
        optim_spec = _to_dict(cfg.optim)
        optimizers_cfg = {"main": optim_spec} if optim_spec else {}
    return models_cfg, optimizers_cfg


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

        state = load_file(str(p))
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


def _build_loss(cfg: RootConfig) -> Any:
    spec = _to_dict(cfg.loss)
    if not spec:
        spec = {"name": "cross_entropy"}
    return _resolve(spec, category="loss")


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
    """Build a ModelParallelStrategy from cfg.parallel.tensor_parallel, or None."""
    par = getattr(cfg, "parallel", None)
    if par is None:
        return None
    tp_cfg = getattr(par, "tensor_parallel", None)
    if tp_cfg is None:
        return None
    tp = int(getattr(par, "tp", 1))
    if tp <= 1:
        return None
    name = str(getattr(tp_cfg, "auto_plan_for", None) or "tensor_parallel")
    try:
        strategy_cls = _registry_get("model_parallel_strategy", "tensor_parallel")
    except Exception:  # strategy not yet registered (plugins not loaded)
        return None
    kwargs: dict[str, Any] = {}
    if hasattr(tp_cfg, "model_dump"):
        kwargs = {k: v for k, v in tp_cfg.model_dump().items() if v is not None}
    return strategy_cls(**kwargs)


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

    All auto-attached callbacks import lazily, no-op when their config block
    is empty/missing, and never raise during construction.
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
            from ..callbacks.invariants import InvariantsCallback

            specs = getattr(cfg, "invariants", None)
            if isinstance(specs, Mapping):
                specs = [specs]
            cb = InvariantsCallback(specs=list(specs) if specs else None)
            bus.add(cb)
            trainer.callbacks.append(cb)
        except Exception:  # noqa: BLE001
            pass

    # FrozenStepCallback — scheduled snapshots in lab mode.
    every = int(_diag_field(cfg, "frozen_step_every", 1000 if mode == "lab" else 0))
    if every > 0 and "FrozenStepCallback" not in have:
        try:
            from ..callbacks.builtins.frozen_step import FrozenStepCallback

            cb = FrozenStepCallback(every=every)
            bus.add(cb)
            trainer.callbacks.append(cb)
        except Exception:  # noqa: BLE001
            pass

    # FileSignalsCallback — file-based runtime knobs in lab.
    rt = getattr(cfg, "realtime_control", None)
    rt_enabled = True if mode == "lab" else False
    if isinstance(rt, Mapping) and "enabled" in rt:
        rt_enabled = bool(rt["enabled"])
    elif hasattr(rt, "enabled"):
        rt_enabled = bool(getattr(rt, "enabled"))
    if rt_enabled and "FileSignalsCallback" not in have:
        try:
            from ..realtime_control.file_signals import FileSignalsCallback

            poll_every = 10
            if isinstance(rt, Mapping):
                poll_every = int(rt.get("poll_every", poll_every))
            elif hasattr(rt, "poll_every"):
                poll_every = int(getattr(rt, "poll_every") or poll_every)
            bus.add(FileSignalsCallback(poll_every=poll_every))
            trainer.callbacks.append(bus.callbacks[-1])
        except Exception:  # noqa: BLE001
            pass

    # CallbackIsolationSink — writes callback_failures.jsonl.
    if "CallbackIsolationSink" not in have:
        try:
            from ..diagnostics.callback_isolation import CallbackIsolationSink

            bus.add(CallbackIsolationSink())
            trainer.callbacks.append(bus.callbacks[-1])
        except Exception:  # noqa: BLE001
            pass


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


def setup_run_from_config(
    config: "str | Path | RootConfig",
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
    _eager_import_components()
    if isinstance(config, (str, Path)):
        config_path: Path | None = Path(config)
        snapshot_yaml = config_path.read_text(encoding="utf-8")
        # load_config is the chokepoint — it already imports cfg.user_modules.
        cfg = load_config(config_path, overrides=overrides or [])
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
        cfg = config
        config_path = None
        snapshot_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump()))
        # This branch bypasses load_config: a RootConfig may have been hand-built
        # without going through the chokepoint, so import its user_modules here
        # (idempotent — free if load_config already ran).
        _import_user_modules(list(getattr(cfg, "user_modules", None) or []))
    if mode is not None:
        cfg.mode = _validate_mode_override(mode)  # type: ignore[union-attr]

    seed_everything(int(cfg.seed))

    resolved_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump()))
    if print_config_only:
        return {"cfg": cfg, "resolved_yaml": resolved_yaml}

    if existing_run_dir is not None:
        run_dir = Path(existing_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"existing_run_dir {run_dir} does not exist")
        # Don't rewrite snapshot/resolved/env — original run owns them. Resume
        # is supposed to be additive.
    else:
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

            warnings.warn("code snapshot failed (see logs); continuing without it")

    # Phase A: distributed topology.
    # _init_parallel returns single_gpu() when cfg.parallel is absent,
    # so all downstream code is topology-agnostic.
    parallel_ctx = _init_parallel(cfg)
    device = parallel_ctx.local_device

    # ----- lineage store --------------------------------------------------
    # Soft dependency. Per-run SQLite only — global aggregate is a
    # documented hook on RootConfig.lineage.global_db.
    lineage_store = None
    try:
        from ..lineage.store import LineageStore as _LineageStore

        lineage_store = _LineageStore(run_dir / "lineage.sqlite")
    except Exception:
        lineage_store = None

    # Normalise model/optimizer declaration into the internal set form. Single
    # entry point — a lone model:/optim: desugars to a one-entry set, so the
    # primary path below is bit-identical for single-model recipes.
    models_cfg, optimizers_cfg = _desugar_models(cfg)
    _trainable_names = [n for n, e in models_cfg.items() if e["trainable"]]
    if not _trainable_names:
        raise ConfigError(
            "recipe defines no trainable model (every `models:` entry is "
            "`trainable: false`)."
        )
    # The "primary" trainable model goes through model surgery / PP / grad_sync
    # and is exposed as ``model=``; any further trainable models (Axis-B —
    # GAN/actor-critic) get their own optimizer on the single-GPU path below.
    _primary_name = _trainable_names[0]
    _primary_entry = models_cfg[_primary_name]

    def _optim_spec_for(entry: dict[str, Any]) -> Any:
        opt_name = entry["optimizer"]
        if opt_name is None:
            opt_name = "main" if "main" in optimizers_cfg else (
                next(iter(optimizers_cfg)) if optimizers_cfg else None
            )
        if opt_name is None or opt_name not in optimizers_cfg:
            raise RuntimeError(
                "recipe is missing an optimizer for a trainable model "
                "(declare `optim:` or `optimizers:` and reference it via the "
                "entry's `optimizer:` field)."
            )
        return optimizers_cfg[opt_name]

    # Phase B: model surgery (TP/SP/EP) — must run on bare model before FSDP/DDP.
    model = _resolve(_primary_entry["spec"], category="model")
    mp_strategy = _build_model_parallel_strategy(cfg)
    if mp_strategy is not None:
        model = mp_strategy.apply(model, parallel_ctx)

    # Phase C: pipeline splitting (PP) — after TP surgery, before DP wrap.
    # PP support requires plugins/distributed/; skip for now when
    # the PipelineSchedule implementation is not yet registered.
    pipeline_schedule = None
    par = getattr(cfg, "parallel", None)
    if par is not None and int(getattr(par, "pp", 1)) > 1:
        try:
            ps_cls = _registry_get("pipeline_schedule", "1f1b")
            pipeline_cfg = getattr(par, "pipeline", None)
            ps_kwargs: dict[str, Any] = {}
            if pipeline_cfg is not None and hasattr(pipeline_cfg, "model_dump"):
                ps_kwargs = {k: v for k, v in pipeline_cfg.model_dump().items() if v is not None}
            pipeline_schedule = ps_cls(**ps_kwargs)
            model = pipeline_schedule.prepare(model, parallel_ctx)
        except Exception:  # PP strategy not yet registered
            pass

    # Phase D: gradient-sync wrap (DDP/FSDP/ZeRO).
    # When grad_sync is None (single-GPU / noop), fall back to the plain path.
    grad_sync = _build_grad_sync_strategy(cfg)
    _primary_optim = _optim_spec_for(_primary_entry)
    if grad_sync is not None and len(_trainable_names) > 1:
        raise ConfigError(
            "multiple trainable models + a gradient-sync strategy (distributed "
            "Axis-B) is not supported yet; multi-optimizer training is single-GPU "
            "for now."
        )
    if grad_sync is not None:
        def optimizer_factory(m: torch.nn.Module) -> Any:
            return _build_optimizer_for(_primary_optim, m)

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
        optimizer = _build_optimizer_for(_primary_optim, model)
        scheduler = _build_scheduler(cfg, optimizer)

    # Build the rest of the model set: frozen aux models (Axis A — teacher/ref/
    # EMA) and any further TRAINABLE models with their own optimizer (Axis B —
    # GAN/actor-critic). The primary's optimizer is keyed by ``optimizers``;
    # each additional trainable entry gets its own (per-entry pairing).
    models: dict[str, Any] = {_primary_name: model}
    optimizers: dict[str, Any] = {_primary_name: optimizer}
    for _name, _entry in models_cfg.items():
        if _name == _primary_name:
            continue
        _aux = _resolve(_entry["spec"], category="model").to(device)
        if _entry["checkpoint"]:
            _load_state_dict_into(_aux, str(_entry["checkpoint"]))
        if _entry["trainable"]:
            optimizers[_name] = _build_optimizer_for(_optim_spec_for(_entry), _aux)
        else:
            for _p in _aux.parameters():
                _p.requires_grad_(False)
            _aux.eval()
        models[_name] = _aux

    loss_fn = _build_loss(cfg)
    callbacks = _build_callbacks(cfg)
    logger = _build_logger(cfg, run_dir)
    ckpt_manager = CheckpointManager(run_dir)

    update_rule_spec = (
        cfg.engine.update_rule if cfg.engine is not None else {"name": "standard"}
    )
    if isinstance(update_rule_spec, Mapping) and update_rule_spec.get("name") == "standard":
        accumulate = cfg.trainer.accumulate if cfg.trainer is not None else 1
        grad_clip = cfg.trainer.grad_clip if cfg.trainer is not None else 1.0
        update_rule = StandardUpdateRule(
            grad_clip=float(grad_clip),
            accumulate_grad_batches=int(accumulate),
        )
    else:
        update_rule = _resolve(dict(update_rule_spec), category="update_rule")

    # AMP / Accelerator. ``mixed_precision: 'no'`` short-circuits to ``None``
    # so the no-AMP path stays raw torch.
    mp = cfg.engine.mixed_precision if cfg.engine is not None else "no"
    accumulate = cfg.trainer.accumulate if cfg.trainer is not None else 1
    accelerator = build_accelerator(
        str(mp), gradient_accumulation_steps=int(accumulate)
    )

    # engine.name dispatches via the registry so plug-ins like
    # ``layer_offload`` can replace the engine without touching the runtime.
    # ``standard`` keeps the direct construction path.
    engine_name = cfg.engine.name if cfg.engine is not None else "standard"
    if engine_name == "standard":
        engine = StandardEngine(
            update_rule=update_rule,
            loss_fn=loss_fn,
            accelerator=accelerator,
        )
    else:
        engine_cls = _registry_get("engine", engine_name)
        engine_kwargs: dict[str, Any] = {
            "update_rule": update_rule,
            "loss_fn": loss_fn,
            "accelerator": accelerator,
        }
        # Forward every non-name field of cfg.engine as a kwarg (e.g.
        # resident_layers / prefetch / storage / pin_memory). The engine
        # constructor decides what to accept; unknown kwargs raise TypeError.
        if cfg.engine is not None:
            engine_raw = cfg.engine.model_dump() if hasattr(cfg.engine, "model_dump") else dict(cfg.engine)
            for k, v in engine_raw.items():
                if k in ("name", "mixed_precision", "update_rule"):
                    continue
                engine_kwargs.setdefault(k, v)
        engine = engine_cls(**engine_kwargs)

    trainer_name = cfg.trainer.name if cfg.trainer is not None else "pretrain"
    # Keystone migration (step 2): the five offline-preference trainers collapsed
    # into one ``preference`` trainer; the algorithm is now the ``loss:`` seam.
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
    # Always exclude grad_clip and accumulate — they are engine-only fields consumed
    # by StandardUpdateRule; no trainer constructor accepts them.
    _RUNTIME_ONLY = {
        "name", "max_steps", "val_every", "ckpt_every", "log_every",
        "grad_clip", "accumulate",
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

    # Build judge and wrap as a tensor-aware reward_fn for RL trainers, via a
    # registrable judge->reward adapter (F2). The judge declares its reward_kind
    # ("pointwise" by default); a recipe `reward_adapter:` overrides it. Any
    # registered pointwise judge can back an RL reward — no isinstance whitelist.
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
                f"lighttrain/rl/reward_adapters.py.)"
            ) from exc
        trainer_kwargs["reward_fn"] = adapter

    # VAR_KEYWORD detection: trainers with **kwargs (DPO/ORPO/…) accept all
    # remaining trainer_kwargs; trainers without (PPO/GRPO) are filtered by
    # their explicit signature.
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

    # Wire ctx components the trainer didn't take in __init__.
    ctx: StepContext = trainer.ctx
    ctx.model = model
    ctx.loss_fn = loss_fn
    ctx.accelerator = accelerator
    ctx.lineage_store = lineage_store
    ctx.run_id = run_dir.name
    # Diagnostics callbacks (nan_hunter, frozen_step, crash_bundle,
    # loss_attribution, file_signals) all read these off ctx.
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

    # Auto-attach default callbacks when running in lab mode and the user
    # hasn't explicitly opted out. The cfg.diagnostics / cfg.invariants /
    # cfg.realtime_control blocks may carry overrides. All auto-attached
    # callbacks land at the *end* of the callback list so user-declared
    # ones still see their events first.
    _auto_attach_m4_callbacks(cfg, trainer, callbacks)

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
    _eager_import_components()
    from ..prepgraph.dag import PrepGraph
    from ..prepgraph.runner import PrepRunner

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
        pool_kind=pool_kind,
    )
    return {
        "cfg": cfg,
        "graph": graph,
        "runner": runner,
        "store_root": Path(store_root),
    }
