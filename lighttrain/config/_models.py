"""Model/optimizer declaration → normalized internal "model set" form.

Single source of truth for turning any model declaration form into one canonical
representation:

* the lone ``model:`` + ``model_profiles:`` + ``optim:`` sugar (single-model), or
* an explicit ``models:`` / ``optimizers:`` named set (Axis-A frozen aux /
  Axis-B multi-trainable).

Shared by the CLI runtime, ``lab.estimate`` and ``export`` so no command
re-parses the model declaration on its own. (The pre-v0.2.x bug class was each
consumer keeping its own parser; this module collapses them.)
"""

from __future__ import annotations

from typing import Any, Mapping

from ._exceptions import ConfigError
from ._resolver import _as_plain_dict
from ._resolver import resolve as _resolve
from ._resolver import select_model_spec


def _field(cfg: Any, key: str) -> Any:
    """Read a top-level config field from either a ``RootConfig`` (attribute
    access) or a plain ``Mapping`` (``lab.estimate``'s public API passes a dict).

    Using this everywhere is a hard rule for this module: no attribute-style
    ``cfg.<field>`` access may remain, or the dict path raises ``AttributeError``.
    """
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _resolve_entry_spec(spec: Any, model_profiles: Any) -> dict[str, Any]:
    """Resolve a ``models:`` entry's ``spec`` — either an inline component spec
    or ``{profile: <name>}`` selecting from the top-level ``model_profiles:``
    catalogue (the variant-selection axis is orthogonal to the model-set axis)."""
    spec = _as_plain_dict(spec)
    if "profile" in spec:
        return select_model_spec(spec["profile"], model_profiles)
    return spec


def normalize_model_set(
    cfg: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Normalise the model/optimizer declaration into the internal set form.

    Returns ``(models_cfg, optimizers_cfg)`` where each models entry is
    ``{spec, trainable, optimizer, checkpoint}`` (``spec`` already resolved to a
    component-spec dict) and ``optimizers_cfg`` maps name -> optimizer spec.

    Single entry point (no double code path downstream): a lone ``model:`` +
    ``model_profiles:`` + ``optim:`` desugars to ``{main: {...}}``. Declaring
    both ``model:`` and ``models:`` is a conflict error.

    Accepts a ``RootConfig`` or a plain ``Mapping`` (the latter is the
    ``lab.estimate`` public-API path).
    """
    models = _field(cfg, "models")
    optimizers = _field(cfg, "optimizers")
    has_lone_model = _field(cfg, "model") is not None

    if models is not None and has_lone_model:
        raise ConfigError(
            "recipe sets both `model:` and `models:`. Use `models:` (the model "
            "set) and drop the lone `model:`; variant selection still works via "
            "each entry's `spec: {profile: <name>}` into `model_profiles:`."
        )

    if models is None:
        # Sugar: lone model:/model_profiles:/optim: → a one-entry set.
        spec = select_model_spec(_field(cfg, "model"), _field(cfg, "model_profiles"))
        models_cfg = {
            "main": {
                "spec": dict(spec),
                "trainable": True,
                "optimizer": "main",
                "checkpoint": None,
            }
        }
        optim_spec = _as_plain_dict(_field(cfg, "optim"))
        optimizers_cfg = {"main": optim_spec} if optim_spec else {}
        return models_cfg, optimizers_cfg

    # Explicit models: set.
    mp = _field(cfg, "model_profiles")
    models_cfg = {}
    for name, raw in _as_plain_dict(models).items():
        entry = _as_plain_dict(raw)
        models_cfg[name] = {
            "spec": _resolve_entry_spec(entry.get("spec"), mp),
            "trainable": bool(entry.get("trainable", True)),
            "optimizer": entry.get("optimizer"),
            "checkpoint": entry.get("checkpoint"),
        }
    if optimizers is not None:
        optimizers_cfg = {k: _as_plain_dict(v) for k, v in _as_plain_dict(optimizers).items()}
    else:
        optim_spec = _as_plain_dict(_field(cfg, "optim"))
        optimizers_cfg = {"main": optim_spec} if optim_spec else {}

    # Validate named optimizer references: a trainable entry naming an optimizer
    # absent from optimizers_cfg is a dangling reference — name it loudly instead
    # of the generic "recipe is missing an optimizer" raised later at build time.
    for name, entry in models_cfg.items():
        opt = entry["optimizer"]
        if entry["trainable"] and opt is not None and opt not in optimizers_cfg:
            raise ConfigError(
                f"models[{name!r}] references optimizer {opt!r} not found in "
                f"optimizers: {sorted(optimizers_cfg)}."
            )
    return models_cfg, optimizers_cfg


def primary_trainable(
    models_cfg: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Return ``(name, entry)`` of the primary (first) trainable model.

    Sole owner of the "no trainable model" error. The primary model is the one
    exposed as ``model=`` / put through surgery / checkpointed — so for
    dry-run/export/estimate ("build one model") it is the model to build.
    """
    for name, entry in models_cfg.items():
        if entry["trainable"]:
            return name, entry
    raise ConfigError(
        "recipe defines no trainable model (every `models:` entry is "
        "`trainable: false`)."
    )


def optim_spec_for(
    entry: Mapping[str, Any], optimizers_cfg: Mapping[str, Any]
) -> Any | None:
    """Resolve an entry's optimizer spec: explicit name → lookup; ``None`` →
    ``main`` (or the first declared optimizer). Returns ``None`` when nothing
    matches — the caller decides whether that is an error (training raises;
    estimate falls back to a generic byte estimate)."""
    opt_name = entry.get("optimizer")
    if opt_name is None:
        opt_name = "main" if "main" in optimizers_cfg else (
            next(iter(optimizers_cfg)) if optimizers_cfg else None
        )
    if opt_name is None:
        return None
    return optimizers_cfg.get(opt_name)


def build_primary_model(cfg: Any) -> tuple[Any, int]:
    """Build the PRIMARY trainable model and return ``(model, n_trainable)``.

    The single "resolve a model declaration → build the one model to
    verify/ship" entry point — used by ``dry-run --build``, ``export`` and
    ``produce-artifact``. The primary is the first trainable entry (the model
    exposed as ``model=`` and checkpointed). The trainable count lets a caller
    (export) warn when it is silently picking one of several trainable models.
    For the full multi-model set (Axis-A/B) use ``normalize_model_set`` directly.
    """
    models_cfg, _ = normalize_model_set(cfg)
    _name, entry = primary_trainable(models_cfg)
    model = _resolve(entry["spec"], category="model")
    n_trainable = sum(1 for e in models_cfg.values() if e["trainable"])
    return model, n_trainable


__all__ = [
    "normalize_model_set",
    "primary_trainable",
    "optim_spec_for",
    "build_primary_model",
]
