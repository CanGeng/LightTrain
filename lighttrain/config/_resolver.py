"""Component resolver — turn a ComponentSpec into a constructed object.

Two routes:

* short name: ``{name: adamw, lr: 1e-4}`` → ``Registry.get('optimizer', 'adamw')(lr=1e-4)``
* dotted target: ``{_target_: lighttrain.optim.AdamW, lr: 1e-4}`` → import + call

The two routes are mutually exclusive (enforced by ComponentSpec validator).
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any, Mapping

from ..registry import get as _registry_get
from ._exceptions import ConfigResolveError
from ._schema import ComponentSpec


def _filter_kwargs(factory: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not in ``factory``'s signature; warn on drops.

    Pass-through when the factory accepts ``**kwargs`` (``VAR_KEYWORD``) or
    when the signature cannot be introspected (builtins / C extensions).

    Drops are warned, not raised, so a recipe authored against one model can
    still build another model when the user CLI-overrides ``model.name`` —
    the OmegaConf-merged ``model:`` block carries all sibling keys and we
    cannot ask the user to ``~model.n_layers`` before every switch.
    Bare ``*args``-only signatures still pass through this filter; positional
    args can't be expressed via kwargs anyway.
    """
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return kwargs
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    accepted = {
        name for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    accepted.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        factory_name = getattr(factory, "__name__", repr(factory))
        sample = dropped[:5]
        suffix = "..." if len(dropped) > 5 else ""
        warnings.warn(
            f"Dropped {len(dropped)} unknown kwargs for {factory_name!r}: "
            f"{sample}{suffix}",
            UserWarning,
            stacklevel=2,
        )
    return filtered


def _coerce(spec: ComponentSpec | Mapping[str, Any]) -> ComponentSpec:
    if isinstance(spec, ComponentSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise ConfigResolveError(
            f"Spec must be a mapping or ComponentSpec, got {type(spec).__name__}"
        )
    data = dict(spec)
    name = data.pop("name", None)
    target = data.pop("_target_", None)
    explicit = data.pop("params", None)
    params = dict(explicit) if explicit else {}
    # Remaining keys become params (sugar: flat form).
    for k, v in data.items():
        params.setdefault(k, v)
    return ComponentSpec(name=name, _target_=target, params=params)


def _import_target(dotted: str) -> Any:
    # ── colon escape-hatch: "pkg.module:ClassName.method" ──────────────────
    if ":" in dotted:
        mod_str, _, attr_str = dotted.partition(":")
        try:
            module = importlib.import_module(mod_str)
        except ImportError as e:
            raise ConfigResolveError(
                f"Cannot import {mod_str!r} for _target_={dotted!r}: {e}"
            ) from e
        obj: Any = module
        try:
            for part in attr_str.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError as e:
            raise ConfigResolveError(
                f"Cannot resolve _target_={dotted!r}: {e}"
            ) from e

    # ── dotted-only: right-peel until an importable prefix is found ─────────
    all_parts = dotted.split(".")
    if len(all_parts) < 2:
        raise ConfigResolveError(f"Invalid _target_ path: {dotted!r}")

    last_err: Exception | None = None
    for split in range(len(all_parts) - 1, 0, -1):
        mod_str = ".".join(all_parts[:split])
        attr_parts = all_parts[split:]
        try:
            module = importlib.import_module(mod_str)
        except ModuleNotFoundError as e:
            # Only continue peeling if the missing module IS exactly the prefix
            # we just tried (meaning the prefix itself doesn't exist as a module).
            # If e.name is anything else — including a sub-module of mod_str
            # (e.g. "pkg.mod.utils") — the prefix EXISTS but has a broken
            # internal import; raise immediately to surface the real error.
            missing = e.name or ""
            if missing == mod_str:
                last_err = e
                continue
            raise ConfigResolveError(
                f"Module {mod_str!r} has a missing dependency "
                f"({missing!r}) while resolving _target_={dotted!r}: {e}"
            ) from e
        except ImportError as e:
            # Module found but failed to load internally — don't peel further.
            raise ConfigResolveError(
                f"Module {mod_str!r} failed to import while resolving "
                f"_target_={dotted!r}: {e}"
            ) from e
        # Module imported — walk attribute chain
        obj = module
        try:
            for part in attr_parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            # Module exists but attribute chain is wrong at this split point;
            # try a shorter prefix (e.g. "transformers" instead of
            # "transformers.AutoTokenizer" for the 3-level case).
            last_err = AttributeError(
                f"{mod_str!r} has no attribute chain {'.'.join(attr_parts)!r}"
            )
            continue

    raise ConfigResolveError(
        f"Cannot resolve _target_={dotted!r}"
        + (f": {last_err}" if last_err else "")
    )


def resolve(
    spec: ComponentSpec | Mapping[str, Any],
    category: str | None = None,
    *,
    instantiate: bool = True,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a ComponentSpec to a class/factory or constructed instance.

    ``category`` is required when ``spec.name`` is used. ``extra_kwargs`` is
    merged into the params (callee precedence) for dependency injection.
    """
    cs = _coerce(spec)
    if cs.name is not None:
        if category is None:
            raise ConfigResolveError(
                "Resolving a short-name spec requires `category`."
            )
        factory = _registry_get(category, cs.name)
    else:
        factory = _import_target(cs.target)  # type: ignore[arg-type]

    if not instantiate:
        return factory

    kwargs = dict(cs.params)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs = _filter_kwargs(factory, kwargs)
    try:
        return factory(**kwargs)
    except TypeError as e:
        factory_name = getattr(factory, "__name__", repr(factory))
        raise ConfigResolveError(
            f"Failed to construct {factory_name}:\n"
            f"  Cause: {type(e).__name__}: {e}\n"
            f"  Params: {kwargs!r}"
        ) from e


__all__ = ["resolve"]
