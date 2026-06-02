"""Eager registry population — import every ``@register``-bearing module so the
component registry is fully populated before recipe specs are resolved.

Replaces the hand-maintained import list that silently drifted (``info_nce`` /
``moe_balance`` went unregistered; ``export`` forgot to call the importer). A
curated list of packages is walked **recursively** so nested registrables
(``data.core.datasets``, ``models.adapters.tiny_lm``, ...) are found
automatically; a brand-new *top-level* package is the one residual maintenance
point — add a line to ``_FIRST_PARTY_PACKAGES``.

Invoked at the ``load_config`` chokepoint (and a couple of non-load_config entry
points), so no command can forget it.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

# Curated, explicit package list. ``walk_packages`` recurses INTO each, so
# nested registrables are discovered. We deliberately do NOT scan the
# ``lighttrain`` top package — that would pull in test/private/example modules
# and their side effects.
# Every top-level package that contains an ``@register`` decorator. ``grep -rl
# '@register(' lighttrain/`` is the source of truth for this list — a brand-new
# top-level package is the one residual maintenance point.
_FIRST_PARTY_PACKAGES: tuple[str, ...] = (
    "lighttrain.architectures",     # arch-profile factories (architecture: transformer)
    "lighttrain.models",            # adapters (tiny_lm/hf_causal), peft (lora/ia3/adalora)
    "lighttrain.data",              # core datasets/samplers/tokenizers/collators + mm
    "lighttrain.prepgraph",         # prep_node graph nodes
    "lighttrain.losses",            # core/distill/preference/aux (info_nce/moe_balance)
    "lighttrain.rl",                # reward_adapters, value_heads
    "lighttrain.optim",             # schedulers, wrappers
    "lighttrain.engine",            # standard engine
    "lighttrain.update_rules",      # standard/sam/mezo/rl
    "lighttrain.distributed",       # grad_sync / model_parallel strategies
    "lighttrain.logging.backends",  # console/jsonl/tb (tb optional)
    "lighttrain.callbacks",         # builtins + invariants + frozen_step
    "lighttrain.trainers",          # pretrain/preference/ppo/grpo/...
    "lighttrain.eval",              # regression_gate invariant (judges are plugins)
    "lighttrain.artifacts",         # producer/store/joined_dataset/dynamic_producer
    "lighttrain.invariants",        # invariant registry
    "lighttrain.diagnostics",       # nan_hunter/dead_neuron/... (some optional)
    "lighttrain.realtime_control",  # file_signals
)
# Bundled opt-in plugins (ship under the lighttrain namespace). Walked here so
# their @register calls land; individual submodules whose third-party dep is
# absent are skipped by the per-module contract below.
_OPTIONAL_PACKAGES: tuple[str, ...] = ("lighttrain.plugins",)

_DONE = False


def _missing_dep_is_internal(exc: ImportError) -> bool:
    """A failed import is a genuine *internal* bug iff the missing module is a
    ``lighttrain.*`` module (a typo'd sibling / broken internal import). A
    missing third-party dep (tensorboard, peft, ...) or a missing external
    ``plugins`` just means an optional backend isn't installed."""
    return (exc.name or "").split(".")[0] == "lighttrain"


def _safe_import(name: str) -> None:
    """Import a module under the failure contract:

    * ``ImportError`` whose missing dep is a ``lighttrain.*`` module → re-raise
      (internal breakage — loud).
    * ``ImportError`` for a missing third-party / external dep → skip (optional).
    * any non-``ImportError`` (e.g. ``RuntimeError`` raised at import time) →
      propagate (loud — never blanket-swallow, unlike the old ``try/except``).
    """
    try:
        importlib.import_module(name)
    except ImportError as exc:
        if _missing_dep_is_internal(exc):
            raise
        # optional third-party / external dep absent — skip quietly
    # non-ImportError propagates by design


def _walk_and_import(pkg: ModuleType) -> None:
    """Recursively import every submodule of ``pkg``, applying the contract to
    both leaf-module imports and ``walk_packages``' own package descent."""
    if not hasattr(pkg, "__path__"):
        return  # plain module — already imported

    def _onerror(name: str) -> None:
        # Called when walk_packages fails to import a (sub)package while
        # descending. Re-raise on internal breakage / non-ImportError; return
        # (continue the walk) for an absent optional dep.
        exc = sys.exc_info()[1]
        if isinstance(exc, ImportError) and not _missing_dep_is_internal(exc):
            return
        raise

    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".", onerror=_onerror):
        _safe_import(info.name)


def import_all_components() -> None:
    """Import every first-party ``@register``-bearing module (+ optional
    plugins) so the registry is fully populated. Idempotent; cheap after the
    first call (guarded by ``_DONE`` — note this only amortises within a single
    process, so pure-parse CLI paths skip it via ``register_components=False``)."""
    global _DONE
    if _DONE:
        return
    for name in _FIRST_PARTY_PACKAGES:
        _walk_and_import(importlib.import_module(name))
    for name in _OPTIONAL_PACKAGES:
        try:
            pkg = importlib.import_module(name)
        except ImportError:
            continue  # external package not installed
        _walk_and_import(pkg)
    _DONE = True


__all__ = ["import_all_components"]
