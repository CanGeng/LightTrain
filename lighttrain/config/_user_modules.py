"""Import user-supplied component modules so their ``@register`` decorators run.

This is the single mechanism behind the recipe ``user_modules:`` field. It lives
in ``config/`` (depending only on ``importlib`` + ``Path``) so that the config
loader can call it without a ``config → cli`` reverse dependency.

It is invoked from exactly one chokepoint — :func:`lighttrain.config.load_config`
— which every recipe-eating command flows through. The only other callers are
two *library* entry points that legitimately bypass ``load_config``
(``setup_run_from_config`` with a pre-parsed ``RootConfig``; ``lab.estimate``
with a raw dict). Each call is idempotent, so a redundant call is free.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

# Process-wide dedup set. Keyed by resolved file path (for ``.py`` paths) or the
# dotted module name. This is a cheap optimization to avoid re-exec'ing a module;
# it is no longer load-bearing for correctness now that ``register()`` is
# idempotent by content identity — a duplicate import simply no-ops.
_IMPORTED_USER_MODULES: set[str] = set()


def import_user_modules(modules: list[str]) -> None:
    """Import every entry in ``user_modules`` so ``@register`` decorators execute.

    Accepts dotted module names (``mypkg.mymod``) and file paths
    (``./plugins/my_optim.py``, ``/abs/path/module.py``). Must run after config
    loading but before any component resolution.

    Idempotent within a process: the second and later calls for a given module
    are no-ops.

    Raises ImportError with context on the first failure.
    """
    for mod in modules:
        _is_path = mod.endswith(".py") or "/" in mod or "\\" in mod
        if _is_path:
            try:
                key = str(Path(mod).expanduser().resolve())
            except (OSError, RuntimeError):
                key = mod  # fall back to raw string if resolve() blows up
        else:
            key = mod
        if key in _IMPORTED_USER_MODULES:
            continue
        try:
            if _is_path:
                p = Path(mod).expanduser().resolve()
                spec = importlib.util.spec_from_file_location(p.stem, p)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load spec from {p}")
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
            else:
                importlib.import_module(mod)
        except (ImportError, FileNotFoundError) as exc:
            raise ImportError(
                f"user_modules: failed to import {mod!r}. "
                "Check that the path exists or the dotted name is on sys.path."
            ) from exc
        _IMPORTED_USER_MODULES.add(key)


__all__ = ["import_user_modules"]
