"""Module replacement / insertion helpers.

Two primitives:

* :func:`replace_module` — swap an existing submodule at a dotted path with
  a new one (``factory`` may be either a ready ``nn.Module`` or a callable
  ``old -> new``).
* :func:`add_named_module` — insert a new submodule at a dotted path that
  doesn't yet exist, creating empty ``nn.Module`` containers along the way.
  HiddenStatesMSELoss(project=True) uses this to attach learned projections
  under ``model._distill_projections.layer_<i>``; because the new module
  lives inside the model's ``named_parameters()`` tree, it automatically:

  * follows ``model.to(device)``;
  * shows up to ``OptimizerWrapper.build(model)`` (when called before the
    optimizer is constructed) and to ``ctx.notify_new_trainable_params``
    (when added later — see ``StandardUpdateRule``);
  * gets saved by the checkpoint manager.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

Factory = nn.Module | Callable[[nn.Module], nn.Module]


def _split_path(dotted: str) -> tuple[list[str], str]:
    parts = dotted.split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"Invalid dotted path: {dotted!r}")
    return parts[:-1], parts[-1]


def _walk(model: nn.Module, parts: list[str]) -> nn.Module:
    cur = model
    for p in parts:
        if not hasattr(cur, p):
            raise AttributeError(
                f"Module path component {p!r} not found on {type(cur).__name__}"
            )
        cur = getattr(cur, p)
    return cur


def replace_module(model: nn.Module, dotted_path: str, factory: Factory) -> nn.Module:
    """Replace the submodule at ``dotted_path`` and return the new module.

    ``factory`` may be:

    * an ``nn.Module`` — installed verbatim
    * a callable ``(old_module) -> new_module`` — receives the existing
      submodule (handy for e.g. wrapping ``nn.Linear`` with a LoRA module)
    """
    parent_parts, leaf = _split_path(dotted_path)
    parent = _walk(model, parent_parts)
    if not hasattr(parent, leaf):
        raise AttributeError(
            f"Cannot replace {dotted_path!r}: {leaf!r} does not exist on "
            f"{type(parent).__name__}"
        )
    old = getattr(parent, leaf)
    new = factory(old) if callable(factory) and not isinstance(factory, nn.Module) else factory
    if not isinstance(new, nn.Module):
        raise TypeError(
            f"replace_module factory must yield nn.Module, got {type(new).__name__}"
        )
    setattr(parent, leaf, new)
    return new


def add_named_module(model: nn.Module, dotted_path: str, module: nn.Module) -> nn.Module:
    """Insert ``module`` at ``dotted_path``, creating intermediate containers.

    Missing intermediate names become empty ``nn.Module`` instances; this is
    enough to register them in the parent's ``_modules`` dict so they appear
    in ``state_dict`` and ``named_parameters``.

    Idempotent only insofar as the existing module at ``dotted_path`` (if
    any) is overwritten — the caller is responsible for not re-inserting
    different ``nn.Module`` instances under the same path mid-training.
    """
    parent_parts, leaf = _split_path(dotted_path)
    cur = model
    for p in parent_parts:
        if not hasattr(cur, p):
            setattr(cur, p, nn.Module())
        nxt = getattr(cur, p)
        if not isinstance(nxt, nn.Module):
            raise TypeError(
                f"Path component {p!r} on {type(cur).__name__} is "
                f"{type(nxt).__name__}, not nn.Module"
            )
        cur = nxt
    setattr(cur, leaf, module)
    return module


def get_submodule(model: nn.Module, dotted_path: str) -> nn.Module:
    """Return submodule at ``dotted_path``; mirrors ``nn.Module.get_submodule``
    but accepts our empty-string special case and raises ``AttributeError``
    consistently."""
    if not dotted_path:
        return model
    parts = dotted_path.split(".")
    return _walk(model, parts[:-1]).__getattr__(parts[-1])


__all__ = ["replace_module", "add_named_module", "get_submodule"]
