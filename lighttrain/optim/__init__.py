"""Optimizer / scheduler framework — the core abstraction (extension point).

Holds the abstract base classes (``OptimizerWrapperBase`` / ``SchedulerBase``)
and the ``ParamGroupSpec`` param-group DSL. Concrete optimizers (``adamw`` /
``lion``) and schedulers (``constant`` / ``linear`` / ``warmup_cosine`` / ``wsd``)
are registered impls in ``lighttrain.builtin_plugins.optim`` (DESIGN §3.3:
protocols/bases in core, concrete impls in builtin_plugins).
"""

from __future__ import annotations

from .base import OptimizerWrapperBase, ParamGroupSpec, SchedulerBase

__all__ = ["OptimizerWrapperBase", "ParamGroupSpec", "SchedulerBase"]
