"""Sweep backend plugins for lighttrain.lab.sweep.

Each backend implements the ``suggest / report / should_prune`` protocol
and is registered as ``@register("sweep_backend", "<name>")``.

Currently available:
  * ``optuna`` — Optuna TPE sampler (requires ``pip install -e '.[sweep]'``)
"""
