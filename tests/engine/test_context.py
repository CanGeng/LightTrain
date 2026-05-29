"""Adversarial tests for StepContext defaults — guards against the classic
mutable-default-argument bug.

StepContext is a frozen-like dataclass with ``field(default_factory=dict)``
for the ``metrics`` / ``extras`` / ``diagnostics`` slots. The right pattern
is required because using plain ``dict()`` as default would make ALL
StepContext instances share the same dict — a silent corruption bug.
"""

from __future__ import annotations

from lighttrain.engine._context import StepContext


def test_stepcontext_defaults_safe_for_single_gpu():
    """Goal: default StepContext is safe to use on single-GPU (no NPEs).

    Pins each field's default explicitly so a refactor that flips a default
    (e.g. ``grad_sync=Any`` instead of ``None``, or ``is_accumulating=True``)
    fails loudly. These defaults are part of the contract: update rules
    use ``getattr(ctx, "grad_sync", None) is None`` to pick the bare
    backward path.
    """
    ctx = StepContext()
    assert ctx.step == 0
    assert ctx.epoch == 0
    assert ctx.global_step == 0
    assert ctx.is_accumulating is False
    assert ctx.metrics == {}
    assert ctx.extras == {}
    assert ctx.diagnostics == {}
    assert ctx.model is None
    assert ctx.optimizer is None
    assert ctx.scheduler is None
    assert ctx.loss_fn is None
    assert ctx.accelerator is None
    assert ctx.bus is None
    assert ctx.logger is None
    assert ctx.grad_sync is None
    assert ctx.parallel_ctx is None
    assert ctx.lineage_store is None
    assert ctx.run_id is None
    assert ctx.run_dir is None
    assert ctx.frozen_step_writer is None
    assert ctx.mode == "lab"


def test_stepcontext_metrics_and_extras_are_per_instance():
    """Goal: each StepContext gets its own metrics/extras/diagnostics dict.

    Construction:
      - build ctx_a
      - mutate ctx_a.metrics, ctx_a.extras, ctx_a.diagnostics
      - build ctx_b
      - assert ctx_b's dicts are still empty

    Catches the classic ``def __init__(..., metrics: dict = {})`` mutable
    default bug — pre-dataclass code is full of these. Using
    ``field(default_factory=dict)`` (the current contract) prevents it.
    A refactor swapping to a plain default would silently leak state
    between trainers.
    """
    ctx_a = StepContext()
    ctx_a.metrics["loss"] = 1.0
    ctx_a.extras["key"] = "value"
    ctx_a.diagnostics["sample"] = [1, 2, 3]

    ctx_b = StepContext()
    assert ctx_b.metrics == {}
    assert ctx_b.extras == {}
    assert ctx_b.diagnostics == {}

    # And the dicts must be DISTINCT objects, not just equal-but-shared.
    assert ctx_a.metrics is not ctx_b.metrics
    assert ctx_a.extras is not ctx_b.extras
    assert ctx_a.diagnostics is not ctx_b.diagnostics


def test_stepcontext_kwargs_constructor_accepts_all_documented_fields():
    """Goal: pin the dataclass signature — common fields can all be set via kwargs.

    Catches a refactor that renames a field (e.g. ``loss_fn`` → ``loss_function``)
    without updating callers — a silent contract break.
    """
    sentinel_model = object()
    sentinel_optim = object()
    sentinel_bus = object()
    sentinel_loss = object()
    sentinel_accel = object()

    ctx = StepContext(
        step=5,
        epoch=2,
        global_step=10,
        model=sentinel_model,
        optimizer=sentinel_optim,
        bus=sentinel_bus,
        loss_fn=sentinel_loss,
        accelerator=sentinel_accel,
    )

    assert ctx.step == 5
    assert ctx.epoch == 2
    assert ctx.global_step == 10
    assert ctx.model is sentinel_model
    assert ctx.optimizer is sentinel_optim
    assert ctx.bus is sentinel_bus
    assert ctx.loss_fn is sentinel_loss
    assert ctx.accelerator is sentinel_accel
