"""Adversarial tests for ``lighttrain.protocols``.

@runtime_checkable Protocols have well-known structural-typing holes:
the runtime check only verifies *method existence*, never *signatures*,
*return types*, or *attribute types*. These tests pin the current contract
so a future tightening (e.g. adding signature checks) is a deliberate,
coordinated change rather than silent surprise.

Coverage:

* CallbackProtocol is empty by design — any object passes (pinned).
* ModelProtocol / LossFnProtocol / SchedulerProtocol / etc. require their
  named methods.
* Protocols do NOT validate signatures or return types (pinned).
* Sweep: every Protocol exported from lighttrain.protocols is
  ``@runtime_checkable``.
* Data carriers (ModelOutput / LossContext / StepOutput) have the expected
  dataclass fields.
* CALLBACK_EVENTS is a non-empty tuple and includes core lifecycle events.
"""

from __future__ import annotations

import pytest
import torch

import lighttrain.protocols as protomod
from lighttrain.protocols import (
    CALLBACK_EVENTS,
    CallbackProtocol,
    CheckpointManagerProtocol,
    LoggerProtocol,
    LossContext,
    LossFnProtocol,
    ModelOutput,
    ModelProtocol,
    OptimizerWrapperProtocol,
    SchedulerProtocol,
    StepOutput,
)

# ---------------------------------------------------------------------------
# CallbackProtocol — empty body is a deliberate design (pinned)
# ---------------------------------------------------------------------------

def test_pin_callback_protocol_is_empty_any_object_passes_isinstance():
    """Pin: ``CallbackProtocol`` has no required methods. Per its source
    docstring: "methods are optional; the EventBus checks via getattr."
    Consequently, ``isinstance(<anything>, CallbackProtocol)`` is True.

    Setup: instances of ``object()``, an int, a list, and a custom class.
    Expected: all four return True for isinstance(CallbackProtocol).

    If you intentionally add a required method to CallbackProtocol (e.g.
    a name attribute or a base on_init method), update this test AND
    document the breaking change in the changelog.
    """
    class C:
        pass

    for obj in [object(), 42, [], C()]:
        assert isinstance(obj, CallbackProtocol), (
            f"CallbackProtocol must accept any object as a structural match "
            f"(empty body); rejected {type(obj).__name__}"
        )


# ---------------------------------------------------------------------------
# ModelProtocol — forward() required, signatures NOT validated
# ---------------------------------------------------------------------------

def test_model_protocol_rejects_object_without_forward():
    """Object lacking ``forward`` is rejected by isinstance check.

    Setup: a class without a ``forward`` method.
    Expected: isinstance returns False.
    """
    class NoForward:
        def predict(self, x): return x

    assert not isinstance(NoForward(), ModelProtocol)


def test_model_protocol_accepts_object_with_forward():
    """Object with a ``forward`` method passes isinstance.

    Setup: a class with ``forward(self, **batch) -> ModelOutput``.
    Expected: isinstance returns True.
    """
    class HasForward:
        def forward(self, **batch):
            return ModelOutput()

    assert isinstance(HasForward(), ModelProtocol)


def test_pin_model_protocol_does_not_validate_forward_signature():
    """Pin: structural typing only checks method *existence*, not signature.
    A ``forward(self)`` (no batch arg) still passes isinstance against
    ``ModelProtocol.forward(self, **batch)``.

    Setup: a class with ``forward(self)`` taking no kwargs.
    Expected: isinstance returns True even though calling it with **batch
    would TypeError.

    If you intentionally add signature validation (via a runtime decorator
    or custom __subclasshook__), update this test.
    """
    class BadSignature:
        def forward(self):  # missing **batch
            return ModelOutput()

    assert isinstance(BadSignature(), ModelProtocol)


def test_pin_model_protocol_does_not_validate_forward_return_type():
    """Pin: structural typing does NOT validate that ``forward`` returns a
    ModelOutput. A method returning ``int`` still passes isinstance.

    Setup: ``forward`` returns 42.
    Expected: isinstance True.

    If you intentionally add return-type validation, update this test.
    """
    class WrongReturn:
        def forward(self, **batch):
            return 42  # not a ModelOutput

    assert isinstance(WrongReturn(), ModelProtocol)


# ---------------------------------------------------------------------------
# LossFnProtocol — __call__ required
# ---------------------------------------------------------------------------

def test_loss_fn_protocol_accepts_callable_object():
    """A class instance with ``__call__`` defined passes isinstance.

    Setup: instance of a class with ``__call__(self, mo, batch, ctx)``.
    Expected: isinstance True.
    """
    class CallableLoss:
        def __call__(self, model_output, batch, ctx):
            return {"loss": torch.tensor(0.0)}

    assert isinstance(CallableLoss(), LossFnProtocol)


def test_loss_fn_protocol_accepts_plain_function():
    """A bare Python function (``def fn(...)``) passes isinstance against
    LossFnProtocol because functions are callable (have ``__call__``).

    Setup: a plain function.
    Expected: isinstance True.
    """
    def loss_fn(model_output, batch, ctx):
        return {"loss": torch.tensor(0.0)}

    assert isinstance(loss_fn, LossFnProtocol)


# ---------------------------------------------------------------------------
# SchedulerProtocol — attr + 3 methods
# ---------------------------------------------------------------------------

def test_scheduler_protocol_requires_step_per_batch_attr_and_step_method():
    """SchedulerProtocol needs step_per_batch, step, state_dict, load_state_dict.

    Setup: two classes — one with all members, one missing ``step``.
    Expected: the complete class passes; the missing-step class fails.
    """
    class Complete:
        step_per_batch = True

        def step(self, *args, **kwargs): pass
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass

    class MissingStep:
        step_per_batch = True

        def state_dict(self): return {}
        def load_state_dict(self, sd): pass

    assert isinstance(Complete(), SchedulerProtocol)
    assert not isinstance(MissingStep(), SchedulerProtocol)


# ---------------------------------------------------------------------------
# OptimizerWrapperProtocol — attribute + method
# ---------------------------------------------------------------------------

def test_optimizer_wrapper_protocol_requires_full_contract():
    """OptimizerWrapperProtocol requires the ``optimizer`` attribute plus the
    full method contract the framework actually calls on the wrapper:
    ``build / step / zero_grad / state_dict / load_state_dict`` (issue #5 —
    the checkpoint manager calls state_dict/load_state_dict on the wrapper).

    The optional ``optim_state_bytes`` hook is NOT part of the runtime-checkable
    surface, so a wrapper without it still satisfies the protocol.
    """
    p = torch.nn.Parameter(torch.tensor([1.0]))

    class Complete:
        def __init__(self):
            self.optimizer = torch.optim.SGD([p], lr=0.01)
        def build(self, model): return self.optimizer
        def step(self, *a, **k): ...
        def zero_grad(self, set_to_none=True): ...
        def state_dict(self): return {}
        def load_state_dict(self, sd): ...

    class NoBuild:
        def __init__(self):
            self.optimizer = torch.optim.SGD([p], lr=0.01)
        def step(self, *a, **k): ...
        def zero_grad(self, set_to_none=True): ...
        def state_dict(self): return {}
        def load_state_dict(self, sd): ...

    class NoCheckpointMethods:
        """Has optimizer + build + step + zero_grad but omits state_dict /
        load_state_dict — the exact README footgun (issue #5)."""
        def __init__(self):
            self.optimizer = torch.optim.SGD([p], lr=0.01)
        def build(self, model): return self.optimizer
        def step(self, *a, **k): ...
        def zero_grad(self, set_to_none=True): ...

    class NoOptimizer:
        def build(self, model): return None
        def step(self, *a, **k): ...
        def zero_grad(self, set_to_none=True): ...
        def state_dict(self): return {}
        def load_state_dict(self, sd): ...

    assert isinstance(Complete(), OptimizerWrapperProtocol)
    assert not isinstance(NoBuild(), OptimizerWrapperProtocol)
    assert not isinstance(NoCheckpointMethods(), OptimizerWrapperProtocol)
    assert not isinstance(NoOptimizer(), OptimizerWrapperProtocol)
    # The optional hook is absent from Complete, yet it still conforms.
    assert not hasattr(Complete(), "optim_state_bytes")


# ---------------------------------------------------------------------------
# LoggerProtocol — five methods, parametrize over "missing" coverage
# ---------------------------------------------------------------------------

_LOGGER_METHODS = (
    "log_scalars",
    "log_histograms",
    "log_text",
    "log_artifact",
    "flush",
)


def _make_complete_logger():
    class CompleteLogger:
        def log_scalars(self, scalars, step): pass
        def log_histograms(self, hists, step): pass
        def log_text(self, text, step): pass
        def log_artifact(self, path, name=None): pass
        def flush(self): pass
    return CompleteLogger()


def test_logger_protocol_complete_implementation_passes():
    """All five LoggerProtocol methods present → isinstance True."""
    assert isinstance(_make_complete_logger(), LoggerProtocol)


@pytest.mark.parametrize("missing", _LOGGER_METHODS)
def test_logger_protocol_missing_any_required_method_fails(missing):
    """If any of the 5 required methods is missing, isinstance is False.

    Goal: pin the required-method set exactly. Adding a new required
    method (or removing one) flips this test.

    Parametrize: one test per required method, removing exactly that one.
    """
    attrs = {m: (lambda self, *a, **k: None) for m in _LOGGER_METHODS if m != missing}
    Cls = type("PartialLogger", (), attrs)
    assert not isinstance(Cls(), LoggerProtocol), (
        f"isinstance accepted a Logger missing {missing!r}"
    )


# ---------------------------------------------------------------------------
# CheckpointManagerProtocol
# ---------------------------------------------------------------------------

def test_checkpoint_manager_protocol_requires_save_load_latest_best():
    """Four required methods: save, load, latest, best.

    Setup: complete and partial (no ``best``) impls.
    Expected: complete passes; missing-best fails.
    """
    class Complete:
        def save(self, step, state, *, kind="step", extras=None): return None
        def load(self, path): return {}
        def latest(self): return None
        def best(self): return None

    class NoBest:
        def save(self, step, state, *, kind="step", extras=None): return None
        def load(self, path): return {}
        def latest(self): return None

    assert isinstance(Complete(), CheckpointManagerProtocol)
    assert not isinstance(NoBest(), CheckpointManagerProtocol)


# ---------------------------------------------------------------------------
# Sweep: every Protocol in lighttrain.protocols is @runtime_checkable
# ---------------------------------------------------------------------------

def _is_protocol_class(obj) -> bool:
    """Heuristic: an attr is a Protocol subclass if it's a class whose MRO
    contains Protocol from typing."""
    if not isinstance(obj, type):
        return False
    try:
        from typing import Protocol as _P
        return _P in obj.__mro__
    except Exception:
        return False


def test_invariant_every_protocol_class_in_protocols_module_is_runtime_checkable():
    """Invariant: every Protocol subclass exported by lighttrain.protocols is
    decorated with ``@runtime_checkable`` so callers can use isinstance().

    Sweep: enumerate the module's public attributes, filter to Protocol
    subclasses, assert each has the ``_is_runtime_protocol`` flag set.

    Goal: catches a refactor that accidentally removes ``@runtime_checkable``
    from one Protocol — would silently break ``isinstance`` checks at all
    call sites.
    """
    failures = []
    for name in protomod.__all__:
        attr = getattr(protomod, name, None)
        if _is_protocol_class(attr):
            # typing.Protocol sets _is_runtime_protocol when @runtime_checkable
            # is applied; absent the decorator the flag is False or missing.
            if not getattr(attr, "_is_runtime_protocol", False):
                failures.append(name)
    assert not failures, (
        f"Protocols missing @runtime_checkable: {failures}. "
        "Without the decorator, isinstance() raises TypeError."
    )


# ---------------------------------------------------------------------------
# Data carriers — ModelOutput / LossContext / StepOutput
# ---------------------------------------------------------------------------

def test_model_output_default_construction_yields_empty_dicts_and_none_loss():
    """``ModelOutput()`` with no args has empty outputs/extras and None loss.

    Closed form: outputs == {}, extras == {}, loss is None, state is None.
    """
    mo = ModelOutput()
    assert mo.outputs == {}
    assert mo.extras == {}
    assert mo.loss is None
    assert mo.state is None
    assert mo.hidden_states is None
    assert mo.attentions is None


def test_model_output_with_logits_round_trip():
    """ModelOutput constructed with a logits tensor round-trips it.

    Setup: ModelOutput(outputs={"logits": tensor([1.0, 2.0])}).
    Closed form: outputs["logits"] matches input tensor via assert_close.
    """
    t = torch.tensor([1.0, 2.0])
    mo = ModelOutput(outputs={"logits": t})
    torch.testing.assert_close(mo.outputs["logits"], t, atol=1e-5, rtol=1e-4)


def test_loss_context_default_field_values():
    """LossContext defaults: step=0, epoch=0, metrics={}, loss_family=None,
    extras={}.
    """
    ctx = LossContext()
    assert ctx.step == 0
    assert ctx.epoch == 0
    assert ctx.metrics == {}
    assert ctx.loss_family is None
    assert ctx.extras == {}


def test_loss_context_each_dataclass_field_present():
    """Pin: LossContext has exactly these fields:
    step, epoch, metrics, loss_family, extras.

    Goal: catches a renamed / removed / added field — would break code
    that relies on attribute access.
    """
    import dataclasses
    expected = {"step", "epoch", "metrics", "loss_family", "extras"}
    actual = {f.name for f in dataclasses.fields(LossContext)}
    assert actual == expected, f"LossContext fields changed: {actual ^ expected}"


def test_step_output_default_field_values():
    """StepOutput defaults: loss=None, metrics={}, logs={}, extras={}."""
    so = StepOutput()
    assert so.loss is None
    assert so.metrics == {}
    assert so.logs == {}
    assert so.extras == {}


def test_step_output_each_dataclass_field_present():
    """Pin: StepOutput has exactly: loss, metrics, logs, extras."""
    import dataclasses
    expected = {"loss", "metrics", "logs", "extras"}
    actual = {f.name for f in dataclasses.fields(StepOutput)}
    assert actual == expected


# ---------------------------------------------------------------------------
# CALLBACK_EVENTS sanity pins
# ---------------------------------------------------------------------------

def test_callback_events_is_nonempty_tuple_of_strings():
    """CALLBACK_EVENTS must be a tuple of unique strings (no duplicates).

    Goal: a stray duplicate would silently double-fire events.
    """
    assert isinstance(CALLBACK_EVENTS, tuple)
    assert len(CALLBACK_EVENTS) > 0
    assert all(isinstance(e, str) for e in CALLBACK_EVENTS)
    assert len(set(CALLBACK_EVENTS)) == len(CALLBACK_EVENTS), (
        f"CALLBACK_EVENTS has duplicates: {CALLBACK_EVENTS}"
    )


def test_callback_events_includes_core_lifecycle_set():
    """CALLBACK_EVENTS must include a known-stable subset; if any are
    removed the framework's lifecycle contract breaks.
    """
    required = {
        "on_train_start",
        "on_train_end",
        "on_step_begin",
        "on_step_end",
        "on_loss_computed",
        "on_backward_pre",
        "on_backward_post",
        "on_optimizer_step_pre",
        "on_optimizer_step_post",
        "on_exception",
    }
    actual = set(CALLBACK_EVENTS)
    missing = required - actual
    assert not missing, f"CALLBACK_EVENTS missing required lifecycle events: {missing}"
