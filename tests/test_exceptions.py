"""Adversarial tests for the lighttrain exception hierarchy.

Covers ``lighttrain.exceptions`` (framework-level), ``lighttrain.config._exceptions``
(config errors), and ``lighttrain.registry._exceptions`` (registry errors).

The exception types are small (5 leaf classes total) but their hierarchy is
load-bearing: callers do ``except RegistryError`` or ``except ConfigError`` to
catch a whole family. A future refactor that re-parents one of these to
``Exception`` directly would silently break those except clauses. These tests
pin the hierarchy + the public re-exports + the ``BatchValidationError``
message format.
"""

from __future__ import annotations

import pytest

from lighttrain.config import ConfigError, ConfigResolveError, ConfigSchemaError
from lighttrain.exceptions import BatchValidationError, LightTrainError
from lighttrain.registry import (
    NotRegisteredError,
    RegistryConflictError,
    RegistryError,
    UnknownCategoryError,
)

# ---------------------------------------------------------------------------
# BatchValidationError — message format + truncation
# ---------------------------------------------------------------------------

def test_batch_validation_error_message_lists_all_missing_keys():
    """str(e) contains every missing key, the trainer name, and the present
    keys (or a truncation marker).

    Setup: missing=['labels', 'attention_mask'], present=['input_ids'].
    Expected: message contains 'pretrain', 'labels', 'attention_mask',
    'input_ids', and the diagnostic hint about the collator.
    """
    e = BatchValidationError(
        trainer_name="pretrain",
        missing_keys=["labels", "attention_mask"],
        present_keys=["input_ids"],
    )
    msg = str(e)
    assert "pretrain" in msg
    assert "labels" in msg
    assert "attention_mask" in msg
    assert "input_ids" in msg
    assert "collator" in msg.lower()


def test_invariant_batch_validation_error_truncates_present_keys_at_ten():
    """Invariant: when ``present_keys`` has more than 10 entries the message
    shows only the first 10 (after sorting) plus a ``... (N more)`` marker.

    Setup: present = 15 distinct keys (k00..k14).
    Expected: message contains the first 10 in sorted order AND a marker
    indicating 5 are hidden.
    """
    present = [f"k{i:02d}" for i in range(15)]  # 15 keys
    e = BatchValidationError(
        trainer_name="ppo", missing_keys=["x"], present_keys=present
    )
    msg = str(e)
    # First 10 sorted keys (k00..k09) must appear; k10..k14 should not.
    for shown in [f"k{i:02d}" for i in range(10)]:
        assert shown in msg
    # k10+ are truncated by the marker; assert at least one of them is absent
    # to verify truncation actually happened (not just longer message).
    assert "k14" not in msg
    assert "(5 more)" in msg


def test_batch_validation_error_with_no_missing_or_present_does_not_crash():
    """Edge case: empty missing + empty present keys.

    Setup: ``BatchValidationError('x', [], [])``.
    Expected: instantiates without raising; str() returns a non-empty string.
    """
    e = BatchValidationError(trainer_name="x", missing_keys=[], present_keys=[])
    msg = str(e)
    assert msg  # non-empty
    assert "x" in msg


def test_batch_validation_error_non_string_keys_coerced_to_string():
    """``present_keys`` is expected to be ``list[str]`` but the constructor
    is defensive and coerces via ``str(k)``.

    Setup: present_keys contains an int (123) and a tuple ('a', 'b').
    Expected: instantiates without TypeError; str repr of the non-string
    keys appears in the message.
    """
    e = BatchValidationError(
        trainer_name="t",
        missing_keys=["m"],
        present_keys=[123, ("a", "b")],
    )
    msg = str(e)
    assert "123" in msg
    # tuple repr survives sorted+str coercion
    assert "(" in msg and "a" in msg


# ---------------------------------------------------------------------------
# Hierarchy pins — these protect except-clauses across the codebase
# ---------------------------------------------------------------------------

def test_invariant_batch_validation_error_subclass_of_lighttrain_error():
    """Invariant: BatchValidationError is a LightTrainError (catchable via
    the framework-base class).

    Setup: instantiate; check isinstance.
    Expected: True.
    """
    e = BatchValidationError("t", [], [])
    assert isinstance(e, LightTrainError)


def test_invariant_lighttrain_error_is_runtime_error_subclass():
    """Invariant: LightTrainError inherits from RuntimeError (NOT plain
    Exception), so legacy ``except RuntimeError`` clauses still catch our errors.

    Setup: raise LightTrainError, catch RuntimeError.
    Expected: caught.
    """
    with pytest.raises(RuntimeError):
        raise LightTrainError("boom")


def test_invariant_config_schema_error_subclass_of_config_error():
    """``except ConfigError`` catches ConfigSchemaError (Pydantic wrap)."""
    e = ConfigSchemaError("schema bad")
    assert isinstance(e, ConfigError)


def test_invariant_config_resolve_error_subclass_of_config_error():
    """``except ConfigError`` catches ConfigResolveError (import failure)."""
    e = ConfigResolveError("can't import")
    assert isinstance(e, ConfigError)


def test_invariant_config_error_subclass_of_exception():
    """ConfigError is a plain Exception subclass — pin so callers can use
    broad ``except Exception``.

    (Note: unlike LightTrainError, ConfigError is just Exception, NOT
    RuntimeError. CLI code paths catch ``(ConfigError, FileNotFoundError)``,
    relying on both being Exception subclasses.)
    """
    e = ConfigError("x")
    assert isinstance(e, Exception)


def test_invariant_registry_conflict_error_subclass_of_registry_error():
    """``except RegistryError`` catches RegistryConflictError."""
    e = RegistryConflictError("dup")
    assert isinstance(e, RegistryError)


def test_invariant_unknown_category_error_subclass_of_registry_error():
    """``except RegistryError`` catches UnknownCategoryError."""
    e = UnknownCategoryError("?")
    assert isinstance(e, RegistryError)


def test_invariant_not_registered_error_subclass_of_registry_error():
    """``except RegistryError`` catches NotRegisteredError."""
    e = NotRegisteredError("none")
    assert isinstance(e, RegistryError)


# ---------------------------------------------------------------------------
# Mutual disjointness — catching one sibling must not catch another
# ---------------------------------------------------------------------------

def test_invariant_registry_conflict_distinct_from_unknown_category():
    """Sibling exception types do NOT overlap: catching one does not catch
    the other (only the common base RegistryError catches both).

    Setup: raise RegistryConflictError, attempt to catch only UnknownCategoryError.
    Expected: NOT caught (propagates).
    """
    with pytest.raises(RegistryConflictError):
        try:
            raise RegistryConflictError("dup")
        except UnknownCategoryError:
            # If this except matched, the outer pytest.raises wouldn't see it.
            pass


def test_invariant_not_registered_distinct_from_conflict():
    """NotRegisteredError and RegistryConflictError do not inherit from each
    other — only from RegistryError.

    Setup: raise NotRegisteredError, attempt to catch RegistryConflictError.
    Expected: not caught.
    """
    with pytest.raises(NotRegisteredError):
        try:
            raise NotRegisteredError("missing")
        except RegistryConflictError:
            pass


def test_invariant_config_schema_error_distinct_from_resolve_error():
    """ConfigSchemaError and ConfigResolveError are sibling classes.

    Setup: raise ConfigSchemaError, attempt to catch ConfigResolveError.
    Expected: not caught.
    """
    with pytest.raises(ConfigSchemaError):
        try:
            raise ConfigSchemaError("bad")
        except ConfigResolveError:
            pass


# ---------------------------------------------------------------------------
# Public re-export surface (catches accidental removal of __all__ entries)
# ---------------------------------------------------------------------------

def test_invariant_lighttrain_exceptions_re_export_surface():
    """Pin: ``lighttrain.exceptions`` exposes the expected names.

    Goal: if a refactor removes BatchValidationError from the public surface
    (or renames it), this test fails to force a coordinated change.
    """
    import lighttrain.exceptions as mod
    public = set(getattr(mod, "__all__", ()))
    assert {"BatchValidationError", "LightTrainError"}.issubset(public)


def test_invariant_lighttrain_registry_re_exports_exception_types():
    """Pin: ``lighttrain.registry`` re-exports RegistryError + its three
    leaf exceptions (these are caught by user code without reaching into
    ``_exceptions``).
    """
    import lighttrain.registry as mod
    public = set(getattr(mod, "__all__", ()))
    for name in (
        "RegistryError",
        "RegistryConflictError",
        "UnknownCategoryError",
        "NotRegisteredError",
    ):
        assert name in public, f"{name} missing from lighttrain.registry __all__"


def test_invariant_lighttrain_config_re_exports_exception_types():
    """Pin: ``lighttrain.config`` re-exports the three config error types."""
    import lighttrain.config as mod
    public = set(getattr(mod, "__all__", ()))
    for name in ("ConfigError", "ConfigSchemaError", "ConfigResolveError"):
        assert name in public, f"{name} missing from lighttrain.config __all__"
