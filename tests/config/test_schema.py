"""Adversarial tests for ``lighttrain.config._schema``.

Coverage:

* ``ComponentSpec`` XOR enforcement (both/neither set → ValidationError).
* ``ComponentSpec`` populate-by-name alias ``_target_`` ↔ ``target``.
* ``RootConfig`` extra-allow pin.
* ``RootConfig`` ``mode`` is validated at construction.
* ``dump_resolved`` round-trips (text-level equality on key fields).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lighttrain.config import (
    ComponentSpec,
    ConfigSchemaError,
    ParallelSection,
    dump_resolved,
    load_config,
)


def test_componentspec_xor_neither_raises():
    """ComponentSpec with neither ``name`` nor ``target`` is rejected.

    Input: ``ComponentSpec()``.
    Expected: ValidationError (Pydantic v2 wraps the model_validator failure).
    """
    with pytest.raises(ValidationError):
        ComponentSpec()


def test_componentspec_xor_both_raises():
    """ComponentSpec with BOTH ``name`` and ``_target_`` is rejected.

    Input: ``ComponentSpec(name='x', _target_='pkg.Cls')``.
    Expected: ValidationError.
    """
    with pytest.raises(ValidationError):
        ComponentSpec(name="x", _target_="pkg.Cls")


def test_componentspec_short_name_with_params():
    """``ComponentSpec(name=..., params=...)`` constructs cleanly.

    Input: name='adamw', params={'lr': 1e-4}.
    Expected: round-trip on ``name`` and ``params`` attrs.
    """
    cs = ComponentSpec(name="adamw", params={"lr": 1e-4})
    assert cs.name == "adamw"
    assert cs.params == {"lr": 1e-4}
    assert cs.target is None


def test_componentspec_target_alias_via_underscored_form():
    """The ``_target_`` alias populates the ``target`` field
    (``populate_by_name=True``).

    Input: ComponentSpec via the ``_target_`` keyword.
    Expected: ``cs.target == 'pkg.Cls'``.
    """
    cs = ComponentSpec(_target_="pkg.Cls", params={"a": 1})
    assert cs.target == "pkg.Cls"
    assert cs.params == {"a": 1}


def test_componentspec_target_via_canonical_field_name():
    """The canonical ``target`` field name also works
    (``populate_by_name=True``).

    Input: ComponentSpec via ``target=`` (no underscores).
    Expected: ``cs.target == 'pkg.Cls'``.
    """
    cs = ComponentSpec(target="pkg.Cls")
    assert cs.target == "pkg.Cls"


def test_pin_root_config_extra_allow(tmp_yaml):
    """Pin: ``RootConfig`` has ``extra='allow'``, so unknown top-level keys
    are kept (not rejected).

    Setup: YAML with experimental field ``my_experimental_field: 42``.
    Expected: cfg.my_experimental_field == 42, no validation error.

    If this is intentionally changed (e.g. to ``extra='forbid'``), update
    this test AND document the breaking change.
    """
    p = tmp_yaml("mode: lab\nmy_experimental_field: 42\n")
    cfg = load_config(p)
    assert cfg.my_experimental_field == 42  # type: ignore[attr-defined]


def test_root_config_invalid_mode_raises_at_load(tmp_yaml):
    """``mode: bogus`` is rejected by the Literal['lab','prod'] validator.

    Input: YAML with ``mode: bogus``.
    Expected: ConfigSchemaError (loader wraps Pydantic ValidationError).
    """
    p = tmp_yaml("mode: bogus\n")
    with pytest.raises(ConfigSchemaError):
        load_config(p)


def test_root_config_seed_non_int_raises(tmp_yaml):
    """``seed`` must be an int.

    Input: YAML with ``seed: not-a-number``.
    Expected: ConfigSchemaError.
    """
    p = tmp_yaml("seed: not-a-number\n")
    with pytest.raises(ConfigSchemaError):
        load_config(p)


def test_parallel_section_defaults_yield_single_gpu():
    """ParallelSection construction with no overrides matches single-GPU defaults.

    Input: ``ParallelSection()``.
    Expected: dp==tp==pp==ep==1, backend=='nccl', force_cpu==False.
    """
    p = ParallelSection()
    assert p.dp == 1 and p.tp == 1 and p.pp == 1 and p.ep == 1
    assert p.backend == "nccl"
    assert p.force_cpu is False


def test_dump_resolved_round_trip_preserves_mode_and_seed(tmp_yaml):
    """``load_config → dump_resolved`` text contains the original key fields.

    Goal: Verify resolved cfg can be serialized back to YAML, and the result
    contains the loaded values.
    Input: YAML with ``mode: prod, seed: 5``.
    Expected: text representation includes both literal lines.
    """
    p = tmp_yaml("mode: prod\nseed: 5\n")
    cfg = load_config(p)
    text = dump_resolved(cfg)
    assert "mode: prod" in text
    assert "seed: 5" in text


def test_root_config_with_typed_trainer_section(tmp_yaml):
    """Typed TrainerSection is constructed from a YAML sub-mapping.

    Input: YAML with a ``trainer`` sub-mapping setting max_steps=123.
    Expected: cfg.trainer.max_steps == 123 and cfg.trainer.grad_clip default = 1.0.
    """
    p = tmp_yaml("trainer:\n  max_steps: 123\n  name: pretrain\n")
    cfg = load_config(p)
    assert cfg.trainer is not None
    assert cfg.trainer.max_steps == 123
    assert cfg.trainer.grad_clip == pytest.approx(1.0)
