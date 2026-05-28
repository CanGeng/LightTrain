"""``train --apply-degrade patch.yaml`` flattens to overrides (DESIGN §18.4)."""

from __future__ import annotations

from lighttrain.cli._app import _flatten_patch_to_overrides


def test_flatten_nested_dict():
    patch = {
        "engine": {"mixed_precision": "bf16"},
        "trainer": {"accumulate": 2, "grad_clip": 1.0},
        "training_tricks": {"gradient_checkpointing": True},
    }
    overrides = _flatten_patch_to_overrides(patch)
    assert "++engine.mixed_precision=bf16" in overrides
    assert "++trainer.accumulate=2" in overrides
    assert "++trainer.grad_clip=1.0" in overrides
    assert "++training_tricks.gradient_checkpointing=True" in overrides


def test_flatten_handles_none_and_lists():
    patch = {"a": None, "b": [1, 2, 3]}
    overrides = _flatten_patch_to_overrides(patch)
    assert "++a=null" in overrides
    # YAML-encoded list value.
    assert any(o.startswith("++b=") for o in overrides)


def test_flatten_ignores_non_dict():
    assert _flatten_patch_to_overrides("scalar") == []
    assert _flatten_patch_to_overrides(None) == []
