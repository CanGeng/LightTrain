"""Verify ``cfg.engine.mixed_precision`` actually wires an Accelerator into
the StepContext (REVIEW #4 / DESIGN §11)."""

from __future__ import annotations

import pytest

from lighttrain.config._exceptions import ConfigError
from lighttrain.utils.accelerate import build_accelerator


def test_build_accelerator_none_for_no_mode():
    assert build_accelerator("no") is None
    assert build_accelerator("none") is None
    assert build_accelerator("") is None


def test_build_accelerator_bf16_returns_object():
    acc = build_accelerator("bf16", gradient_accumulation_steps=2)
    assert acc is not None
    # accelerator should expose the API surface StandardUpdateRule uses
    assert hasattr(acc, "autocast")
    assert hasattr(acc, "backward")
    assert hasattr(acc, "clip_grad_norm_")


def test_build_accelerator_rejects_garbage():
    with pytest.raises(ConfigError):
        build_accelerator("garbage")
