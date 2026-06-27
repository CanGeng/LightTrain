"""Tests for vLLM docstring tone alignment (PLAN_v0.5.5 Block E).

Pins the migration from "stub" wording (which undersold an actually-complete
backend) to "available when installed" — consistent with lighttrain's
fail-loud philosophy (the backend raises ``ImportError`` at construction
time when ``vllm`` is absent, never silently no-ops).

* ``test_module_docstring_says_available_not_stub``
* ``test_class_docstring_says_available_not_stub``
* ``test_registry_short_name_unchanged``
"""
from __future__ import annotations

from pathlib import Path

_VLLM_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lighttrain" / "builtin_plugins" / "rl" / "backends" / "vllm" / "__init__.py"
)


def _read_source() -> str:
    assert _VLLM_MODULE_PATH.exists(), f"missing {_VLLM_MODULE_PATH}"
    return _VLLM_MODULE_PATH.read_text(encoding="utf-8")


def test_module_docstring_says_available_not_stub() -> None:
    """Module docstring no longer self-describes as a ``stub``."""
    src = _read_source()
    assert "stub" not in src.lower(), (
        "vllm backend module still self-describes as 'stub'; "
        "per Block E it should say 'available when installed' instead"
    )
    assert "available when" in src.lower() or "available when ``vllm`` is installed" in src


def test_class_docstring_says_constructor_raises_without_vllm() -> None:
    """Class docstring frames the missing-dep failure as a constructor-level
    ``ImportError``, not a permanent stub identity."""
    src = _read_source()
    assert "Constructing this backend without" in src or "without ``vllm`` installed raises" in src


def test_registry_short_name_unchanged() -> None:
    """``@register("rl_backend", "vllm")`` short name is preserved — the CLI /
    recipes depend on it and Block E explicitly must not perturb that."""
    src = _read_source()
    assert '@register("rl_backend", "vllm")' in src
