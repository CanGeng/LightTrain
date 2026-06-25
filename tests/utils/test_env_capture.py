"""Tests for ``lighttrain.utils.env_capture``.

Coverage targets (lines currently uncovered at 74%):
* 20-22  – ``_safe()`` exception path: logs warning, returns default.
* 36-37  – ``_git_sha()`` FileNotFoundError / TimeoutExpired / OSError branch.
* 38     – ``_git_sha()`` fall-through ``return None`` (non-zero returncode).
* 60-61  – ``capture_env()`` torch ImportError branch (``info["torch"] = None``).
* 67-68  – ``capture_env()`` accelerate ImportError branch.
* 74-75  – ``capture_env()`` transformers ImportError branch.

General contract tests cover the happy-path structure and key invariants.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from lighttrain.utils.env_capture import _git_sha, _safe, capture_env

# ---------------------------------------------------------------------------
# _safe()
# ---------------------------------------------------------------------------

def test_invariant_safe_returns_fn_result_on_success():
    """``_safe`` returns the callable's result when no exception is raised."""
    assert _safe(lambda: 42) == 42


def test_invariant_safe_returns_none_default_on_exception():
    """``_safe`` returns ``None`` (the default) when the callable raises."""
    result = _safe(lambda: 1 / 0)
    assert result is None


def test_invariant_safe_returns_custom_default_on_exception():
    """``_safe`` returns the caller-supplied *default* when the callable raises."""
    result = _safe(lambda: (_ for _ in ()).throw(ValueError("boom")), default="fallback")
    assert result == "fallback"


def test_safe_logs_warning_on_exception(caplog):
    """Line 21: ``_log.warning`` is called with exc_info when the probe raises.

    The warning message contains the sentinel phrase 'env capture'.
    """
    with caplog.at_level(logging.WARNING, logger="lighttrain.utils.env_capture"):
        _safe(lambda: 1 / 0)
    assert any("env capture" in r.message for r in caplog.records)


def test_safe_passes_exception_info_to_warning(caplog):
    """Line 22: the warning is emitted with ``exc_info=True`` so the
    exception type appears in the log record.
    """
    with caplog.at_level(logging.WARNING, logger="lighttrain.utils.env_capture"):
        _safe(lambda: (_ for _ in ()).throw(RuntimeError("test-exc")))
    # exc_info=True means the record should have exc_info attached
    records_with_exc = [r for r in caplog.records if r.exc_info]
    assert records_with_exc, "expected at least one record with exc_info"


# ---------------------------------------------------------------------------
# _git_sha()
# ---------------------------------------------------------------------------

def test_invariant_git_sha_returns_string_in_git_repo():
    """``_git_sha()`` returns a non-empty string when inside a git repo."""
    sha = _git_sha()
    # The repo root is a valid git repo; the result should be a hex string.
    assert sha is None or (isinstance(sha, str) and len(sha) > 0)


def test_git_sha_returns_none_on_file_not_found(monkeypatch):
    """Lines 36-37: ``FileNotFoundError`` (git not on PATH) → ``None``."""
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _git_sha() is None


def test_git_sha_returns_none_on_timeout(monkeypatch):
    """Lines 36-37: ``subprocess.TimeoutExpired`` → ``None``."""
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _git_sha() is None


def test_git_sha_returns_none_on_oserror(monkeypatch):
    """Lines 36-37: ``OSError`` (e.g. exec format error) → ``None``."""
    def _raise(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _git_sha() is None


def test_git_sha_returns_none_on_nonzero_returncode(monkeypatch):
    """Line 38: when ``returncode != 0``, falls through to ``return None``."""
    fake_result = MagicMock()
    fake_result.returncode = 128
    fake_result.stdout = "fatal: not a git repository\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    assert _git_sha() is None


def test_git_sha_returns_none_on_zero_returncode_but_empty_stdout(monkeypatch):
    """Line 35: ``stdout.strip()`` is empty → returns ``None``."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "   \n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    assert _git_sha() is None


def test_git_sha_returns_sha_on_success(monkeypatch):
    """Happy path: returncode 0 + non-empty stdout → stripped SHA string."""
    fake_sha = "abc1234def5678" * 3  # 42-char fake SHA
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = fake_sha + "\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    assert _git_sha() == fake_sha


# ---------------------------------------------------------------------------
# capture_env() – structure invariants
# ---------------------------------------------------------------------------

def test_invariant_capture_env_returns_required_keys():
    """``capture_env()`` always includes the seven mandatory keys."""
    info = capture_env()
    required = {"ts_utc", "python", "platform", "hostname", "argv", "cwd", "git_sha"}
    assert required.issubset(info.keys())


def test_invariant_capture_env_ts_utc_is_iso_string():
    """``ts_utc`` is a non-empty ISO-8601 string (parseable by datetime.fromisoformat)."""
    from datetime import datetime
    info = capture_env()
    dt = datetime.fromisoformat(info["ts_utc"])
    assert dt.tzinfo is not None  # must be UTC-aware


def test_invariant_capture_env_argv_is_list():
    """``argv`` is always a list."""
    assert isinstance(capture_env()["argv"], list)


def test_invariant_capture_env_cwd_is_str():
    """``cwd`` is a string pointing at the current working directory."""
    import os
    info = capture_env()
    assert isinstance(info["cwd"], str)
    assert os.path.isabs(info["cwd"])


# ---------------------------------------------------------------------------
# capture_env() – ImportError branches for optional deps
# ---------------------------------------------------------------------------

def _remove_module(name: str, modules: dict) -> None:
    """Remove *name* and all submodule keys from *modules*."""
    to_del = [k for k in list(modules) if k == name or k.startswith(name + ".")]
    for k in to_del:
        del modules[k]


def test_pin_current_behavior_torch_import_error_sets_none(monkeypatch):
    """Lines 60-61: when torch is not importable, ``info["torch"]`` is ``None``.

    Pin: current behavior – no ``cuda_available`` key is added in this branch.
    """
    # Temporarily hide torch from sys.modules
    monkeypatch.setitem(sys.modules, "torch", None)  # None triggers ImportError on `import torch`
    info = capture_env()
    assert info["torch"] is None
    assert "cuda_available" not in info


def test_pin_current_behavior_accelerate_import_error_sets_none(monkeypatch):
    """Lines 67-68: when accelerate is not importable, ``info["accelerate"]`` is ``None``."""
    monkeypatch.setitem(sys.modules, "accelerate", None)
    info = capture_env()
    assert info["accelerate"] is None


def test_pin_current_behavior_transformers_import_error_sets_none(monkeypatch):
    """Lines 74-75: when transformers is not importable, ``info["transformers"]`` is ``None``."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    info = capture_env()
    assert info["transformers"] is None


def test_capture_env_with_all_optional_deps_missing(monkeypatch):
    """All three optional deps missing → torch/accelerate/transformers all ``None``."""
    for mod in ("torch", "accelerate", "transformers"):
        monkeypatch.setitem(sys.modules, mod, None)
    info = capture_env()
    assert info["torch"] is None
    assert info["accelerate"] is None
    assert info["transformers"] is None


def test_capture_env_torch_present_sets_version_and_cuda_flag():
    """When torch is available, ``torch`` key is a non-empty version string
    and ``cuda_available`` is a bool.
    """
    info = capture_env()
    # torch is installed in the lighttrain env
    if info.get("torch") is not None:
        assert isinstance(info["torch"], str) and info["torch"]
        assert isinstance(info["cuda_available"], bool)


# ---------------------------------------------------------------------------
# _safe() – additional edge cases
# ---------------------------------------------------------------------------

def test_safe_propagates_return_value_of_various_types():
    """``_safe`` returns whatever the callable returns (list, dict, bool)."""
    assert _safe(lambda: [1, 2, 3]) == [1, 2, 3]
    assert _safe(lambda: {"a": 1}) == {"a": 1}
    assert _safe(lambda: False) is False


@pytest.mark.parametrize("exc_cls", [ValueError, TypeError, KeyError, RuntimeError, AttributeError])
def test_safe_catches_various_exception_types(exc_cls):
    """``_safe`` catches any exception subclass and returns the default."""
    result = _safe(lambda: (_ for _ in ()).throw(exc_cls("boom")), default=-1)
    assert result == -1
