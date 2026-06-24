"""Shared diagnostic-failure helpers for product-presence assertions.

When a test asserts a callback/diagnostic artifact was produced and it was NOT,
a bare ``assert files`` / ``assert path.exists()`` fails with an opaque empty
list or ``False``. The ``expect_*`` helpers fold the product directory's
``rglob('*')`` tree (and optionally captured ``lighttrain`` WARNING logs / CLI
stdout) into the AssertionError so failures self-diagnose — mirroring the
``frozen_step`` template at ``tests/cli/test_cli_freeze_replay.py``.

Pass/fail semantics are unchanged:
  expect_nonempty(coll, root)  == assert coll
  expect_count(coll, n, root)  == assert len(coll) == n
  expect_exists(path, root)    == assert path is not None and path.exists()
"""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any


def _tree(root: Path) -> str:
    if not root.exists():
        return f"(missing: {root} does not exist)"
    entries = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    return "\n".join(entries) if entries else "(empty)"


def _raise(summary: str, root: Path, *, caplog: Any | None, stdout: str | None) -> None:
    sections = [summary]
    if caplog is not None:
        sections.append(f"--- lighttrain WARNING logs ---\n{caplog.text or '(none)'}")
    sections.append(f"--- product tree ({root.name}) ---\n{_tree(root)}")
    if stdout is not None:
        sections.append(f"--- stdout ---\n{stdout}")
    raise AssertionError("\n".join(sections))


def expect_nonempty(
    coll: Sized,
    root: str | Path,
    *,
    what: str,
    caplog: Any | None = None,
    stdout: str | None = None,
) -> None:
    """Assert ``coll`` is non-empty (``assert coll``); else dump ``root``'s tree."""
    if len(coll) > 0:
        return
    _raise(
        f"expected {what}, but the collection is empty.",
        Path(root),
        caplog=caplog,
        stdout=stdout,
    )


def expect_count(
    coll: Sized,
    n: int,
    root: str | Path,
    *,
    what: str,
    caplog: Any | None = None,
    stdout: str | None = None,
) -> None:
    """Assert ``len(coll) == n``; else dump ``root``'s tree."""
    if len(coll) == n:
        return
    _raise(
        f"expected exactly {n} {what}, got {len(coll)}.",
        Path(root),
        caplog=caplog,
        stdout=stdout,
    )


def expect_exists(
    path: str | Path | None,
    root: str | Path,
    *,
    what: str,
    caplog: Any | None = None,
    stdout: str | None = None,
) -> None:
    """Assert ``path is not None and path.exists()``; else dump ``root``'s tree."""
    if path is not None and Path(path).exists():
        return
    _raise(
        f"expected {what} at {path!s}, but it does not exist.",
        Path(root),
        caplog=caplog,
        stdout=stdout,
    )
