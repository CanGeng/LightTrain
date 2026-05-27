"""Minimal .env loader for HF endpoint / token plumbing.

We avoid the python-dotenv dependency on purpose; the parser handles the 95%
case (KEY=VALUE, comments, blank lines, optional quoting). Existing
environment variables are not overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        # Strip inline comment only when value is unquoted.
        value = value.rstrip()
        if value and value[0] not in ("'", '"'):
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        out[key] = _strip_quotes(value)
    return out


def load_dotenv_if_present(cwd: Path | None = None) -> list[str]:
    """Load ``.env`` from ``cwd`` (or current working dir) into ``os.environ``.

    Returns the list of keys actually written (those not previously set).
    Silent no-op if the file is missing.
    """
    cwd = Path(cwd) if cwd else Path.cwd()
    path = cwd / ".env"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    written: list[str] = []
    for k, v in parse_dotenv(text).items():
        if k not in os.environ:
            os.environ[k] = v
            written.append(k)
    return written


__all__ = ["load_dotenv_if_present", "parse_dotenv"]
