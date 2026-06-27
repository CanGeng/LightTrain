"""Corpus reader for the prune tool.

Walks a directory tree recursively and yields ``str`` lines/text fields from
``.json`` / ``.jsonl`` / ``.txt`` files. Recognized corpus keys are the seven
output-side fields commonly produced by alpaca / sharegpt / openassistant
datasets; chatml-style template concatenation is intentionally NOT done here
(prune tooling should not inject chat token conventions — users supply their
own corpus layout).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

# unified seven corpus keys (drops the legacy "intruction" typo from
# voca-prune and adds the supervised text/raw-prompt key).
CORPUS_KEYS: tuple[str, ...] = (
    "text",
    "prompt",
    "query",
    "response",
    "instruction",
    "input",
    "output",
)


def iter_corpus_texts(corpus_dir: Path) -> Iterator[str]:
    """Yield text strings from every recognized corpus file under ``corpus_dir``."""
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield line
        elif suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield from _extract_keys(item)
            elif isinstance(data, dict):
                yield from _extract_keys(data)
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield from _extract_keys(obj)


def _extract_keys(obj: dict) -> Iterator[str]:
    for k in CORPUS_KEYS:
        v = obj.get(k)
        if isinstance(v, str):
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    yield x


__all__ = ["CORPUS_KEYS", "iter_corpus_texts"]
