"""Per-token language whitelist filter (``--support-lang``).

Adds a token id to ``seen`` iff its string form is detected by
``langdetect`` as one of the supported languages, or the token is a
``<...>`` / ``[...]`` special placeholder. Specials are auto-retained so
the chat template scaffolding (``<|im_start|>`` etc.) survives pruning even
when no corpus path is given.

``langdetect`` is an optional dependency — :func:`lang_filter` raises
``ImportError`` with an actionable message when missing (install
``lighttrain[prune]``).
"""
from __future__ import annotations


def _import_langdetect_or_raise() -> None:
    try:
        import langdetect  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "--support-lang requires 'langdetect'. Install it with: "
            "pip install 'lighttrain[prune]'"
        ) from exc


def _is_special_token(s: str) -> bool:
    return (
        (s.startswith("<") and s.endswith(">") and len(s) > 2)
        or (s.startswith("[") and s.endswith("]") and len(s) > 2)
    )


def lang_filter(old_bytes_list: list[bytes], support_lang: list[str]) -> set[int]:
    """Return ids whose token-string is one of ``support_lang`` (or a special).

    ``DetectorFactory.seed=0`` is set for reproducibility.
    """
    _import_langdetect_or_raise()
    from langdetect import DetectorFactory
    from langdetect import detect as langdetect_detect

    DetectorFactory.seed = 0

    seen: set[int] = set()
    for i, b in enumerate(old_bytes_list):
        try:
            token_str = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _is_special_token(token_str):
            seen.add(i)
            continue
        try:
            if langdetect_detect(token_str) in support_lang:
                seen.add(i)
        except Exception:  # noqa: BLE001 langdetect raises LangDetectException on random bytes
            pass
    return seen


__all__ = ["lang_filter"]
