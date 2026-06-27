"""Tokenizer file output + mapping/seen JSON writers for the prune tool.

The "fingerprint" field of ``token_mapping.json`` and ``seen_ids.json`` is
the old tokenizer's ``vocab_size``. This is intentionally simpler than a
hash: it is sufficient for cross-batch ``--inherit-ids`` safety (using a
file saved against a different base tokenizer raises) without adding
hashing complexity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_pruned_tokenizer(
    old_tokenizer,
    new_bytes_list: list[bytes],
    new_merges: list,
    output_path: Path,
) -> None:
    """Write ``tokenizer.json`` / ``tokenizer_config.json`` /
    ``special_tokens_map.json`` / ``added_tokens.json`` (when present) for
    the pruned vocab.

    Ports ``voca-prune/main.py:save_tokenizer_json_all``:
      1. ``tokenizer.json``: replace ``model.vocab`` and ``model.merges`` of
         the backend tokenizer JSON; the rest (normalizer, pre_tokenizer,
         added_tokens) is preserved untouched.
      2. ``tokenizer_config.json``: copy of the old ``init_kwargs`` (with
         ``AddedToken`` objects JSON-serialized), vocab_size updated, and
         ``_name_or_path``/``name_or_path`` removed so it doesn't point back
         to the old path.
      3. ``special_tokens_map.json``: copy of the old special tokens map.
      4. ``added_tokens.json``: copy of the source dir's file, if it exists.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1.
    id2token_new = {
        i: new_bytes_list[i].decode("utf-8", errors="ignore")
        for i in range(len(new_bytes_list))
    }
    token2id_new = {v: k for k, v in id2token_new.items()}

    old_tok_json = json.loads(old_tokenizer.backend_tokenizer.to_str())
    old_tok_json["model"] = {
        "type": old_tok_json["model"].get("type", "BPE"),
        "dropout": None,
        "vocab": token2id_new,
        "merges": new_merges,
    }
    (output_path / "tokenizer.json").write_text(
        json.dumps(old_tok_json, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------ 2.
    def _convert_added_token(obj: Any) -> Any:
        if hasattr(obj, "__getstate__"):
            return obj.__getstate__()
        if isinstance(obj, list):
            return [_convert_added_token(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert_added_token(v) for k, v in obj.items()}
        return obj

    new_tokenizer_config = _convert_added_token(old_tokenizer.init_kwargs)
    if not isinstance(new_tokenizer_config, dict):
        new_tokenizer_config = {}
    new_tokenizer_config["vocab_size"] = len(token2id_new)
    for key in ("_name_or_path", "name_or_path"):
        new_tokenizer_config.pop(key, None)

    (output_path / "tokenizer_config.json").write_text(
        json.dumps(new_tokenizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ 3.
    (output_path / "special_tokens_map.json").write_text(
        json.dumps(old_tokenizer.special_tokens_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ 4.
    src_added = Path(old_tokenizer.name_or_path) / "added_tokens.json"
    if src_added.exists():
        (output_path / "added_tokens.json").write_bytes(src_added.read_bytes())


def save_mapping_and_seen(
    output_path: Path,
    mapping_new2old: list[int],
    tokenizer_fingerprint: str,
    seen: set[int],
) -> None:
    """Write ``token_mapping.json`` and ``seen_ids.json``.

    ``token_mapping.json`` is the authoritative new2old map consumed by
    ``--remap-embed`` and ``check-tokenizer``. ``seen_ids.json`` is the
    cross-batch inheritance file consumed by ``--inherit-ids``.

    ``tokenizer_fingerprint`` is the old vocab_size (see module docstring).
    """
    mapping_payload = {
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "new_vocab_size": len(mapping_new2old),
        "new2old": mapping_new2old,
    }
    (output_path / "token_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=False), encoding="utf-8"
    )

    seen_payload = {
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "ids": sorted(seen),
    }
    (output_path / "seen_ids.json").write_text(
        json.dumps(seen_payload, ensure_ascii=False), encoding="utf-8"
    )


def load_seen_ids_json(path: Path, *, expected_vocab_size: int) -> set[int]:
    """Load a ``seen_ids.json`` for ``--inherit-ids`` cross-batch union.

    Raise ``ValueError`` if the fingerprint (old vocab_size) doesn't match —
    prevents users from accidentally unioning with a file saved against a
    different base tokenizer.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("tokenizer_fingerprint")) != str(expected_vocab_size):
        raise ValueError(
            f"inherit-ids file '{path}' tokenizer fingerprint mismatch: "
            f"expected vocab_size={expected_vocab_size}, got "
            f"{payload.get('tokenizer_fingerprint')}"
        )
    return set(payload["ids"])


__all__ = [
    "load_seen_ids_json",
    "save_mapping_and_seen",
    "save_pruned_tokenizer",
]
