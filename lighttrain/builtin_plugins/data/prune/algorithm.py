"""Lossless bool-set vocabulary pruning algorithm.

Two stages:
  1. :func:`compute_seen_set` — collect token ids that appear *or* are
     needed as sub-fragments of retained tokens (fixpoint worklist closure).
  2. :func:`make_mapping`&nbsp;→ :func:`prune_merges` — derive the new2old id
     list (ascending old ids, preserving BPE id compression) and the subset
     of BPE merge rules that remains valid.

Counting is **bool** (set union), not frequency — lossless pruning has no
need for a frequency table, so the metric is just "did this token appear in
any retained text, or appears as a sub-fragment of one that did?".
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def compute_seen_set(
    *,
    vocab_size: int,
    old_bytes_list: list[bytes],
    corpus_ids: Iterable[set[int]] | None = None,
    support_lang_ids: set[int] | None = None,
    inherit_ids: Path | None = None,
) -> set[int]:
    """Build the set of old ids to retain.

    Sources combined via set union:
      * ``corpus_ids``: per-text encoded id-sets (one set per text).
      * ``support_lang_ids``: ids matched by the language whitelist filter.
      * ``inherit_ids``: a previously-saved ``seen_ids.json`` (cross-batch
        union) — see :func:`lighttrain.builtin_plugins.data.prune.save.load_seen_ids_json`.

    Then runs a fixpoint worklist closure: every retained token of byte
    length > 1 contributes all of its sub-byte-fragments that are also
    standalone vocab tokens. This keeps the *composition closure* intact
    so any retained token can still be encoded as the same id sequence.

    Finally, specials (ids >= ``len(old_bytes_list)``) are force-retained
    defensively — for HF tokenizers ``get_vocab()`` already includes specials
    in ``old_bytes_list`` (so this branch is a no-op); the guard exists for
    tokenizers that report specials separately.
    """
    seen: set[int] = set()

    if corpus_ids is not None:
        for ids in corpus_ids:
            seen |= ids
    if support_lang_ids is not None:
        seen |= support_lang_ids
    if inherit_ids is not None:
        from .save import load_seen_ids_json

        seen |= load_seen_ids_json(inherit_ids, expected_vocab_size=vocab_size)

    # Fixpoint worklist closure over sub-fragments.
    bytes_to_index: dict[bytes, int] = {b: i for i, b in enumerate(old_bytes_list)}
    work: list[int] = list(seen)
    while work:
        i = work.pop()
        b = old_bytes_list[i]
        n = len(b)
        if n <= 1:
            continue
        for start in range(0, n):
            for end in range(start + 1, n + 1):
                j = bytes_to_index.get(b[start:end])
                if j is not None and j not in seen:
                    seen.add(j)
                    work.append(j)

    # Defensive special-token retention (no-op when get_vocab already
    # includes specials — the common HF case).
    for i in range(len(old_bytes_list), vocab_size):
        seen.add(i)
    return seen


def make_mapping(seen: set[int], vocab_size: int) -> list[int]:
    """Return ``new2old``: new id -> old id, sorted ascending by old id.

    Sorting by old id preserves BPE id compression (low-frequency chars
    keep low ids; the new mapping's monotonic ordering is also required by
    :func:`check_tokenizer` to re-derive equivalence with the old tokenizer.
    """
    _ = vocab_size  # vocab_size only narrows the universe; `seen` is already filtered
    return sorted(seen)


def prune_merges(
    old_tokenizer, old_bytes_list: list[bytes], seen: set[int]
) -> list:
    """Return the kept merge rules from the old tokenizer's BPE merges.

    A merge ``[p1, p2] -> p1+p2`` is kept iff all three tokens have retained
    ids in ``seen``. Supports both the list form (``[p1, p2]``) and the
    string form (``"p1 p2"``).
    """
    old_tok_json = json.loads(old_tokenizer.backend_tokenizer.to_str())
    old_vocab_str2id = old_tokenizer.get_vocab()

    if "model" not in old_tok_json or "merges" not in old_tok_json["model"]:
        return []

    new_merges: list = []
    for merge_rule in old_tok_json["model"]["merges"]:
        if isinstance(merge_rule, list) and len(merge_rule) == 2:
            p1_str, p2_str = merge_rule[0], merge_rule[1]
        elif isinstance(merge_rule, str) and " " in merge_rule:
            p1_str, p2_str = merge_rule.split(" ", 1)
        else:
            continue
        p1_id = old_vocab_str2id.get(p1_str)
        p2_id = old_vocab_str2id.get(p2_str)
        merged_id = old_vocab_str2id.get(p1_str + p2_str)
        if p1_id is None or p2_id is None or merged_id is None:
            continue
        if p1_id in seen and p2_id in seen and merged_id in seen:
            new_merges.append(merge_rule)
    return new_merges


__all__ = ["compute_seen_set", "make_mapping", "prune_merges"]
