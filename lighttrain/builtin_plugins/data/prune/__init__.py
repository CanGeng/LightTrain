"""Lossless bool-set vocabulary pruning for lighttrain.

Two orchestration entry points:

* :func:`prune_tokenizer` — build a pruned tokenizer from a corpus (or
  ``--support-lang`` whitelist + optional ``--inherit-ids`` cross-batch
  union) and write the new tokenizer files plus ``token_mapping.json`` /
  ``seen_ids.json``. Optionally also slice model weights
  (``--remap-embed``).
* :func:`check_tokenizer` — equivalence verification: re-encode every
  corpus text with the new tokenizer and assert
  ``new_ids[k] -> new2old[new_ids[k]] == old_ids[k]``. Returns the
  mismatch count; the CLI exits non-zero on any mismatch.

Algorithm at a glance: :func:`compute_seen_set` collects retained ids via
bool set union (corpus + lang filter + inherit-ids), then closes the
sub-fragment composition via a fixpoint worklist. ``mapping_new2old`` is
sorted by old id (preserves BPE id compression). See
``PLAN_v0.5.5.md`` Block C for the design rationale.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

from .algorithm import compute_seen_set, make_mapping, prune_merges
from .corpus import iter_corpus_texts
from .langfilter import lang_filter
from .remap import remap_config_and_generation, remap_embed_and_lm_heads
from .save import save_mapping_and_seen, save_pruned_tokenizer

__all__ = ["check_tokenizer", "prune_tokenizer"]


def prune_tokenizer(
    *,
    tokenizer_path: Path,
    out: Path,
    corpus: Path | None = None,
    support_lang: list[str] | None = None,
    inherit_ids: Path | None = None,
    remap_embed: Path | None = None,
) -> None:
    """Run the lossless prune pipeline and write outputs under ``out``."""
    from transformers import AutoTokenizer

    old_tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True
    )
    vocab_size = len(old_tokenizer)
    old_bytes_list = [
        tok.encode("utf-8")
        for tok, _ in sorted(old_tokenizer.get_vocab().items(), key=lambda x: x[1])
    ]

    corpus_ids: Iterator[set[int]] | None = None
    if corpus is not None:
        def _iter() -> Iterator[set[int]]:
            for text in iter_corpus_texts(corpus):
                yield set(old_tokenizer.encode(text))

        corpus_ids = _iter()

    support_lang_ids: set[int] | None = None
    if support_lang:
        support_lang_ids = lang_filter(old_bytes_list, support_lang)

    seen = compute_seen_set(
        vocab_size=vocab_size,
        old_bytes_list=old_bytes_list,
        corpus_ids=corpus_ids,
        support_lang_ids=support_lang_ids,
        inherit_ids=inherit_ids,
    )

    mapping_new2old = make_mapping(seen, vocab_size)
    new_bytes_list = [old_bytes_list[i] for i in mapping_new2old]
    new_merges = prune_merges(old_tokenizer, old_bytes_list, seen)

    out.mkdir(parents=True, exist_ok=True)
    save_pruned_tokenizer(old_tokenizer, new_bytes_list, new_merges, out)
    save_mapping_and_seen(out, mapping_new2old, str(vocab_size), seen)

    if remap_embed is not None:
        remap_embed_and_lm_heads(remap_embed, out, mapping_new2old)
        remap_config_and_generation(remap_embed, out, mapping_new2old)


def check_tokenizer(
    *,
    old_tokenizer_path: Path,
    new_tokenizer_path: Path,
    corpus: Path,
) -> int:
    """Re-encode every corpus text and count mismatches against ``token_mapping.json``.

    Returns the mismatch count (``0`` means equivalent). The CLI wraps this
    into a non-zero exit code when the count is positive (failure-first).
    """
    import json

    from transformers import AutoTokenizer

    old_tok = AutoTokenizer.from_pretrained(
        str(old_tokenizer_path), trust_remote_code=True
    )
    new_tok = AutoTokenizer.from_pretrained(
        str(new_tokenizer_path), trust_remote_code=True
    )
    mapping = json.loads((new_tokenizer_path / "token_mapping.json").read_text("utf-8"))
    new2old = mapping["new2old"]

    mismatch_count = 0
    for text in iter_corpus_texts(corpus):
        old_ids = old_tok.encode(text)
        new_ids = new_tok.encode(text)
        if len(old_ids) != len(new_ids):
            mismatch_count += 1
            _print_mismatch(text, old_ids, new_ids, new2old)
            continue
        for old_id, new_id in zip(old_ids, new_ids, strict=True):
            if new_id >= len(new2old) or old_id != new2old[new_id]:
                mismatch_count += 1
                _print_mismatch(text, old_ids, new_ids, new2old)
                break
    return mismatch_count


def _print_mismatch(
    text: str,
    old_ids: list[int],
    new_ids: list[int],
    new2old: list[int],
) -> None:
    snippet = text[:80].replace("\n", " ")
    mapped = [new2old[i] if i < len(new2old) else -1 for i in new_ids]
    first_diff = next(
        (p for p, (a, b) in enumerate(zip(old_ids, mapped, strict=False)) if a != b),
        None,
    )
    print(
        f"  MISMATCH: text={snippet!r}\n"
        f"    old_ids (len={len(old_ids)}): {old_ids[:10]}...\n"
        f"    new_ids (len={len(new_ids)}): {new_ids[:10]}...\n"
        f"    new2old mapped: {mapped[:10]}...\n"
        f"    first divergence at position {first_diff}",
        file=sys.stderr,
    )
