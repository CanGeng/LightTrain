"""lighttrain prune-tokenizer / check-tokenizer CLI.

Lossless (bool-set) vocabulary pruning tool. Ports and tightens the
``voca-prune`` reference implementation (no chatml template, no frequency
counting, no lossy target-size path), exposed as two ``lighttrain`` CLI
subcommands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from lighttrain.builtin_plugins.data.core.tokenizers import QWEN3_BASELINE_DIR
from lighttrain.builtin_plugins.data.prune import check_tokenizer, prune_tokenizer


def prune_tokenizer_cmd(
    tokenizer: Path = typer.Option(
        QWEN3_BASELINE_DIR,
        "--tokenizer",
        help="Base tokenizer directory (defaults to the vendored Qwen3-0.6B).",
    ),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Corpus directory (.json/.jsonl/.txt, recursive)."
    ),
    support_lang: list[str] | None = typer.Option(
        None,
        "--support-lang",
        help="Keep languages (e.g. zh en). Required if --corpus is not given.",
    ),
    inherit_ids: Path | None = typer.Option(
        None,
        "--inherit-ids",
        help="Previous run's seen_ids.json for cross-batch set union.",
    ),
    remap_embed: Path | None = typer.Option(
        None,
        "--remap-embed",
        help="Base model dir (.safetensors); if given, also slice embedding/lm_head weights and remap configs.",
    ),
    out: Path = typer.Option(
        Path("./pruned_tok"), "--out", "-o", help="Output directory."
    ),
) -> None:
    """Prune a tokenizer losslessly from a corpus and/or language whitelist."""
    if corpus is None and not (support_lang or []):
        print("ERROR: must provide --corpus or --support-lang", file=sys.stderr)
        raise typer.Exit(1)
    prune_tokenizer(
        tokenizer_path=tokenizer,
        out=out,
        corpus=corpus,
        support_lang=support_lang or None,
        inherit_ids=inherit_ids,
        remap_embed=remap_embed,
    )
    print(f"==> Pruned tokenizer saved to {out}")


def check_tokenizer_cmd(
    old: Path = typer.Option(..., "--old", help="Original tokenizer directory."),
    new: Path = typer.Option(..., "--new", help="Pruned tokenizer directory."),
    corpus: Path = typer.Option(..., "--corpus", help="Equivalence-check corpus dir."),
) -> None:
    """Verify a pruned tokenizer encodes-equivalent to the original."""
    mismatch = check_tokenizer(
        old_tokenizer_path=old, new_tokenizer_path=new, corpus=corpus
    )
    if mismatch > 0:
        print(f"VERIFICATION FAILED: {mismatch} mismatches", file=sys.stderr)
        raise typer.Exit(1)
    print("VERIFIED: all samples encode-equivalent")
