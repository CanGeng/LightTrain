"""Tests for ``lighttrain prune-tokenizer`` / ``check-tokenizer`` CLI and the
underlying ``lighttrain.builtin_plugins.data.prune`` package.

Covers PLAN_v0.5.5 Block C / T3-T5:
  * T3 ``test_lossless_equivalence``: end-to-end CLI run on a tiny corpus,
    check-tokenizer exits 0.
  * T4 ``test_recurse_closure``: ``compute_seen_set`` closes the
    sub-fragment composition (every kept token's sub-fragments are kept).
  * T5 ``test_support_lang``: the language whitelist filter adds ids whose
    token-string is in ``support_lang`` and skips other languages; uses
    ``pytest.importorskip("langdetect")`` so the test is skipped when the
    optional dependency is absent (per Q18 decision — no new marker).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.builtin_plugins.data.core.tokenizers import QWEN3_BASELINE_DIR
from lighttrain.builtin_plugins.data.prune import prune_tokenizer
from lighttrain.builtin_plugins.data.prune.algorithm import (
    compute_seen_set,
    make_mapping,
)
from lighttrain.builtin_plugins.data.prune.langfilter import lang_filter
from lighttrain.cli._app import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# T3 — end-to-end lossless equivalence
# ---------------------------------------------------------------------------


def _build_corpus(corpus_dir: Path) -> None:
    """A small corpus with mixed .txt and .jsonl records."""
    (corpus_dir / "a.txt").write_text(
        "hello world\n"
        "foo bar baz qux\n"
        "the quick brown fox jumps over the lazy dog\n",
        encoding="utf-8",
    )
    (corpus_dir / "b.jsonl").write_text(
        json.dumps({"text": "The quick brown fox. The lazy dog."}) + "\n"
        + json.dumps({"prompt": "Is qux a real word?", "response": "Probably not."}) + "\n",
        encoding="utf-8",
    )


def test_prune_tokenizer_writes_tokenizer_files(tmp_path: Path) -> None:
    """``prune_tokenizer`` (python entry) writes the documented file group."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _build_corpus(corpus)
    out = tmp_path / "pruned"
    prune_tokenizer(tokenizer_path=QWEN3_BASELINE_DIR, out=out, corpus=corpus)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "token_mapping.json", "seen_ids.json"):
        assert (out / f).exists(), f"missing output file: {f}"


def test_prune_tokenizer_mapping_is_sorted_unique(tmp_path: Path) -> None:
    """``mapping_new2old`` is ascending and duplicate-free.

    Pin the contract used by ``check-tokenizer`` and ``--remap-embed``:
    strictly increasing old-id mapping (monotonic keeps BPE id compression).
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _build_corpus(corpus)
    out = tmp_path / "pruned"
    prune_tokenizer(tokenizer_path=QWEN3_BASELINE_DIR, out=out, corpus=corpus)
    m = json.loads((out / "token_mapping.json").read_text("utf-8"))["new2old"]
    assert m == sorted(m)
    assert len(set(m)) == len(m)


def test_lossless_equivalence_cli(runner: CliRunner, tmp_path: Path) -> None:
    """End-to-end: prune then check-tokenizer exits 0 on the same corpus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _build_corpus(corpus)
    out = tmp_path / "pruned"

    res = runner.invoke(
        app,
        [
            "prune-tokenizer",
            "--tokenizer", str(QWEN3_BASELINE_DIR),
            "--corpus", str(corpus),
            "--out", str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (out / "token_mapping.json").exists()

    res2 = runner.invoke(
        app,
        [
            "check-tokenizer",
            "--old", str(QWEN3_BASELINE_DIR),
            "--new", str(out),
            "--corpus", str(corpus),
        ],
    )
    assert res2.exit_code == 0, res2.stdout
    assert "VERIFIED" in res2.stdout


def test_prune_requires_corpus_or_support_lang(runner: CliRunner, tmp_path: Path) -> None:
    """``prune-tokenizer`` with neither --corpus nor --support-lang exits 1."""
    res = runner.invoke(
        app,
        ["prune-tokenizer", "--tokenizer", str(QWEN3_BASELINE_DIR),
         "--out", str(tmp_path / "x")],
    )
    assert res.exit_code == 1


def test_check_tokenizer_failure_first_on_tampered_mapping(tmp_path: Path) -> None:
    """``check-tokenizer`` reports mismatches when the token mapping is broken.

    Lossless equivalence against corpus holds under correct mapping; here we
    deliberately shrink ``new2old`` to a degenerate list (all zeros) so the
    mapped-back ids no longer match the original corpus encoding. The check
    must return >0 mismatches.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _build_corpus(corpus)
    out = tmp_path / "pruned"
    prune_tokenizer(tokenizer_path=QWEN3_BASELINE_DIR, out=out, corpus=corpus)
    # Tamper: zero out new2old so re-encoding can't match.
    payload = json.loads((out / "token_mapping.json").read_text("utf-8"))
    payload["new2old"] = [0] * len(payload["new2old"])
    (out / "token_mapping.json").write_text(json.dumps(payload), encoding="utf-8")

    from lighttrain.builtin_plugins.data.prune import check_tokenizer

    mismatches = check_tokenizer(
        old_tokenizer_path=QWEN3_BASELINE_DIR, new_tokenizer_path=out, corpus=corpus
    )
    assert mismatches > 0


# ---------------------------------------------------------------------------
# T4 — recurse closure property
# ---------------------------------------------------------------------------


def test_recurse_closure_subfragments_kept() -> None:
    """Every sub-fragment of a kept token that exists in the old vocab is kept."""

    # Construct a tiny byte-level "vocab": single bytes + a couple of merges.
    # id 0..4 = a, b, c, d, e (single bytes)
    # id 5    = "ab", id 6 = "abc"
    # seen = {6} only -> "abc" kept -> closure adds "ab", "b", "bc", "a", "c"
    old_bytes: list[bytes] = [
        b"a", b"b", b"c", b"d", b"e", b"ab", b"abc",
    ]

    # "abc" (id 6) is the only seen-from-corpus token; closure should pull in
    # every subset-substring that exists in the vocab: "ab" (5), "a" (0),
    # "b" (1), "bc" (not present, skipped), "c" (2).
    seen = compute_seen_set(
        vocab_size=len(old_bytes),
        old_bytes_list=old_bytes,
        corpus_ids=[{6}],   # one text covering just id 6
    )
    # Closed set: 6 + every available sub-fragment of b"abc".
    assert seen == {0, 1, 2, 5, 6}, seen
    # Specials branch: range(len(old_bytes), vocab_size) is empty here.

    mapping = make_mapping(seen, vocab_size=len(old_bytes))
    assert mapping == sorted(seen)


def test_specials_branch_force_kept_when_outside_bytes_list() -> None:
    """Ids >= ``len(old_bytes_list)`` are force-retained by the guard branch."""
    old_bytes = [b"a", b"b"]  # two normal tokens; vocab_size claims 4
    seen = compute_seen_set(
        vocab_size=4, old_bytes_list=old_bytes, corpus_ids=[{0}]
    )
    # 0 kept by corpus; sub-fragments of b"a" (len 1) loop skipped; specials
    # 2,3 force-added by the range(2, 4) loop.
    assert seen == {0, 2, 3}


# ---------------------------------------------------------------------------
# T5 — --support-lang filter (langdetect optional dep)
# ---------------------------------------------------------------------------


def test_support_lang_filter_zh_vs_en_skips_other_languages() -> None:
    """``lang_filter`` retains tokens whose language is in ``support_lang``,
    and always keeps ``<...>`` / ``[...]`` placeholders.

    ``langdetect`` emits language codes like ``zh-cn`` (not bare ``zh``);
    users pass the codes exactly as langdetect produces them — this pins
    voca-prune's direct-membership matching, no prefix normalization.

    Uses ``pytest.importorskip`` so the test is skipped cleanly when
    ``langdetect`` is not installed (lighttrain's prune extras).
    """
    pytest.importorskip("langdetect")

    # byte forms of representative token strings:
    # "的" (zh-cn), "是" (zh-cn), "the" (en), "le" (en-ish),
    # "<|im_start|>" (special), invalid utf-8 byte (silently dropped)
    old_bytes = [
        "的".encode(),          # 0 -> langdetect: zh-cn
        "是".encode(),          # 1 -> zh-cn
        b"the",         # 2 -> en
        b"\xff",                       # 3 invalid utf-8 — skipped silently
        b"<|im_start|>",  # 4 special placeholder
    ]
    seen = lang_filter(old_bytes, support_lang=["zh-cn"])
    # zh-cn ids 0,1 retained; en id 2 not matched; id 3 silently dropped;
    # id 4 special auto-kept (handled before langdetect call).
    assert 0 in seen and 1 in seen
    assert 4 in seen
    assert 2 not in seen
    assert 3 not in seen


def test_support_lang_filter_missing_dep_raises_with_help_message(monkeypatch) -> None:
    """``lang_filter`` raises ``ImportError`` pointing at ``lighttrain[prune]``
    when ``langdetect`` is not installed.

    The error message is the actionable install instruction users see.
    """
    import builtins

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "langdetect" or name.startswith("langdetect."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    # Also clear any cached langdetect from prior tests.
    import sys

    monkeypatch.delitem(sys.modules, "langdetect", raising=False)
    for mod in list(sys.modules):
        if mod.startswith("langdetect"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError, match="lighttrain\\[prune\\]"):
        lang_filter([b"the"], support_lang=["en"])
