"""Edge-case tests for ``lighttrain.data.cache._shards``.

Coverage / invariants pinned:

* **cache_key** is deterministic, 16-hex chars, varies with every field,
  and treats ``chat_template=None`` identically to ``chat_template=""``.
* **ShardWriter.__post_init__** creates ``out_dir`` and clamps
  ``shard_size`` to ``>= 1`` (including from 0 / negative).
* **write / write_many** auto-flush once ``shard_size`` rows accumulate.
* **finalize** writes ``shards.json`` manifest (fmt / shards / total_rows)
  and flushes a trailing partial shard.
* **_flush** writes JSONL by default and Parquet when ``fmt="parquet"``.
* **read_manifest** → None on missing file, None on invalid JSON, dict otherwise.
* **iter_rows** round-trips JSONL + Parquet shards via the manifest, scans
  loose ``shard-*.jsonl`` lexicographically when no manifest exists, and skips
  blank lines.
* **shard_state** / **count_rows** read from the manifest with safe defaults.

Note: the ``fmt == "parquet" and not _HAS_PARQUET`` jsonl-fallback (line 79)
is skipped — pyarrow is installed in this env so the guard is unreachable
without monkeypatching the module flag.
"""

from __future__ import annotations

import json

import pytest

from lighttrain.data.cache import _shards
from lighttrain.data.cache._shards import (
    ShardWriter,
    cache_key,
    count_rows,
    iter_rows,
    read_manifest,
    shard_state,
)

# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_invariant_cache_key_is_deterministic_16_hex():
    """``cache_key`` returns the same 16-char lowercase-hex digest for equal args."""
    a = cache_key(tokenizer="gpt2")
    b = cache_key(tokenizer="gpt2")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokenizer": "llama"},
        {"tokenizer": "gpt2", "chat_template": "{{x}}"},
        {"tokenizer": "gpt2", "raw_data_version": "1"},
        {"tokenizer": "gpt2", "preprocess_code": "strip()"},
    ],
)
def test_invariant_cache_key_varies_with_each_field(kwargs):
    """Changing any single field changes the digest (line 47-55 payload)."""
    baseline = cache_key(tokenizer="gpt2")
    assert cache_key(**kwargs) != baseline


def test_invariant_cache_key_none_chat_template_equals_empty():
    """``chat_template=None`` and ``""`` collapse to the same key (line 49)."""
    assert cache_key(tokenizer="t", chat_template=None) == cache_key(
        tokenizer="t", chat_template=""
    )


def test_invariant_cache_key_matches_manual_sha256():
    """The digest is the 16-char prefix of sha256 over the sorted compact JSON."""
    import hashlib

    payload = {
        "tokenizer": "gpt2",
        "chat_template": "",
        "raw_data_version": "0",
        "preprocess_code": "",
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert cache_key(tokenizer="gpt2") == expected


# ---------------------------------------------------------------------------
# ShardWriter.__post_init__
# ---------------------------------------------------------------------------


def test_invariant_post_init_creates_out_dir(tmp_path):
    """``__post_init__`` makes the (nested) output directory."""
    target = tmp_path / "nested" / "cache"
    assert not target.exists()
    ShardWriter(target)
    assert target.is_dir()


@pytest.mark.parametrize("raw,expected", [(0, 1), (-5, 1), (1, 1), (7, 7)])
def test_invariant_post_init_clamps_shard_size_to_min_one(tmp_path, raw, expected):
    """``shard_size`` is clamped to ``max(1, int(...))`` (line 80)."""
    w = ShardWriter(tmp_path, shard_size=raw)
    assert w.shard_size == expected


# ---------------------------------------------------------------------------
# write / write_many auto-flush
# ---------------------------------------------------------------------------


def test_invariant_write_autoflushes_at_shard_size(tmp_path):
    """``write`` triggers ``_flush`` once ``shard_size`` rows accrue (line 85-86)."""
    w = ShardWriter(tmp_path, shard_size=2)
    w.write({"i": 0})
    assert w._shard_idx == 0  # not yet flushed
    w.write({"i": 1})  # hits the threshold → flush
    assert w._shard_idx == 1
    assert w._current == []
    assert (tmp_path / "shard-00000.jsonl").exists()


def test_invariant_write_many_flushes_across_multiple_shards(tmp_path):
    """``write_many`` rolls over shards; total_rows tracks every row."""
    w = ShardWriter(tmp_path, shard_size=2)
    w.write_many([{"i": k} for k in range(5)])
    assert w._total_rows == 5
    assert w._shard_idx == 2  # two full shards flushed
    assert len(w._current) == 1  # trailing partial still buffered


# ---------------------------------------------------------------------------
# finalize + manifest
# ---------------------------------------------------------------------------


def test_invariant_finalize_flushes_partial_and_writes_manifest(tmp_path):
    """``finalize`` flushes the trailing shard and writes ``shards.json``."""
    w = ShardWriter(tmp_path, shard_size=10)
    w.write_many([{"i": k} for k in range(3)])
    manifest = w.finalize()
    assert manifest["fmt"] == "jsonl"
    assert manifest["total_rows"] == 3
    assert len(manifest["shards"]) == 1
    shard = manifest["shards"][0]
    assert shard == {
        "index": 0,
        "path": "shard-00000.jsonl",
        "rows": 3,
        "complete": True,
    }
    on_disk = json.loads((tmp_path / "shards.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_invariant_finalize_empty_writer_has_zero_shards(tmp_path):
    """``finalize`` with no rows yields an empty shard list, total_rows 0."""
    w = ShardWriter(tmp_path)
    manifest = w.finalize()
    assert manifest["shards"] == []
    assert manifest["total_rows"] == 0


def test_invariant_finalize_does_not_double_flush_aligned_rows(tmp_path):
    """When rows align to ``shard_size``, finalize adds no extra empty shard."""
    w = ShardWriter(tmp_path, shard_size=2)
    w.write_many([{"i": 0}, {"i": 1}])  # exactly one full shard flushed
    manifest = w.finalize()
    assert len(manifest["shards"]) == 1
    assert manifest["total_rows"] == 2


# ---------------------------------------------------------------------------
# _flush — jsonl vs parquet
# ---------------------------------------------------------------------------


def test_invariant_flush_writes_jsonl_compact_lines(tmp_path):
    """JSONL shards are compact (no spaces) one-object-per-line."""
    w = ShardWriter(tmp_path, shard_size=10, fmt="jsonl")
    w.write({"a": 1, "b": [1, 2]})
    w.finalize()
    text = (tmp_path / "shard-00000.jsonl").read_text(encoding="utf-8")
    assert text == '{"a":1,"b":[1,2]}\n'


def test_invariant_flush_writes_parquet_when_requested(tmp_path):
    """``fmt="parquet"`` writes a ``.parquet`` shard (line 110-112)."""
    pytest.importorskip("pyarrow")
    w = ShardWriter(tmp_path, shard_size=10, fmt="parquet")
    w.write_many([{"x": 1}, {"x": 2}])
    manifest = w.finalize()
    assert manifest["fmt"] == "parquet"
    path = tmp_path / "shard-00000.parquet"
    assert path.exists()
    assert manifest["shards"][0]["path"] == "shard-00000.parquet"


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------


def test_invariant_read_manifest_none_when_missing(tmp_path):
    """``read_manifest`` returns None when ``shards.json`` is absent (line 132-133)."""
    assert read_manifest(tmp_path) is None


def test_invariant_read_manifest_none_on_invalid_json(tmp_path):
    """Corrupt ``shards.json`` yields None, not an exception (line 136-137)."""
    (tmp_path / "shards.json").write_text("{not valid json", encoding="utf-8")
    assert read_manifest(tmp_path) is None


def test_invariant_read_manifest_returns_parsed_dict(tmp_path):
    """A valid manifest is parsed and returned verbatim (line 135)."""
    payload = {"fmt": "jsonl", "shards": [], "total_rows": 0}
    (tmp_path / "shards.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_manifest(tmp_path) == payload


# ---------------------------------------------------------------------------
# iter_rows
# ---------------------------------------------------------------------------


def test_invariant_iter_rows_roundtrips_jsonl_via_manifest(tmp_path):
    """``iter_rows`` replays JSONL shards in manifest order (line 148-155)."""
    w = ShardWriter(tmp_path, shard_size=2)
    rows = [{"i": k} for k in range(5)]
    w.write_many(rows)
    w.finalize()
    assert list(iter_rows(tmp_path)) == rows


def test_invariant_iter_rows_roundtrips_parquet_via_manifest(tmp_path):
    """``iter_rows`` reads Parquet shards back through pyarrow (line 151-153)."""
    pytest.importorskip("pyarrow")
    w = ShardWriter(tmp_path, shard_size=2, fmt="parquet")
    rows = [{"i": k} for k in range(3)]
    w.write_many(rows)
    w.finalize()
    assert list(iter_rows(tmp_path)) == rows


def test_invariant_iter_rows_scans_loose_jsonl_without_manifest(tmp_path):
    """With no manifest, loose ``shard-*.jsonl`` are scanned lexicographically
    (line 143-147)."""
    # Write two shards by hand, intentionally out of creation order.
    (tmp_path / "shard-00001.jsonl").write_text('{"i":2}\n{"i":3}\n', encoding="utf-8")
    (tmp_path / "shard-00000.jsonl").write_text('{"i":0}\n{"i":1}\n', encoding="utf-8")
    assert read_manifest(tmp_path) is None
    assert list(iter_rows(tmp_path)) == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]


def test_invariant_iter_rows_skips_blank_lines(tmp_path):
    """``_iter_jsonl`` ignores blank / whitespace-only lines (line 161-163)."""
    (tmp_path / "shard-00000.jsonl").write_text(
        '{"i":0}\n\n   \n{"i":1}\n', encoding="utf-8"
    )
    assert list(iter_rows(tmp_path)) == [{"i": 0}, {"i": 1}]


def test_invariant_iter_rows_defaults_fmt_to_jsonl_when_absent(tmp_path):
    """A manifest lacking ``fmt`` falls back to jsonl reading (line 148 default)."""
    (tmp_path / "shard-00000.jsonl").write_text('{"i":0}\n', encoding="utf-8")
    manifest = {
        "shards": [{"index": 0, "path": "shard-00000.jsonl", "rows": 1}],
        "total_rows": 1,
    }
    (tmp_path / "shards.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert list(iter_rows(tmp_path)) == [{"i": 0}]


# ---------------------------------------------------------------------------
# shard_state / count_rows
# ---------------------------------------------------------------------------


def test_invariant_shard_state_lists_completed_shards(tmp_path):
    """``shard_state`` returns the manifest's shard list (line 168-169)."""
    w = ShardWriter(tmp_path, shard_size=2)
    w.write_many([{"i": k} for k in range(3)])
    w.finalize()
    state = shard_state(tmp_path)
    assert [s["index"] for s in state] == [0, 1]
    assert all(s["complete"] for s in state)


def test_invariant_shard_state_empty_without_manifest(tmp_path):
    """No manifest → empty shard-state list (line 169 ``else``)."""
    assert shard_state(tmp_path) == []


def test_invariant_count_rows_reads_total_from_manifest(tmp_path):
    """``count_rows`` returns ``total_rows`` from the manifest (line 174)."""
    w = ShardWriter(tmp_path, shard_size=2)
    w.write_many([{"i": k} for k in range(5)])
    w.finalize()
    assert count_rows(tmp_path) == 5


def test_invariant_count_rows_zero_without_manifest(tmp_path):
    """No manifest → ``count_rows`` is 0 (line 174 ``else``)."""
    assert count_rows(tmp_path) == 0


def test_pin_current_behavior_count_rows_missing_total_defaults_zero(tmp_path):
    """Pin: a manifest without ``total_rows`` makes ``count_rows`` return 0
    via ``.get(..., 0)`` rather than raising."""
    (tmp_path / "shards.json").write_text(
        json.dumps({"fmt": "jsonl", "shards": []}), encoding="utf-8"
    )
    assert count_rows(tmp_path) == 0


# ---------------------------------------------------------------------------
# module flag sanity
# ---------------------------------------------------------------------------


def test_invariant_has_parquet_flag_is_true_in_this_env():
    """Guard against silent jsonl-fallback: pyarrow is a base dep here."""
    pytest.importorskip("pyarrow")
    assert _shards._HAS_PARQUET is True
