"""Tests for ``lighttrain.data.prepgraph._io``.

Coverage targets (uncovered lines as of the baseline):
  * 45, 48   -- fsync_file OSError-swallowing branch
  * 72, 73   -- read_manifest JSONDecodeError → returns None
  * 83–92    -- write_shard_state (happy path + tmp→replace)
  * 96–105   -- read_shard_state (missing file / valid list / bad JSON / non-list)
  * 121      -- commit raises when staging is incomplete
  * 136–140  -- cleanup_staging removes orphaned dirs without a manifest

General edge cases also covered:
  * staging_dir and final_dir path construction
  * ensure_dir idempotency
  * write_manifest → read_manifest round-trip
  * is_complete on missing / corrupt / valid manifest
  * commit happy-path: staging promoted to final, old final rmtree'd first
  * cleanup_staging skips non-directories and dirs WITH a manifest
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lighttrain.data.prepgraph._io import (
    MANIFEST_NAME,
    SHARD_COMPLETE_NAME,
    cleanup_staging,
    commit,
    ensure_dir,
    final_dir,
    fsync_file,
    is_complete,
    read_manifest,
    read_shard_state,
    staging_dir,
    write_manifest,
    write_shard_state,
)

# ---------------------------------------------------------------------------
# staging_dir / final_dir path construction
# ---------------------------------------------------------------------------


def test_invariant_staging_dir_path(tmp_path: Path) -> None:
    """``staging_dir`` returns ``<store_root>/tmp/<fp>``."""
    p = staging_dir(tmp_path, "abc123")
    assert p == tmp_path / "tmp" / "abc123"


def test_invariant_final_dir_path(tmp_path: Path) -> None:
    """``final_dir`` returns ``<store_root>/<kind>/<name>/<fp>``."""
    p = final_dir(tmp_path, "tokenize", "tok", "def456")
    assert p == tmp_path / "tokenize" / "tok" / "def456"


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------


def test_invariant_ensure_dir_creates_nested_dirs(tmp_path: Path) -> None:
    """``ensure_dir`` creates deeply nested directories and returns the path."""
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.is_dir()
    assert result == target


def test_invariant_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    """Calling ``ensure_dir`` twice on the same dir does not raise."""
    target = tmp_path / "x"
    ensure_dir(target)
    ensure_dir(target)  # must not raise
    assert target.is_dir()


# ---------------------------------------------------------------------------
# fsync_file — OSError swallowing (lines 45, 48)
# ---------------------------------------------------------------------------


def test_invariant_fsync_file_succeeds_on_regular_file(tmp_path: Path) -> None:
    """``fsync_file`` completes without error on a normal file."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    fsync_file(f)  # must not raise


def test_invariant_fsync_file_swallows_oserror(tmp_path: Path) -> None:
    """``fsync_file`` swallows ``OSError`` raised by ``os.fsync`` (lines 45, 48).

    Simulates a filesystem that does not support fsync (e.g. Windows tmp).
    """
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")

    with patch("os.fsync", side_effect=OSError("fsync not supported")):
        fsync_file(f)  # must not raise


def test_invariant_fsync_file_swallows_oserror_from_open(tmp_path: Path) -> None:
    """``fsync_file`` swallows ``OSError`` raised by ``os.open`` itself (line 45)."""
    with patch("os.open", side_effect=OSError("no such file")):
        fsync_file(tmp_path / "nonexistent.bin")  # must not raise


# ---------------------------------------------------------------------------
# write_manifest / read_manifest
# ---------------------------------------------------------------------------


def test_invariant_write_manifest_creates_manifest_file(tmp_path: Path) -> None:
    """``write_manifest`` creates ``MANIFEST_COMPLETE.json`` in the target dir."""
    path = write_manifest(tmp_path, {"rows": 10, "schema": "v1"})
    assert path.name == MANIFEST_NAME
    assert path.exists()


def test_invariant_write_manifest_round_trips_payload(tmp_path: Path) -> None:
    """Payload written by ``write_manifest`` is recovered by ``read_manifest``."""
    payload = {"rows": 42, "schema_version": "v2", "tag": "test"}
    write_manifest(tmp_path, payload)
    result = read_manifest(tmp_path)
    assert result == payload


def test_invariant_write_manifest_sorts_keys(tmp_path: Path) -> None:
    """``write_manifest`` serialises with ``sort_keys=True``."""
    write_manifest(tmp_path, {"z": 1, "a": 2})
    raw = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert list(data.keys()) == sorted(data.keys())


def test_invariant_read_manifest_returns_none_when_missing(tmp_path: Path) -> None:
    """``read_manifest`` returns ``None`` when no manifest file exists."""
    assert read_manifest(tmp_path) is None


def test_invariant_read_manifest_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    """``read_manifest`` returns ``None`` when the JSON is malformed (lines 72–73)."""
    (tmp_path / MANIFEST_NAME).write_text("{ this is not json }", encoding="utf-8")
    assert read_manifest(tmp_path) is None


def test_invariant_read_manifest_returns_none_on_truncated_file(tmp_path: Path) -> None:
    """``read_manifest`` handles a truncated / empty file gracefully."""
    (tmp_path / MANIFEST_NAME).write_text("", encoding="utf-8")
    assert read_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


def test_invariant_is_complete_false_when_no_manifest(tmp_path: Path) -> None:
    """``is_complete`` returns ``False`` when no manifest is present."""
    assert is_complete(tmp_path) is False


def test_invariant_is_complete_false_when_manifest_is_corrupt(tmp_path: Path) -> None:
    """``is_complete`` returns ``False`` when manifest exists but is corrupt."""
    (tmp_path / MANIFEST_NAME).write_text("bad json", encoding="utf-8")
    assert is_complete(tmp_path) is False


def test_invariant_is_complete_true_when_manifest_valid(tmp_path: Path) -> None:
    """``is_complete`` returns ``True`` when a valid manifest is present."""
    write_manifest(tmp_path, {"rows": 5})
    assert is_complete(tmp_path) is True


# ---------------------------------------------------------------------------
# write_shard_state (lines 83–92)
# ---------------------------------------------------------------------------


def test_invariant_write_shard_state_creates_complete_json(tmp_path: Path) -> None:
    """``write_shard_state`` writes ``complete.json`` into the target dir."""
    shards = [{"shard": 0, "rows": 100}, {"shard": 1, "rows": 200}]
    path = write_shard_state(tmp_path, shards)
    assert path.name == SHARD_COMPLETE_NAME
    assert path.exists()


def test_invariant_write_shard_state_round_trips_data(tmp_path: Path) -> None:
    """Data written by ``write_shard_state`` is recovered by ``read_shard_state``."""
    shards = [{"shard": 0, "rows": 50}, {"shard": 1, "rows": 75}]
    write_shard_state(tmp_path, shards)
    result = read_shard_state(tmp_path)
    assert result == shards


def test_invariant_write_shard_state_creates_parent_dirs(tmp_path: Path) -> None:
    """``write_shard_state`` creates missing parent directories."""
    target = tmp_path / "nested" / "sub"
    write_shard_state(target, [{"shard": 0}])
    assert (target / SHARD_COMPLETE_NAME).exists()


def test_invariant_write_shard_state_accepts_iterable(tmp_path: Path) -> None:
    """``write_shard_state`` accepts any ``Iterable[Mapping]``, not just lists."""
    def _gen():
        yield {"shard": 0}
        yield {"shard": 1}

    path = write_shard_state(tmp_path, _gen())
    assert path.exists()
    result = read_shard_state(tmp_path)
    assert len(result) == 2


def test_invariant_write_shard_state_uses_atomic_replace(tmp_path: Path) -> None:
    """``write_shard_state`` must not leave a ``.json.tmp`` file behind."""
    write_shard_state(tmp_path, [{"shard": 0}])
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"stale tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# read_shard_state (lines 96–105)
# ---------------------------------------------------------------------------


def test_invariant_read_shard_state_returns_empty_list_when_missing(tmp_path: Path) -> None:
    """``read_shard_state`` returns ``[]`` when no ``complete.json`` exists (line 97–98)."""
    assert read_shard_state(tmp_path) == []


def test_invariant_read_shard_state_parses_valid_list(tmp_path: Path) -> None:
    """``read_shard_state`` returns the list of dicts when the file is valid (lines 100–102)."""
    shards = [{"shard": 0, "ok": True}, {"shard": 1, "ok": False}]
    (tmp_path / SHARD_COMPLETE_NAME).write_text(
        json.dumps(shards, indent=2), encoding="utf-8"
    )
    result = read_shard_state(tmp_path)
    assert result == shards


def test_invariant_read_shard_state_returns_empty_on_bad_json(tmp_path: Path) -> None:
    """``read_shard_state`` returns ``[]`` on malformed JSON (lines 103–105)."""
    (tmp_path / SHARD_COMPLETE_NAME).write_text("{ not json ]", encoding="utf-8")
    assert read_shard_state(tmp_path) == []


def test_invariant_read_shard_state_returns_empty_when_not_a_list(tmp_path: Path) -> None:
    """``read_shard_state`` returns ``[]`` when the file contains a non-list (line 101 else)."""
    (tmp_path / SHARD_COMPLETE_NAME).write_text(
        json.dumps({"shard": 0}), encoding="utf-8"  # a dict, not a list
    )
    assert read_shard_state(tmp_path) == []


def test_pin_current_behavior_read_shard_state_converts_each_item_to_dict(tmp_path: Path) -> None:
    """Pin: each item is passed through ``dict(x)`` even if it was already a dict.

    This is the ``[dict(x) for x in data]`` expression at line 102.
    Ensures sub-items with extra Mapping types are normalised to plain dicts.
    """
    shards = [{"a": 1}, {"b": 2}]
    (tmp_path / SHARD_COMPLETE_NAME).write_text(json.dumps(shards), encoding="utf-8")
    result = read_shard_state(tmp_path)
    assert all(type(x) is dict for x in result)


# ---------------------------------------------------------------------------
# commit (lines 108–126)
# ---------------------------------------------------------------------------


def test_invariant_commit_raises_when_staging_incomplete(tmp_path: Path) -> None:
    """``commit`` raises ``RuntimeError`` when staging lacks a manifest (line 121)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    final = tmp_path / "final"
    with pytest.raises(RuntimeError, match="incomplete staging"):
        commit(staging, final)


def test_invariant_commit_promotes_staging_to_final(tmp_path: Path) -> None:
    """``commit`` atomically moves staging to final when staging is complete."""
    staging = tmp_path / "staging"
    staging.mkdir()
    write_manifest(staging, {"rows": 3})
    (staging / "shard_0.bin").write_bytes(b"\x00" * 16)

    final = tmp_path / "final"
    commit(staging, final)

    assert final.is_dir()
    assert (final / MANIFEST_NAME).exists()
    assert not staging.exists()


def test_invariant_commit_removes_existing_final_before_replace(tmp_path: Path) -> None:
    """``commit`` removes a pre-existing final dir before the atomic rename (line 124–125)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    write_manifest(staging, {"rows": 5})

    final = tmp_path / "final"
    final.mkdir()
    # Put a stale file in final to prove it gets cleaned
    (final / "stale.txt").write_text("old", encoding="utf-8")

    commit(staging, final)

    assert final.is_dir()
    assert is_complete(final)
    assert not (final / "stale.txt").exists()


def test_invariant_commit_creates_parent_dirs_of_final(tmp_path: Path) -> None:
    """``commit`` creates the parent of ``final`` if it does not exist."""
    staging = tmp_path / "staging"
    staging.mkdir()
    write_manifest(staging, {"rows": 1})

    final = tmp_path / "a" / "b" / "c"
    commit(staging, final)
    assert is_complete(final)


# ---------------------------------------------------------------------------
# cleanup_staging (lines 129–141)
# ---------------------------------------------------------------------------


def test_invariant_cleanup_staging_returns_zero_when_tmp_missing(tmp_path: Path) -> None:
    """``cleanup_staging`` returns 0 when the tmp dir does not exist."""
    assert cleanup_staging(tmp_path) == 0


def test_invariant_cleanup_staging_removes_incomplete_staging_dirs(tmp_path: Path) -> None:
    """``cleanup_staging`` removes dirs without a manifest (lines 136–140)."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    # Two orphaned staging dirs (no manifest)
    for name in ("fp_aaa", "fp_bbb"):
        d = tmp_dir / name
        d.mkdir()
        (d / "partial.bin").write_bytes(b"\xde\xad")

    count = cleanup_staging(tmp_path)
    assert count == 2
    assert not (tmp_dir / "fp_aaa").exists()
    assert not (tmp_dir / "fp_bbb").exists()


def test_invariant_cleanup_staging_keeps_complete_staging_dirs(tmp_path: Path) -> None:
    """``cleanup_staging`` does NOT remove dirs that have a valid manifest."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    good = tmp_dir / "fp_good"
    good.mkdir()
    write_manifest(good, {"rows": 10})

    count = cleanup_staging(tmp_path)
    assert count == 0
    assert good.is_dir()


def test_invariant_cleanup_staging_skips_non_directories(tmp_path: Path) -> None:
    """``cleanup_staging`` ignores plain files inside the tmp dir (lines 136–137)."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    # A plain file should not be counted or removed
    (tmp_dir / "stray.txt").write_text("oops", encoding="utf-8")

    count = cleanup_staging(tmp_path)
    assert count == 0
    assert (tmp_dir / "stray.txt").exists()


def test_invariant_cleanup_staging_mixed_dirs(tmp_path: Path) -> None:
    """``cleanup_staging`` removes incomplete dirs and keeps complete ones."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    # Complete dir — must be kept
    complete = tmp_dir / "fp_complete"
    complete.mkdir()
    write_manifest(complete, {"rows": 7})

    # Incomplete dirs — must be removed
    for name in ("fp_crash1", "fp_crash2"):
        d = tmp_dir / name
        d.mkdir()

    count = cleanup_staging(tmp_path)
    assert count == 2
    assert complete.is_dir()
    assert not (tmp_dir / "fp_crash1").exists()
    assert not (tmp_dir / "fp_crash2").exists()


@pytest.mark.parametrize(
    "fp,kind,name,expected_suffix",
    [
        ("abc123", "tokenize", "tok", "tokenize/tok/abc123"),
        ("xyz", "load", "raw", "load/raw/xyz"),
        ("fp-hyphen", "validate", "v", "validate/v/fp-hyphen"),
    ],
)
def test_invariant_final_dir_parametrized(tmp_path, fp, kind, name, expected_suffix) -> None:
    """``final_dir`` builds the right path for various (kind, name, fp) triples."""
    result = final_dir(tmp_path, kind, name, fp)
    assert result == tmp_path / expected_suffix
