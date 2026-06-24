"""Adversarial tests for ``lighttrain.builtin_plugins.artifacts.store``.

Three backends — ``safetensors-shards``, ``memmap-fixed``, ``parquet-rows`` —
each parametrized in the common tests and given backend-specific tests for
their unique semantics:

  * ``safetensors-shards`` — shard-boundary pin (≥ ``shard_size`` triggers flush),
    pending-dict round-trip before flush
  * ``memmap-fixed`` — shape-mismatch rejection
  * ``parquet-rows`` — row-append/order semantics

Common tests parametrized across all backends:
  * round-trip tensor values (``assert_close``, not just shape)
  * idempotent ``put`` — first write wins (ART-IDEMPOTENT)
  * ``MANIFEST_COMPLETE.json`` is the LAST file replaced (ART-MANIFEST order)
  * ``MANIFEST_COMPLETE.json`` lands via ``.tmp + os.replace`` (ART-MANIFEST atomicity)
  * ``put`` after finalize raises
  * ``open_artifact_store`` rejects incomplete dirs
  * ``open_artifact_store`` round-trip preserves values
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from lighttrain.builtin_plugins.artifacts import (
    ArtifactHeader,
    ArtifactIncompleteError,
    MemmapFixedStore,
    ParquetRowStore,
    SafetensorsShardStore,
    StaleArtifactError,
    open_artifact_store,
)

# --------------------------------------------------------------------------- #
# Backend factory                                                             #
# --------------------------------------------------------------------------- #


_BACKENDS = ["safetensors-shards", "memmap-fixed", "parquet-rows"]

# Backends whose ``__init__`` reloads state from disk and thus support
# ``open_artifact_store`` returning a populated store. All three backends now
# reload, so this mirrors ``_BACKENDS``.
_BACKENDS_WITH_RELOAD = ["safetensors-shards", "memmap-fixed", "parquet-rows"]


def _make_store(backend: str, root: Path, *, shard_size: int = 1000):
    """Construct a fresh store of the requested backend at ``root``."""
    header = ArtifactHeader(
        producer_signature="test", dtype="torch.float32",
        field_schema={"logits": "f32"},
    )
    if backend == "safetensors-shards":
        return SafetensorsShardStore(root, shard_size=shard_size, header=header)
    if backend == "memmap-fixed":
        return MemmapFixedStore(root, header=header)
    if backend == "parquet-rows":
        return ParquetRowStore(root, header=header)
    raise ValueError(backend)


# --------------------------------------------------------------------------- #
# Common: value round-trip                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_put_get_value_roundtrip(backend: str, tmp_path: Path) -> None:
    """Tensors round-trip with identical values, not just identical shapes.

    Input: 3 samples × 2 tensors. Use ``assert_close`` per tensor with
    ``atol=1e-5, rtol=1e-4``. Catches bugs where a backend writes garbage
    bytes / wrong dtype / wrong shape but keeps the key set intact.

    Uses the same store instance for put and get (in-session round-trip)
    so it works for parquet-rows too. The post-reopen variant lives in
    ``test_open_artifact_store_round_trip_via_reload``.
    """
    torch.manual_seed(0)
    samples = [
        ("s1", {"logits": torch.randn(3, 4), "vals": torch.tensor([1.0, 2.0])}),
        ("s2", {"logits": torch.randn(3, 4), "vals": torch.tensor([3.0, 4.0])}),
        ("s3", {"logits": torch.randn(3, 4), "vals": torch.tensor([5.0, 6.0])}),
    ]
    store = _make_store(backend, tmp_path / "store")
    for sid, tensors in samples:
        store.put(sid, tensors)
    store.finalize()

    for sid, tensors in samples:
        loaded = store.get(sid)
        for k, v in tensors.items():
            torch.testing.assert_close(
                loaded[k], v, atol=1e-5, rtol=1e-4,
                msg=f"backend={backend} sample={sid} field={k}",
            )


@pytest.mark.parametrize("backend", _BACKENDS_WITH_RELOAD)
def test_open_artifact_store_round_trip_via_reload(
    backend: str, tmp_path: Path
) -> None:
    """``open_artifact_store`` on a finalized store returns identical tensors.

    Cross-process round-trip: write, finalize, open fresh, read. All backends
    have a reload path (see ``_BACKENDS_WITH_RELOAD``).
    """
    torch.manual_seed(0)
    samples = [
        ("s1", {"logits": torch.randn(3, 4)}),
        ("s2", {"logits": torch.randn(3, 4)}),
    ]
    store = _make_store(backend, tmp_path / "store")
    for sid, tensors in samples:
        store.put(sid, tensors)
    store.finalize()

    re_store = open_artifact_store(tmp_path / "store")
    for sid, tensors in samples:
        loaded = re_store.get(sid)
        torch.testing.assert_close(
            loaded["logits"], tensors["logits"], atol=1e-5, rtol=1e-4,
        )


# --------------------------------------------------------------------------- #
# Common: idempotent put (ART-IDEMPOTENT)                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_invariant_put_idempotent_first_write_wins(
    backend: str, tmp_path: Path
) -> None:
    """``put(sid, T1)`` (made durable) then ``put(sid, T2)`` keeps T1.

    Invariant (resume-safety): a producer that crashes and restarts can
    safely re-call ``put`` for samples it has ALREADY persisted. The
    contract is first-persisted-write wins.

    Backend nuances:
        * safetensors-shards: the ``_index`` guard (store.py:201-202) only
          checks the post-flush index; within a session before the first
          flush, two ``put``s to the same sid overwrite each other in
          ``_pending``. We exercise the documented contract by calling
          ``_flush_shard()`` between the two puts so the second put hits
          the ``if sample_id in self._index: return`` branch.
        * memmap-fixed: ``_index`` is populated synchronously on every put,
          so idempotency works without an explicit flush.
        * parquet-rows: ``_index`` is populated synchronously on put, so
          same-session idempotency works.

    Pin: post-finalize ``get(sid)`` returns T1; the manifest count equals
    the number of unique samples (1), not the number of put() calls (2).
    """
    torch.manual_seed(0)
    t1 = {"logits": torch.tensor([[1.0, 2.0]])}
    t2 = {"logits": torch.tensor([[9.0, 9.0]])}

    store = _make_store(backend, tmp_path / "store")
    store.put("s1", t1)
    if backend == "safetensors-shards":
        # Make the first write durable (populates _index) so the contract's
        # post-flush idempotency engages on the second put.
        store._flush_shard()
    store.put("s1", t2)  # MUST be ignored — first-persisted-write wins
    store.finalize()

    loaded = store.get("s1")
    torch.testing.assert_close(
        loaded["logits"], t1["logits"], atol=1e-5, rtol=1e-4,
    )

    manifest = json.loads(
        (tmp_path / "store" / "MANIFEST_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 1


def test_invariant_put_idempotent_within_session_no_flush_needed(
    tmp_path: Path,
) -> None:
    """Regression pin (closed v0.1.6, L2): ``SafetensorsShardStore.put``
    is idempotent even before the first ``_flush_shard()`` — same-session
    duplicate puts to the same ``sample_id`` are no-ops without needing
    to flush in between.

    Pre-fix behavior: only ``_index`` (post-flush) was checked, so two
    puts to the same sid while both still buffered in ``_pending`` would
    silently overwrite each other, violating the resume-safe / first-
    write-wins contract within a single producer session.

    Pin: ``shard_size=1000`` ensures neither put triggers an auto-flush;
    after ``finalize()`` the persisted value MUST equal the first put's
    tensor, not the second's.
    """
    torch.manual_seed(0)
    t1 = {"logits": torch.tensor([[1.0, 2.0]])}
    t2 = {"logits": torch.tensor([[9.0, 9.0]])}

    store = _make_store("safetensors-shards", tmp_path / "store", shard_size=1000)
    store.put("s1", t1)
    # No flush — both puts stay in ``_pending``; the second must no-op.
    store.put("s1", t2)
    store.finalize()

    loaded = store.get("s1")
    torch.testing.assert_close(
        loaded["logits"], t1["logits"], atol=1e-5, rtol=1e-4,
    )

    manifest = json.loads(
        (tmp_path / "store" / "MANIFEST_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 1


# --------------------------------------------------------------------------- #
# Common: MANIFEST_COMPLETE write order + atomicity                           #
# --------------------------------------------------------------------------- #


def _record_replaces(monkeypatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    real = os.replace

    def _wrap(src, dst, *a, **kw):
        calls.append((str(src), str(dst)))
        return real(src, dst, *a, **kw)

    monkeypatch.setattr("os.replace", _wrap)
    return calls


@pytest.mark.parametrize("backend", _BACKENDS)
def test_invariant_finalize_writes_manifest_complete_last(
    backend: str, tmp_path: Path, monkeypatch
) -> None:
    """``MANIFEST_COMPLETE.json`` is the LAST ``os.replace`` target.

    Invariant: the presence-marker is written after the data and the
    sample-index; partial finalize never produces a ``MANIFEST_COMPLETE.json``
    that points at incomplete data.
    """
    calls = _record_replaces(monkeypatch)
    store = _make_store(backend, tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([[1.0, 2.0]])})
    store.finalize()
    manifest_replaces = [
        i for i, (_s, d) in enumerate(calls) if d.endswith("MANIFEST_COMPLETE.json")
    ]
    assert manifest_replaces, f"backend={backend}: no MANIFEST_COMPLETE replace observed"
    assert manifest_replaces[-1] == len(calls) - 1, (
        f"backend={backend}: MANIFEST_COMPLETE replaced before final-pass "
        f"replaces: {calls[manifest_replaces[-1] + 1 :]}"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_invariant_manifest_complete_atomic_via_tmp_replace(
    backend: str, tmp_path: Path, monkeypatch
) -> None:
    """``MANIFEST_COMPLETE.json`` lands via ``.tmp + os.replace``, not a direct
    open-and-write.

    Invariant: protects against a concurrent reader observing a partially
    written manifest.
    """
    calls = _record_replaces(monkeypatch)
    store = _make_store(backend, tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([[1.0]])})
    store.finalize()
    manifest_call = next(
        (s, d) for s, d in calls if d.endswith("MANIFEST_COMPLETE.json")
    )
    src, _dst = manifest_call
    assert src.endswith(".tmp"), (
        f"backend={backend}: MANIFEST_COMPLETE replaced from non-.tmp source {src}"
    )


# --------------------------------------------------------------------------- #
# Common: finalize then put → error; open requires presence marker            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_put_after_finalize_raises(backend: str, tmp_path: Path) -> None:
    """Every backend guards against post-finalize ``put``.

    Contract: ``put`` raises ``RuntimeError`` so an accidental "one more
    sample" after the manifest landed is caught loudly. All three backends
    carry this ``self._finalized`` guard (parquet/memmap gained it alongside
    the parquet reload path).
    """
    store = _make_store(backend, tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([[1.0]])})
    store.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        store.put("s2", {"logits": torch.tensor([[2.0]])})


@pytest.mark.parametrize("backend", _BACKENDS)
def test_open_artifact_store_rejects_incomplete(
    backend: str, tmp_path: Path
) -> None:
    """A store dir without ``MANIFEST_COMPLETE.json`` cannot be opened.

    Invariant: incomplete artifacts surface as a typed exception rather
    than silently appearing empty.
    """
    store = _make_store(backend, tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([[1.0]])})
    # Deliberately do not call finalize → MANIFEST_COMPLETE.json missing.
    with pytest.raises(ArtifactIncompleteError):
        open_artifact_store(tmp_path / "store")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_iter_keys_after_finalize_count(backend: str, tmp_path: Path) -> None:
    """``iter_keys`` yields one key per unique ``put`` after finalize.

    Pin: the iteration contract matches the put-set; uses in-session
    iteration so it works for parquet-rows too.
    """
    store = _make_store(backend, tmp_path / "store")
    for sid in ("s1", "s2", "s3"):
        store.put(sid, {"logits": torch.tensor([[float(ord(sid[-1]))]])})
    store.finalize()
    assert sorted(store.iter_keys()) == ["s1", "s2", "s3"]


# --------------------------------------------------------------------------- #
# safetensors-shards specific: boundary + pending round-trip                  #
# --------------------------------------------------------------------------- #


def test_pin_shard_boundary_inclusive_flush(tmp_path: Path) -> None:
    """Shard flush triggers when the pending dict size reaches ``shard_size``
    (``>=``, not ``>``).

    Input: shard_size=2; put s1, s2, s3.
    Analytical (runner.py:206-207): ``if len(self._pending) >= self.shard_size:
    self._flush_shard()`` → after s2, pending size is 2 → flush; s1 and s2
    land in shard 0. s3 then lands alone in shard 1.

    If this behavior is intentionally changed (e.g. to strict ``>``),
    update this test AND bump SCHEMA_VERSION (or document the breaking change).
    """
    store = SafetensorsShardStore(tmp_path / "store", shard_size=2)
    for sid in ("s1", "s2", "s3"):
        store.put(sid, {"logits": torch.tensor([[float(ord(sid[-1]))]])})
    store.finalize()

    # Inspect on-disk index AFTER finalize (flush has happened).
    idx = json.loads(
        (tmp_path / "store" / "manifest.json").read_text(encoding="utf-8")
    )["sample_to_shard"]
    assert idx["s1"] == 0
    assert idx["s2"] == 0
    assert idx["s3"] == 1


def test_shard_off_by_one_smaller_no_orphan_shard(tmp_path: Path) -> None:
    """Exactly N samples with shard_size=N produces ONE shard, no orphan shard_1.

    Pin: no ``shard_00001.safetensors`` is created when only one shard's
    worth of samples is written.
    """
    store = SafetensorsShardStore(tmp_path / "store", shard_size=3)
    for sid in ("s1", "s2", "s3"):
        store.put(sid, {"logits": torch.tensor([[float(ord(sid[-1]))]])})
    store.finalize()
    shards = sorted((tmp_path / "store").glob("shard_*.safetensors"))
    assert len(shards) == 1, f"got shards: {[p.name for p in shards]}"
    assert shards[0].name == "shard_00000.safetensors"


def test_pending_get_before_flush(tmp_path: Path) -> None:
    """``get`` returns pending samples before they are flushed to a shard.

    Pin: the pending-dict read path (store.py:247-248) provides a coherent
    view of put-not-yet-flushed tensors.
    """
    torch.manual_seed(0)
    t = torch.randn(2, 3)
    store = SafetensorsShardStore(tmp_path / "store", shard_size=100)
    store.put("s1", {"logits": t})
    # No finalize, no flush → s1 still lives in pending dict.
    out = store.get("s1")
    torch.testing.assert_close(out["logits"], t, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# memmap-fixed specific: shape rejection                                      #
# --------------------------------------------------------------------------- #


def test_memmap_fixed_rejects_mismatched_shape(tmp_path: Path) -> None:
    """A second ``put`` with a different per-sample shape raises ``ValueError``.

    Contract: memmap-fixed is fixed-shape by design; runtime mismatches must
    fail loud, not silently truncate or extend the binary.
    """
    store = MemmapFixedStore(tmp_path / "store")
    store.put("s1", {"logits": torch.tensor([1.0, 2.0, 3.0])})
    with pytest.raises(ValueError, match="identical shapes"):
        store.put("s2", {"logits": torch.tensor([1.0, 2.0])})


# --------------------------------------------------------------------------- #
# parquet-rows specific: append-order semantics                               #
# --------------------------------------------------------------------------- #


def test_parquet_rows_append_order_preserved(tmp_path: Path) -> None:
    """``iter_keys`` preserves the insertion order of ``put`` calls (in-session).

    Pin: parquet-rows stores ``sample_id -> row_idx`` and iterating yields
    keys in their original insertion order.

    The row-order contract on disk is held by the sample_to_row index in
    ``manifest.json`` (which the reload path uses as the index source of truth).
    """
    store = ParquetRowStore(tmp_path / "store")
    for sid in ("alpha", "bravo", "charlie"):
        store.put(sid, {"logits": torch.tensor([[float(ord(sid[0]))]])})
    store.finalize()
    assert list(store.iter_keys()) == ["alpha", "bravo", "charlie"]
    # Sanity: the persisted manifest index agrees with the in-memory order.
    idx = json.loads(
        (tmp_path / "store" / "manifest.json").read_text(encoding="utf-8")
    )["sample_to_row"]
    assert idx == {"alpha": 0, "bravo": 1, "charlie": 2}


def test_parquet_reopen_with_heterogeneous_fields(tmp_path: Path) -> None:
    """Reload must strip the union-schema None columns, not crash or cross fields.

    Two samples carry *different* tensor field sets, so ``finalize`` writes a
    union schema where each row has ``None`` for the columns it lacks. On reopen
    the reload path must drop those None triples; ``get`` then returns exactly
    each sample's own fields.
    """
    root = tmp_path / "store"
    store = ParquetRowStore(root)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0]), "mask": torch.tensor([1])})
    store.put("s2", {"logits": torch.tensor([3.0, 4.0])})  # no "mask"
    store.finalize()

    re_store = open_artifact_store(root)
    g1 = re_store.get("s1")
    g2 = re_store.get("s2")
    assert set(g1.keys()) == {"logits", "mask"}
    assert set(g2.keys()) == {"logits"}  # the union-schema None "mask" was stripped
    torch.testing.assert_close(g1["logits"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(g2["logits"], torch.tensor([3.0, 4.0]))


def test_parquet_reopen_corrupt_partial_triple_fails_loud(tmp_path: Path) -> None:
    """A row whose payload/shape/dtype triple is partially None is fail-loud.

    Hand-build a finalized store whose ``rows.parquet`` has a payload byte
    column but a missing (``None``) shape sidecar — reopening must raise rather
    than silently mis-read or drop the field.
    """
    root = tmp_path / "store"
    store = ParquetRowStore(root)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    store.finalize()

    # Corrupt rows.parquet: null out the logits__shape column for the only row.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(str(root / "rows.parquet"))
    cols = {name: table.column(name) for name in table.column_names}
    cols["logits__shape"] = pa.array([None], type=table.schema.field("logits__shape").type)
    corrupt = pa.table(cols)
    pq.write_table(corrupt, str(root / "rows.parquet"))

    with pytest.raises(ValueError, match="partially-None"):
        open_artifact_store(root)


def test_parquet_reopen_corrupt_manifest_missing_index_fails_loud(tmp_path: Path) -> None:
    """A finalized store whose manifest.json lost ``sample_to_row`` is fail-loud.

    Regression: ``payload.get("sample_to_row", {})`` used to treat a truncated
    manifest (``{}``) as an empty store even though rows.parquet still held data,
    silently returning zero samples. The reload now raises instead.
    """
    root = tmp_path / "store"
    store = ParquetRowStore(root)
    store.put("s1", {"logits": torch.tensor([1.0, 2.0])})
    store.finalize()

    # Drop sample_to_row entirely while rows.parquet / MANIFEST_COMPLETE remain.
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_to_row"):
        open_artifact_store(root)


# --------------------------------------------------------------------------- #
# open_artifact_store: stale-header detection (F5 / DESIGN §25.3)              #
# --------------------------------------------------------------------------- #


def test_open_artifact_store_rejects_stale_header_by_default(tmp_path: Path) -> None:
    """When the caller passes an ``expected_header`` whose ``data_version`` does
    not match the on-disk header, ``open_artifact_store`` raises
    ``StaleArtifactError``.

    Invariant (F5): a producer-version / data-version mismatch is fail-loud by
    default so a training run never silently consumes artifacts produced by an
    incompatible upstream.
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.header.producer_signature = "v1"
    store.header.data_version = "alpha"
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    expected = ArtifactHeader(producer_signature="v1", data_version="beta")
    with pytest.raises(StaleArtifactError):
        open_artifact_store(root, expected_header=expected)


def test_open_artifact_store_allow_stale_bypasses_mismatch(tmp_path: Path) -> None:
    """``allow_stale=True`` opens a header-mismatched store anyway, returning a
    populated store.

    Escape hatch: an operator who knowingly accepts the version drift can
    still read the artifacts.
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.header.producer_signature = "v1"
    store.header.data_version = "alpha"
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    expected = ArtifactHeader(producer_signature="v1", data_version="beta")
    opened = open_artifact_store(root, expected_header=expected, allow_stale=True)
    assert "s" in list(opened.iter_keys())


def test_open_artifact_store_no_expected_header_skips_stale_check(
    tmp_path: Path,
) -> None:
    """With no ``expected_header`` argument, no staleness check runs and the
    store opens normally.

    Pin: the stale-check is opt-in; the common path (open without an
    expectation) must not regress into a spurious raise.
    """
    root = tmp_path / "store"
    store = SafetensorsShardStore(root)
    store.put("s", {"x": torch.zeros(2)})
    store.finalize()

    opened = open_artifact_store(root)
    assert opened.contains("s")
