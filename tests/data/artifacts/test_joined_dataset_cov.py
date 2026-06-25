"""Coverage-drive tests for ``lighttrain.builtin_plugins.data.artifacts.joined_dataset``.

Targets the lines not reached by the existing ``test_joined_dataset.py``:

* 94  — ``ValueError`` when a join entry has no ``store``/``path`` key
* 98  — ``ArtifactHeader.from_dict`` path when ``expected_header`` is a Mapping
* 106 — ``reload()`` re-opens stores
* 117 — ``on_artifact_new_version()`` delegates to ``_open_stores``
* 120-122 — ``__len__``: present path + absent path (``TypeError``)
* 127 — ``__getitem__`` raises ``TypeError`` for non-Mapping base sample
* 153-156 — ``__iter__`` skips ``None`` rows produced by ``missing='drop'``
* 161-165 — ``_resolve_base`` with a Mapping spec (registry name, with/without tokenizer)
* 178-188 — ``_parse_shape``: list branch, int branch, unparseable → fallback ``()``
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.data.artifacts import (
    ArtifactHeader,
    ArtifactJoinedDataset,
    SafetensorsShardStore,
)
from lighttrain.builtin_plugins.data.artifacts.joined_dataset import (
    _default_namespace,
    _parse_shape,
    _resolve_base,
)
from lighttrain.registry import register

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_store(
    root,
    *,
    samples: dict,
    field_schema: dict | None = None,
    producer_signature: str = "",
) -> None:
    """Write and finalize a SafetensorsShardStore under ``root``."""
    header = ArtifactHeader(
        producer_signature=producer_signature,
        field_schema=field_schema or {},
    )
    store = SafetensorsShardStore(root, header=header)
    for sid, tensors in samples.items():
        store.put(sid, tensors)
    store.finalize()


class _MapDS:
    """Minimal map-style dataset wrapping a list of dicts."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class _NoLenDS:
    """Dataset with ``__getitem__`` but deliberately no ``__len__``."""

    def __getitem__(self, idx: int) -> dict:
        return {"id": str(idx)}


class _NonMappingDS:
    """Dataset that returns a string instead of a Mapping."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> str:  # wrong return type on purpose
        return "this-is-not-a-dict"


# ---------------------------------------------------------------------------
# _open_stores: missing store key → ValueError  (line 94)
# ---------------------------------------------------------------------------


def test_invariant_missing_store_key_raises_value_error() -> None:
    """A join entry with neither ``store`` nor ``path`` raises ``ValueError``
    at construction time (line 94)."""
    with pytest.raises(ValueError, match="each join entry needs"):
        ArtifactJoinedDataset(
            _MapDS([]),
            join=[{"namespace": "feat"}],  # no 'store' or 'path'
        )


# ---------------------------------------------------------------------------
# _open_stores: expected_header as Mapping → ArtifactHeader.from_dict (line 98)
# ---------------------------------------------------------------------------


def test_expected_header_mapping_is_converted_to_artifact_header(
    tmp_path,
) -> None:
    """When ``expected_header`` is a plain ``dict`` it is converted via
    ``ArtifactHeader.from_dict`` before being passed to ``open_artifact_store``
    (line 98).  Matching producer_signature should not raise."""
    root = tmp_path / "art"
    _make_store(
        root,
        samples={"s1": {"v": torch.zeros(2)}},
        producer_signature="my_producer",
    )
    ds = ArtifactJoinedDataset(
        _MapDS([{"id": "s1"}]),
        join=[
            {
                "store": str(root),
                "namespace": "ns",
                "expected_header": {"producer_signature": "my_producer"},
                "missing": "require",
            }
        ],
    )
    row = ds[0]
    assert row is not None
    assert "aux.ns.v" in row


# ---------------------------------------------------------------------------
# reload() re-opens stores (line 106)
# ---------------------------------------------------------------------------


def test_invariant_reload_reopens_stores(tmp_path) -> None:
    """``reload()`` clears and re-builds ``_stores`` from the on-disk specs;
    the count stays the same and subsequent reads still succeed (line 106)."""
    root = tmp_path / "art"
    _make_store(root, samples={"s1": {"v": torch.zeros(2)}})
    ds = ArtifactJoinedDataset(
        _MapDS([{"id": "s1"}]),
        join=[{"store": str(root), "namespace": "ns", "missing": "require"}],
    )
    stores_before = len(ds._stores)
    ds.reload()
    stores_after = len(ds._stores)
    assert stores_before == stores_after == 1
    # Data still readable after reload
    row = ds[0]
    assert row is not None
    assert "aux.ns.v" in row


# ---------------------------------------------------------------------------
# on_artifact_new_version() delegates to _open_stores (line 117)
# ---------------------------------------------------------------------------


def test_on_artifact_new_version_calls_open_stores(tmp_path) -> None:
    """``on_artifact_new_version`` is an EventBus callback that re-opens stores.
    After calling it, the store list has the same length and data is still
    accessible (line 117)."""
    root = tmp_path / "art"
    _make_store(root, samples={"s1": {"v": torch.zeros(2)}})
    ds = ArtifactJoinedDataset(
        _MapDS([{"id": "s1"}]),
        join=[{"store": str(root), "namespace": "ns", "missing": "require"}],
    )
    # Dispatch with arbitrary kwargs — the method accepts **_
    ds.on_artifact_new_version(path="/new/path", step=42)
    assert len(ds._stores) == 1
    row = ds[0]
    assert row is not None


# ---------------------------------------------------------------------------
# __len__: both branches (lines 120-122)
# ---------------------------------------------------------------------------


def test_invariant_len_delegates_to_base(tmp_path) -> None:
    """``len(ds)`` returns the base dataset's length when base has ``__len__``
    (line 121)."""
    ds = ArtifactJoinedDataset(_MapDS([{"id": "a"}, {"id": "b"}]), join=[])
    assert len(ds) == 2


def test_invariant_len_raises_when_base_has_no_len() -> None:
    """``len(ds)`` raises ``TypeError`` when the base dataset lacks ``__len__``
    (line 122)."""
    ds = ArtifactJoinedDataset(_NoLenDS(), join=[])
    with pytest.raises(TypeError, match="no __len__"):
        len(ds)


# ---------------------------------------------------------------------------
# __getitem__: non-Mapping sample raises TypeError (line 127)
# ---------------------------------------------------------------------------


def test_invariant_getitem_non_mapping_raises_type_error() -> None:
    """When the base dataset returns something that is not a ``Mapping``,
    ``__getitem__`` raises ``TypeError`` naming the offending type (line 127)."""
    ds = ArtifactJoinedDataset(_NonMappingDS(), join=[])
    with pytest.raises(TypeError, match="str.*want Mapping"):
        ds[0]


# ---------------------------------------------------------------------------
# __iter__: skips None rows produced by missing='drop' (lines 153-156)
# ---------------------------------------------------------------------------


def test_invariant_iter_skips_dropped_samples(tmp_path) -> None:
    """``__iter__`` filters out ``None`` values produced when
    ``missing='drop'``: only samples present in the store appear in the
    iterated sequence (lines 153-156)."""
    root = tmp_path / "art"
    _make_store(root, samples={"s1": {"v": torch.zeros(2)}})
    # s2 is not in the store → will produce None under 'drop'
    base = _MapDS([{"id": "s1"}, {"id": "s2"}, {"id": "s1"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "ns", "missing": "drop"}],
    )
    rows = list(ds)
    assert len(rows) == 2
    assert all(r["id"] == "s1" for r in rows)


def test_iter_empty_when_all_dropped(tmp_path) -> None:
    """If every sample is missing and policy is 'drop', iteration yields
    nothing (no error)."""
    root = tmp_path / "art"
    _make_store(root, samples={"present": {"v": torch.zeros(1)}})
    base = _MapDS([{"id": "gone"}])
    ds = ArtifactJoinedDataset(
        base,
        join=[{"store": str(root), "namespace": "ns", "missing": "drop"}],
    )
    assert list(ds) == []


# ---------------------------------------------------------------------------
# _resolve_base: Mapping spec (lines 161-165)
# ---------------------------------------------------------------------------

# Register a tiny test-only dataset for resolver tests.  Using a unique name
# avoids collisions with production registry entries.
class _RegDS:
    """Tiny dataset registered for resolver-path coverage tests."""

    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict:  # pragma: no cover
        raise IndexError


register("dataset", "_cov_joined_ds_test")(_RegDS)


def test_resolve_base_mapping_without_tokenizer() -> None:
    """When ``base`` is a ``Mapping``, ``_resolve_base`` calls the registry
    resolver and returns a constructed dataset (lines 160-165).
    Without a tokenizer the spec is passed as-is."""
    result = _resolve_base({"name": "_cov_joined_ds_test"}, tokenizer=None)
    assert isinstance(result, _RegDS)
    assert result.tokenizer is None


def test_resolve_base_mapping_with_tokenizer() -> None:
    """When ``tokenizer`` is not ``None``, it is injected via ``setdefault``
    into the resolver spec so the constructed dataset receives it (line 164)."""
    result = _resolve_base({"name": "_cov_joined_ds_test"}, tokenizer="MY_TOK")
    assert isinstance(result, _RegDS)
    assert result.tokenizer == "MY_TOK"


def test_resolve_base_non_mapping_passthrough() -> None:
    """Non-Mapping ``base`` is returned unchanged (line 166)."""
    raw = _MapDS([{"id": "x"}])
    assert _resolve_base(raw) is raw


# ---------------------------------------------------------------------------
# _parse_shape (lines 173-188)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_str, expected",
    [
        ("(3,)", (3,)),         # tuple branch (line 177)
        ("(2, 4)", (2, 4)),     # multi-dim tuple
        ("[3, 4]", (3, 4)),     # list branch (line 179)
        ("[3.0, 4.0]", (3, 4)), # list with floats → int() cast
        ("5", (5,)),            # int branch (line 181)
        ("()", ()),             # empty tuple → handled by tuple branch
        ("[]", ()),             # empty list → tuple(int(x) for x in []) = ()
    ],
)
def test_invariant_parse_shape_valid(shape_str: str, expected: tuple) -> None:
    """``_parse_shape`` converts valid shape strings to int tuples."""
    assert _parse_shape(shape_str) == expected


def test_pin_current_behavior_parse_shape_invalid_falls_back_to_empty() -> None:
    """``_parse_shape`` with an unparseable string logs a warning and returns
    ``()`` rather than raising (lines 182-188).

    [PIN] This pins the current silent-fallback contract. If the design changes
    to raise on bad schema strings, this test should be updated.
    """
    assert _parse_shape("not_a_shape") == ()


def test_pin_current_behavior_parse_shape_nested_list_falls_back(
    caplog,
) -> None:
    """A nested list (e.g. ``'[[1, 2]]'``) triggers the exception path inside
    ``_parse_shape`` because ``int([[1, 2]])`` fails; the function falls back
    to ``()`` and logs a warning (lines 182-188).

    [PIN] This pins the current behavior — nested lists are silently treated
    as empty shape.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="lighttrain"):
        result = _parse_shape("[[1, 2]]")
    assert result == ()
    assert "could not parse shape" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _default_namespace: edge-case — name starts with underscore (no first segment)
# ---------------------------------------------------------------------------


def test_invariant_default_namespace_empty_first_segment() -> None:
    """When the store dir name starts with ``_`` (e.g. ``_v2``), the first
    ``split('_')`` element is ``''``; the function falls back to the full name
    (not a hardcoded ``'aux'``) so the namespace stays meaningful."""
    assert _default_namespace("_v2") == "_v2"


def test_invariant_default_namespace_multi_segment() -> None:
    """Store name ``feat_v2`` → namespace ``feat`` (first underscore segment)."""
    assert _default_namespace("feat_v2") == "feat"


# ---------------------------------------------------------------------------
# Join entry: 'path' alias for 'store' key is accepted
# ---------------------------------------------------------------------------


def test_join_entry_path_alias_accepted(tmp_path) -> None:
    """A join entry using ``path`` instead of ``store`` as the root key is
    accepted; ``cfg.pop`` in ``_open_stores`` tries both (line 92)."""
    root = tmp_path / "art"
    _make_store(root, samples={"s1": {"v": torch.zeros(2)}})
    ds = ArtifactJoinedDataset(
        _MapDS([{"id": "s1"}]),
        join=[{"path": str(root), "namespace": "ns", "missing": "require"}],
    )
    row = ds[0]
    assert "aux.ns.v" in row


# ---------------------------------------------------------------------------
# allow_stale_artifact propagated from constructor default
# ---------------------------------------------------------------------------


def test_allow_stale_artifact_default_from_constructor(tmp_path) -> None:
    """``allow_stale_artifact=True`` at the dataset level is inherited by each
    join spec that does not override it; stale stores open without error."""
    root = tmp_path / "art"
    _make_store(
        root,
        samples={"s1": {"v": torch.zeros(2)}},
        producer_signature="v1",
    )
    # Provide a mismatched expected_header — would raise StaleArtifactError
    # unless allow_stale=True is propagated.
    ds = ArtifactJoinedDataset(
        _MapDS([{"id": "s1"}]),
        join=[
            {
                "store": str(root),
                "namespace": "ns",
                "expected_header": {"producer_signature": "v2"},
                "missing": "require",
            }
        ],
        allow_stale_artifact=True,
    )
    row = ds[0]
    assert row is not None
