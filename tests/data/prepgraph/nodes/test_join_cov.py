"""Coverage tests for ``lighttrain.builtin_plugins.data.prepgraph.nodes.join``.

Pins every previously-uncovered branch:

* L67  — ``JoinNode.__init__`` raises ``ValueError`` when ``stores`` is empty.
* L79  — ``JoinNode.run`` raises ``ValueError`` when inputs != 1.
* L85  — ``JoinNode.run`` raises ``RuntimeError`` when upstream has no rows.
* L94  — store spec with neither ``store`` nor ``path`` raises ``ValueError``.
* L140-145 — ``fill_zero`` branch populates ``aux.<ns>.<k>`` with zeros from
              ``header.field_schema`` and calls ``continue`` (not raise).
* L165-175 — ``_parse_shape`` covers tuple / list / int / parse-error paths.

Plus general edge-case tests for the public helpers ``_default_namespace`` and
``_join_one``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from lighttrain.builtin_plugins.data.artifacts import (
    ArtifactHeader,
    SafetensorsShardStore,
)

# Register built-in prep_node kinds (including "join")
from lighttrain.builtin_plugins.data.prepgraph import nodes as _nodes  # noqa: F401
from lighttrain.builtin_plugins.data.prepgraph.nodes.join import (
    JoinNode,
    _default_namespace,
    _join_one,
    _parse_shape,
)
from lighttrain.data.prepgraph.node import NodeResult, RunContext

# ---------------------------------------------------------------------------
# Store builder helper (avoids ModelForwardProducer / TinyCausalLM weight init)
# ---------------------------------------------------------------------------


def _build_store(
    root: Path,
    *,
    samples: dict[str, dict[str, torch.Tensor]],
    field_schema: dict[str, str] | None = None,
) -> Path:
    """Write a finalized safetensors-shards store in *root*."""
    header = ArtifactHeader(
        producer_signature="test-cov",
        dtype="torch.float32",
        field_schema=field_schema or {},
    )
    store = SafetensorsShardStore(root, header=header)
    for sid, tensors in samples.items():
        store.put(sid, tensors)
    store.finalize()
    return root


def _make_ctx(upstream_rows: list[dict[str, Any]], tmp_path: Path) -> RunContext:
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {
        "up": NodeResult(fingerprint="fp", schema_kind="rows", rows=list(upstream_rows))
    }
    return ctx


# ---------------------------------------------------------------------------
# L67 — JoinNode.__init__ raises on empty stores list
# ---------------------------------------------------------------------------


def test_invariant_init_raises_on_empty_stores() -> None:
    """JoinNode must reject an empty ``stores`` list at construction time (L67)."""
    with pytest.raises(ValueError, match="at least one entry in `stores:`"):
        JoinNode(
            name="bad",
            inputs=["up"],
            config={"stores": []},
        )


def test_invariant_init_raises_when_stores_absent() -> None:
    """JoinNode must reject a config with no ``stores`` key at all (L67)."""
    with pytest.raises(ValueError, match="at least one entry in `stores:`"):
        JoinNode(name="bad", inputs=["up"], config={})


# ---------------------------------------------------------------------------
# L79 — run() raises on wrong number of inputs
# ---------------------------------------------------------------------------


def test_invariant_run_raises_on_zero_inputs(tmp_path: Path) -> None:
    """JoinNode.run raises ValueError when the node was wired with 0 inputs (L79)."""
    # We must bypass __init__ store-path checks so use a dummy store entry;
    # the ValueError is thrown before the store is even opened.
    node = JoinNode(
        name="join",
        inputs=[],  # <-- wrong
        config={"stores": [{"store": "/dev/null"}]},
    )
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="expects exactly 1 input"):
        node.run(ctx)


def test_invariant_run_raises_on_two_inputs(tmp_path: Path) -> None:
    """JoinNode.run raises ValueError when wired with 2 inputs (L79)."""
    node = JoinNode(
        name="join",
        inputs=["a", "b"],
        config={"stores": [{"store": "/dev/null"}]},
    )
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="expects exactly 1 input"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# L85 — run() raises when upstream rows is None
# ---------------------------------------------------------------------------


def test_invariant_run_raises_when_upstream_rows_none(tmp_path: Path) -> None:
    """JoinNode.run raises RuntimeError when upstream NodeResult has rows=None (L85)."""
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={"stores": [{"store": "/dev/null"}]},
    )
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {
        "up": NodeResult(fingerprint="fp", schema_kind="rows", rows=None)
    }
    with pytest.raises(RuntimeError, match="has no rows"):
        node.run(ctx)


def test_invariant_run_raises_when_upstream_missing(tmp_path: Path) -> None:
    """JoinNode.run raises RuntimeError when the upstream key is absent (L85)."""
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={"stores": [{"store": "/dev/null"}]},
    )
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {}  # "up" not present
    with pytest.raises(RuntimeError, match="has no rows"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# L94 — store spec missing both "store" and "path" keys
# ---------------------------------------------------------------------------


def test_invariant_run_raises_on_store_spec_without_path(tmp_path: Path) -> None:
    """Each store spec must supply ``store`` (or ``path``) key; absence raises ValueError (L94)."""
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={"stores": [{"namespace": "ns"}]},  # neither "store" nor "path"
    )
    ctx = _make_ctx([{"id": "s0", "x": 1}], tmp_path)
    with pytest.raises(ValueError, match="requires `store`"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# L140-145 — fill_zero branch: missing sample → zeros from header.field_schema
# ---------------------------------------------------------------------------


def test_invariant_fill_zero_populates_zeros_from_field_schema(tmp_path: Path) -> None:
    """fill_zero substitutes zero-valued lists using ``header.field_schema`` shapes (L140-145).

    The aux key must exist, have the declared shape, and be all-zero.
    """
    torch.manual_seed(42)
    art_root = tmp_path / "art"
    _build_store(
        art_root,
        samples={"present": {"logits": torch.ones(4)}},
        field_schema={"logits": "(4,)"},
    )
    upstream_rows = [
        {"id": "present", "x": 1},
        {"id": "missing-sample", "x": 2},
    ]
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "teacher"}],
            "missing": "fill_zero",
        },
    )
    res = node.run(_make_ctx(upstream_rows, tmp_path))
    rows = list(res.rows)
    # Both rows survive (fill_zero does not drop)
    assert len(rows) == 2, rows
    missing_row = next(r for r in rows if r["id"] == "missing-sample")
    filled = missing_row["aux.teacher.logits"]
    # Stored as list (tolist())
    assert isinstance(filled, list)
    assert len(filled) == 4
    assert all(v == 0.0 for v in filled), filled


def test_pin_current_behavior_fill_zero_no_field_schema_no_aux_keys(
    tmp_path: Path,
) -> None:
    """Pin: fill_zero with empty field_schema adds no aux keys (L140-145 loop is no-op).

    DEBATABLE: the current code silently adds nothing when the store header has
    no field_schema entries.  A stricter design might raise, but we pin the
    current behaviour.
    """
    art_root = tmp_path / "art"
    _build_store(
        art_root,
        samples={"present": {"logits": torch.ones(2)}},
        field_schema={},  # deliberately empty
    )
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "teacher"}],
            "missing": "fill_zero",
        },
    )
    ctx = _make_ctx([{"id": "missing-sample"}], tmp_path)
    res = node.run(ctx)
    rows = list(res.rows)
    assert len(rows) == 1
    aux_keys = [k for k in rows[0] if k.startswith("aux.")]
    assert aux_keys == [], f"Expected no aux keys but got {aux_keys}"


# ---------------------------------------------------------------------------
# L165-175 — _parse_shape: all four code paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_str, expected",
    [
        ("(4,)", (4,)),          # tuple literal → returned as-is
        ("(2, 3)", (2, 3)),       # two-element tuple
        ("[5, 6]", (5, 6)),       # list literal → converted to tuple
        ("7", (7,)),              # bare int → wrapped in 1-tuple
    ],
)
def test_invariant_parse_shape_valid(shape_str: str, expected: tuple[int, ...]) -> None:
    """``_parse_shape`` must map valid shape strings to tuples correctly (L165-170)."""
    assert _parse_shape(shape_str) == expected


def test_invariant_parse_shape_int_scalar() -> None:
    """``_parse_shape`` wraps a bare integer in a 1-tuple (L171-172)."""
    result = _parse_shape("8")
    assert result == (8,)


def test_pin_current_behavior_parse_shape_unparseable_returns_empty_tuple() -> None:
    """Pin: ``_parse_shape`` returns ``()`` for any string that ``ast.literal_eval``
    cannot parse or whose type is not tuple/list/int (L173-175).

    DEBATABLE: the current code logs a warning and returns ``()`` rather than
    raising; this pins that lenient behaviour.
    """
    assert _parse_shape("not_a_shape") == ()
    assert _parse_shape("{'key': 1}") == ()  # dict — not tuple/list/int → ()


def test_invariant_parse_shape_float_returns_empty_tuple() -> None:
    """Pin: a float literal (not int/list/tuple) falls through to return ``()`` (L173-175)."""
    assert _parse_shape("3.14") == ()


# ---------------------------------------------------------------------------
# _default_namespace helper (L160-161)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_ns",
    [
        ("teacher_v1", "teacher"),
        ("aux", "aux"),
        ("simple", "simple"),
        ("a_b_c", "a"),
    ],
)
def test_invariant_default_namespace(name: str, expected_ns: str) -> None:
    """``_default_namespace`` returns the substring before the first underscore."""
    assert _default_namespace(name) == expected_ns


def test_invariant_default_namespace_leading_underscore_returns_name() -> None:
    """Fixed: a name whose first '_'-segment is empty (e.g. '_v1') falls back to
    the full name, not a hardcoded 'aux', so the namespace stays meaningful.
    """
    assert _default_namespace("_v1") == "_v1"


# ---------------------------------------------------------------------------
# _join_one standalone tests (L125-157)
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal stub satisfying the interface used by _join_one."""

    def __init__(
        self,
        data: dict[str, dict[str, Any]],
        *,
        field_schema: dict[str, str] | None = None,
    ) -> None:
        self._data = data
        self.root = Path("/fake/root")
        self.header = MagicMock()
        self.header.field_schema = field_schema or {}

    def contains(self, sid: str) -> bool:
        return sid in self._data

    def get(self, sid: str) -> dict[str, Any]:
        return self._data[sid]


def test_invariant_join_one_hit_returns_merged_row() -> None:
    """``_join_one`` merges found tensors into a copy of the row."""
    store = _FakeStore({"s0": {"feat": torch.tensor([1.0, 2.0])}})
    cfg = {"namespace": "ns", "missing": "require"}
    result = _join_one({"id": "s0", "x": 9}, [(cfg, store)], "id")
    assert result is not None
    assert result["x"] == 9
    assert "aux.ns.feat" in result


def test_invariant_join_one_drop_returns_none_on_miss() -> None:
    """``_join_one`` returns None when missing='drop' and sample absent."""
    store = _FakeStore({})
    cfg = {"namespace": "ns", "missing": "drop"}
    result = _join_one({"id": "gone"}, [(cfg, store)], "id")
    assert result is None


def test_invariant_join_one_require_raises_on_miss() -> None:
    """``_join_one`` raises KeyError when missing='require' and sample absent."""
    store = _FakeStore({"other": {}})
    cfg = {"namespace": "ns", "missing": "require"}
    with pytest.raises(KeyError, match="not present"):
        _join_one({"id": "gone"}, [(cfg, store)], "id")


def test_invariant_join_one_fill_zero_uses_field_schema_shape() -> None:
    """``_join_one`` fill_zero branch substitutes zeros using field_schema."""
    store = _FakeStore({}, field_schema={"logits": "(3,)"})
    cfg = {"namespace": "ns", "missing": "fill_zero"}
    result = _join_one({"id": "miss"}, [(cfg, store)], "id")
    assert result is not None
    assert "aux.ns.logits" in result
    assert result["aux.ns.logits"] == [0.0, 0.0, 0.0]


def test_invariant_join_one_tensor_tolist_called() -> None:
    """``_join_one`` calls ``.tolist()`` on tensor values when present."""
    t = torch.tensor([7.0, 8.0])
    store = _FakeStore({"s1": {"vec": t}})
    cfg = {"namespace": "ns", "missing": "require"}
    result = _join_one({"id": "s1"}, [(cfg, store)], "id")
    assert result is not None
    assert result["aux.ns.vec"] == [7.0, 8.0]


def test_invariant_join_one_non_tensor_value_stored_as_is() -> None:
    """``_join_one`` stores plain Python values (no tolist) as-is."""
    store = _FakeStore({"s1": {"score": 0.99}})
    cfg = {"namespace": "ns", "missing": "require"}
    result = _join_one({"id": "s1"}, [(cfg, store)], "id")
    assert result is not None
    assert result["aux.ns.score"] == pytest.approx(0.99)


def test_invariant_join_one_derive_sample_id_when_id_key_absent() -> None:
    """``_join_one`` falls back to ``derive_sample_id`` when row lacks the id key."""
    from lighttrain.data.core._schema import derive_sample_id

    row = {"input_ids": [1, 2, 3]}
    derived = derive_sample_id(row)
    store = _FakeStore({derived: {"feat": torch.zeros(2)}})
    cfg = {"namespace": "ns", "missing": "require"}
    result = _join_one(row, [(cfg, store)], "id")
    assert result is not None
    # The derived id must have been written back into the row
    assert result["id"] == derived


# ---------------------------------------------------------------------------
# Integration: fill_zero in full JoinNode.run pipeline (multi-row)
# ---------------------------------------------------------------------------


def test_invariant_run_fill_zero_full_pipeline(tmp_path: Path) -> None:
    """End-to-end: fill_zero rows survive the run() loop and carry zeros (L140-145)."""
    torch.manual_seed(0)
    art_root = tmp_path / "art"
    _build_store(
        art_root,
        samples={"hit": {"emb": torch.ones(2, 3)}},
        field_schema={"emb": "(2, 3)"},
    )
    upstream_rows = [
        {"id": "hit"},
        {"id": "miss1"},
        {"id": "miss2"},
    ]
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "enc"}],
            "missing": "fill_zero",
        },
    )
    res = node.run(_make_ctx(upstream_rows, tmp_path))
    rows = list(res.rows)
    assert len(rows) == 3
    hit_row = next(r for r in rows if r["id"] == "hit")
    # Real tensor stored as flat list from tolist()
    assert hit_row["aux.enc.emb"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    for row in rows:
        if row["id"] != "hit":
            filled = row["aux.enc.emb"]
            # zeros() of shape (2, 3) → [[0,0,0],[0,0,0]]
            assert filled == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# Integration: extras dict contains row_count
# ---------------------------------------------------------------------------


def test_invariant_run_result_extras_row_count(tmp_path: Path) -> None:
    """``NodeResult.extras['row_count']`` must match the number of surviving rows."""
    art_root = tmp_path / "art"
    _build_store(art_root, samples={"s0": {"v": torch.zeros(1)}})
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"store": str(art_root), "namespace": "ns"}],
            "missing": "drop",
        },
    )
    upstream_rows = [{"id": "s0"}, {"id": "gone"}]
    res = node.run(_make_ctx(upstream_rows, tmp_path))
    rows = list(res.rows)
    assert res.extras["row_count"] == len(rows) == 1


# ---------------------------------------------------------------------------
# Integration: store spec supports "path" as alias for "store"
# ---------------------------------------------------------------------------


def test_invariant_run_supports_path_alias_for_store_key(tmp_path: Path) -> None:
    """Store spec with ``path`` key (instead of ``store``) must be accepted."""
    art_root = tmp_path / "art"
    _build_store(art_root, samples={"s0": {"v": torch.zeros(1)}})
    node = JoinNode(
        name="join",
        inputs=["up"],
        config={
            "stores": [{"path": str(art_root), "namespace": "ns"}],
            "missing": "require",
        },
    )
    res = node.run(_make_ctx([{"id": "s0"}], tmp_path))
    rows = list(res.rows)
    assert len(rows) == 1
    assert "aux.ns.v" in rows[0]
