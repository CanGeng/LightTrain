"""Tests for ``lighttrain.builtin_plugins.data.prepgraph.nodes.materialize``.

Coverage targets (previously uncovered lines):

  44  — no inputs → ValueError
  62  — ``elif layout == "memmap":`` branch entered
  63  — seq_len extracted from config
  64  — seq_len <= 0 guard (line 64 condition)
  65  — ValueError raised for seq_len <= 0
  68  — dtype extracted from config
  69  — fields extracted from config (with and without custom value)
  74  — dtypes dict built from fields × dtype
  75  — extras["seq_len"] set
  76  — extras["dtype"] set
  77  — extras["fields"] set
  78  — write_memmap called
  81  — store = MemmapDataset(out_dir) returned
  83  — unknown layout → ValueError

General edge cases also pinned:

  * rows layout with default config
  * rows layout with parquet fmt
  * rows layout shard_size coercion
  * NodeResult fields and store accessibility
  * registration in global registry
  * __all__ export
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lighttrain.builtin_plugins.data.prepgraph.nodes.materialize import MaterializeNode
from lighttrain.data.cache._memmap import MemmapDataset
from lighttrain.data.cache._rows import _RowsDataset
from lighttrain.data.prepgraph.node import NodeResult, RunContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(tmp_path: Path, upstream: dict[str, Any] | None = None) -> RunContext:
    """Build a minimal RunContext pointing at *tmp_path*."""
    from lighttrain.data.prepgraph.node import RunContext
    return RunContext(
        store_root=tmp_path,
        workers=1,
        upstream=upstream or {},
    )


def _upstream_result(rows: list[dict]) -> NodeResult:
    """Wrap an in-memory row list in a NodeResult (simulates an upstream node)."""
    return NodeResult(fingerprint="fp0", rows=rows)


def _make_rows_node(name: str = "mat", config: dict | None = None) -> MaterializeNode:
    """Return a MaterializeNode with the given name, inputs=['src'], and config."""
    return MaterializeNode(
        name=name,
        inputs=["src"],
        config=config or {},
    )


def _make_memmap_rows(n: int, seq_len: int) -> list[dict]:
    """Build *n* synthetic rows, each with three int lists of length *seq_len*."""
    return [
        {
            "input_ids": list(range(i, i + seq_len)),
            "position_ids": list(range(seq_len)),
            "document_ids": [0] * seq_len,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Class-level invariants
# ---------------------------------------------------------------------------


def test_invariant_kind_and_schema_kind() -> None:
    """``kind`` is 'materialize' and ``schema_kind`` is 'materialized'."""
    assert MaterializeNode.kind == "materialize"
    assert MaterializeNode.schema_kind == "materialized"


def test_invariant_all_exports() -> None:
    """``__all__`` contains ``MaterializeNode``."""
    from lighttrain.builtin_plugins.data.prepgraph.nodes.materialize import __all__
    assert "MaterializeNode" in __all__


def test_invariant_registered_in_registry() -> None:
    """``MaterializeNode`` is registered under ``prep_node / materialize``."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.materialize  # noqa: F401
    from lighttrain.registry import get_registry

    entry = get_registry().get("prep_node", "materialize")
    assert entry is MaterializeNode


# ---------------------------------------------------------------------------
# Line 44 — no inputs → ValueError
# ---------------------------------------------------------------------------


def test_invariant_no_inputs_raises_value_error(tmp_path: Path) -> None:
    """``run()`` raises ``ValueError`` when the node has no upstream inputs (line 44)."""
    node = MaterializeNode(name="mat", inputs=[], config={})
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="requires upstream input"):
        node.run(ctx)


def test_invariant_no_inputs_error_includes_node_name(tmp_path: Path) -> None:
    """The ValueError message contains the node name (line 44)."""
    node = MaterializeNode(name="my_special_mat", inputs=[], config={})
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="my_special_mat"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# rows layout (default) — lines 53-61
# ---------------------------------------------------------------------------


def test_invariant_rows_layout_default_returns_node_result(tmp_path: Path) -> None:
    """Default layout=``rows`` returns a NodeResult with a _RowsDataset store."""
    rows = [{"a": 1}, {"b": 2}, {"c": 3}]
    node = _make_rows_node()
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert isinstance(result, NodeResult)
    assert result.schema_kind == "materialized"
    assert result.fingerprint == ""


def test_invariant_rows_layout_store_is_rows_dataset(tmp_path: Path) -> None:
    """rows layout attaches a ``_RowsDataset`` as the store."""
    rows = [{"x": i} for i in range(5)]
    node = _make_rows_node()
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert isinstance(result.store, _RowsDataset)


def test_invariant_rows_layout_extras(tmp_path: Path) -> None:
    """rows layout extras contain ``row_count``, ``layout``, ``fmt``, and ``shards``."""
    rows = [{"i": i} for i in range(7)]
    node = _make_rows_node(config={"layout": "rows", "fmt": "jsonl"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    e = result.extras
    assert e["row_count"] == 7
    assert e["layout"] == "rows"
    assert e["fmt"] == "jsonl"
    assert isinstance(e["shards"], int)
    assert e["shards"] >= 1


def test_invariant_rows_layout_rows_field_contains_input_rows(tmp_path: Path) -> None:
    """``result.rows`` is the same list of rows passed in by the upstream."""
    input_rows = [{"k": 0}, {"k": 1}]
    node = _make_rows_node()
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(input_rows)})
    result = node.run(ctx)
    assert result.rows == input_rows


def test_invariant_rows_layout_store_readable(tmp_path: Path) -> None:
    """The _RowsDataset store reads back the written shards correctly."""
    rows = [{"text": f"item{i}"} for i in range(4)]
    node = _make_rows_node()
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    store = result.store
    assert len(store) == 4
    assert store[0]["text"] == "item0"
    assert store[3]["text"] == "item3"


def test_invariant_rows_layout_empty_upstream(tmp_path: Path) -> None:
    """rows layout with an empty upstream list writes 0 rows and creates the store."""
    node = _make_rows_node()
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result([])})
    result = node.run(ctx)
    assert result.extras["row_count"] == 0
    assert len(result.store) == 0


def test_invariant_rows_layout_custom_shard_size(tmp_path: Path) -> None:
    """A shard_size of 2 over 5 rows produces multiple shards."""
    rows = [{"n": i} for i in range(5)]
    node = _make_rows_node(config={"layout": "rows", "shard_size": 2})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert result.extras["row_count"] == 5
    assert result.extras["shards"] >= 2


def test_invariant_rows_layout_parquet_fmt(tmp_path: Path) -> None:
    """rows layout with ``fmt='parquet'`` sets extras['fmt'] to parquet."""
    rows = [{"val": 42}]
    node = _make_rows_node(config={"layout": "rows", "fmt": "parquet"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert result.extras["fmt"] == "parquet"


# ---------------------------------------------------------------------------
# memmap layout — lines 62-81
# ---------------------------------------------------------------------------


def test_invariant_memmap_layout_returns_node_result(tmp_path: Path) -> None:
    """memmap layout succeeds and returns a NodeResult (line 62, 81)."""
    rows = _make_memmap_rows(3, 8)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 8})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert isinstance(result, NodeResult)


def test_invariant_memmap_layout_store_is_memmap_dataset(tmp_path: Path) -> None:
    """memmap layout attaches a ``MemmapDataset`` as the store (line 81)."""
    rows = _make_memmap_rows(2, 4)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 4})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert isinstance(result.store, MemmapDataset)


def test_invariant_memmap_layout_extras(tmp_path: Path) -> None:
    """memmap extras contain seq_len, dtype, layout, fields, row_count (lines 75-77)."""
    rows = _make_memmap_rows(4, 16)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 16, "dtype": "int32"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    e = result.extras
    assert e["seq_len"] == 16
    assert e["dtype"] == "int32"
    assert e["layout"] == "memmap"
    assert e["row_count"] == 4
    assert isinstance(e["fields"], list)
    assert len(e["fields"]) >= 1


def test_invariant_memmap_layout_default_dtype_is_int64(tmp_path: Path) -> None:
    """When dtype is absent, it defaults to 'int64' (line 68)."""
    rows = _make_memmap_rows(2, 8)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 8})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert result.extras["dtype"] == "int64"


def test_invariant_memmap_layout_default_fields(tmp_path: Path) -> None:
    """Default fields are ('input_ids', 'position_ids', 'document_ids') (line 69)."""
    rows = _make_memmap_rows(2, 4)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 4})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert set(result.extras["fields"]) == {"input_ids", "position_ids", "document_ids"}


def test_invariant_memmap_layout_custom_fields(tmp_path: Path) -> None:
    """Custom fields override the default (line 69)."""
    rows = [{"x": [1, 2, 3, 4], "y": [0, 0, 0, 0]} for _ in range(2)]
    node = _make_rows_node(
        config={"layout": "memmap", "seq_len": 4, "fields": ["x", "y"]}
    )
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert set(result.extras["fields"]) == {"x", "y"}


def test_invariant_memmap_layout_dtypes_dict_uses_dtype(tmp_path: Path) -> None:
    """Each field in the dtypes dict uses the configured dtype (line 74)."""
    rows = _make_memmap_rows(2, 4)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 4, "dtype": "int32"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    store: MemmapDataset = result.store
    # The header dtypes must all be int32
    for f, dt in store.header.dtypes.items():
        assert dt == "int32", f"field {f} has dtype {dt!r}, expected 'int32'"


def test_invariant_memmap_layout_data_readable_by_dataset(tmp_path: Path) -> None:
    """MemmapDataset reads back the written rows correctly (write_memmap path, line 78)."""
    seq_len = 8
    rows = _make_memmap_rows(3, seq_len)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": seq_len})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    store: MemmapDataset = result.store
    assert len(store) == 3
    # First item's input_ids starts from 0 (matches the synthetic row builder)
    assert store[0]["input_ids"][0] == 0


def test_pin_current_behavior_memmap_empty_rows_raises(tmp_path: Path) -> None:
    """Pin current behavior: memmap layout with empty upstream raises ``ValueError``
    when constructing MemmapDataset because numpy cannot mmap an empty file.

    This is an unsupported edge case in the current implementation — line 81 is
    still exercised (store = MemmapDataset(out_dir)); the exception propagates
    from numpy's mmap layer. If zero-row support is ever added, update this test.
    """
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 4})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result([])})
    with pytest.raises((ValueError, FileNotFoundError)):
        node.run(ctx)


def test_invariant_memmap_seq_len_zero_raises(tmp_path: Path) -> None:
    """seq_len=0 raises ``ValueError`` about needing ``seq_len`` (line 64-65)."""
    rows = _make_memmap_rows(2, 4)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": 0})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    with pytest.raises(ValueError, match="seq_len"):
        node.run(ctx)


def test_invariant_memmap_seq_len_negative_raises(tmp_path: Path) -> None:
    """Negative seq_len raises ``ValueError`` (line 64-65)."""
    rows = _make_memmap_rows(2, 4)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": -1})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    with pytest.raises(ValueError, match="seq_len"):
        node.run(ctx)


def test_invariant_memmap_seq_len_error_includes_node_name(tmp_path: Path) -> None:
    """The seq_len ValueError includes the node name (line 65)."""
    node = _make_rows_node(name="pack_mat", config={"layout": "memmap", "seq_len": 0})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result([{"x": 1}])})
    with pytest.raises(ValueError, match="pack_mat"):
        node.run(ctx)


def test_invariant_memmap_seq_len_missing_raises(tmp_path: Path) -> None:
    """Absent ``seq_len`` key defaults to 0, triggering the guard (line 63-65)."""
    node = _make_rows_node(config={"layout": "memmap"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result([{"x": 1}])})
    with pytest.raises(ValueError, match="seq_len"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# Line 83 — unknown layout → ValueError
# ---------------------------------------------------------------------------


def test_invariant_unknown_layout_raises(tmp_path: Path) -> None:
    """An unrecognised ``layout`` value raises ValueError (line 83)."""
    rows = [{"a": 1}]
    node = _make_rows_node(config={"layout": "zarr"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    with pytest.raises(ValueError, match="unknown layout"):
        node.run(ctx)


def test_invariant_unknown_layout_error_includes_layout_value(tmp_path: Path) -> None:
    """The unknown-layout ValueError names the offending layout (line 83)."""
    rows = [{"a": 1}]
    node = _make_rows_node(config={"layout": "fancy_layout"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    with pytest.raises(ValueError, match="fancy_layout"):
        node.run(ctx)


def test_invariant_unknown_layout_error_includes_node_name(tmp_path: Path) -> None:
    """The unknown-layout error message contains the node name (line 83)."""
    rows = [{"a": 1}]
    node = _make_rows_node(name="bad_mat", config={"layout": "nope"})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    with pytest.raises(ValueError, match="bad_mat"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# Parametrized layout / dtype matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 4, 32])
def test_invariant_memmap_various_seq_lens(tmp_path: Path, seq_len: int) -> None:
    """memmap layout works across several seq_len values."""
    rows = _make_memmap_rows(2, seq_len)
    node = _make_rows_node(config={"layout": "memmap", "seq_len": seq_len})
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})
    result = node.run(ctx)
    assert result.extras["seq_len"] == seq_len
    assert len(result.store) == 2


@pytest.mark.parametrize(
    "layout, bad_val",
    [
        ("rows", None),          # does NOT raise
        ("memmap", None),        # seq_len=0 raises
        ("", None),              # empty string treated as unknown layout
        ("ROWS", None),          # case-sensitive: treated as unknown
    ],
)
def test_pin_current_behavior_layout_case_sensitive(
    tmp_path: Path, layout: str, bad_val: Any
) -> None:
    """Pin: layout matching is case-sensitive; 'ROWS'/'memmap' without seq_len etc.
    behave as documented.

    Note: this pins current behavior. If layout normalisation is ever added,
    update accordingly.
    """
    rows = [{"a": 1}]
    config: dict = {"layout": layout}
    if layout == "memmap":
        config["seq_len"] = 0  # will raise ValueError about seq_len
    node = _make_rows_node(config=config)
    ctx = _ctx(tmp_path, upstream={"src": _upstream_result(rows)})

    if layout == "rows":
        # Normal happy path
        result = node.run(ctx)
        assert result.extras["layout"] == "rows"
    elif layout == "memmap":
        with pytest.raises(ValueError, match="seq_len"):
            node.run(ctx)
    else:
        # "" or "ROWS" fall into the else branch → unknown layout
        with pytest.raises(ValueError, match="unknown layout"):
            node.run(ctx)


# ---------------------------------------------------------------------------
# Upstream selection: first input wins
# ---------------------------------------------------------------------------


def test_invariant_first_input_is_used_when_multiple_defined(tmp_path: Path) -> None:
    """When multiple inputs are declared, only the first is consumed (line 45)."""
    rows_a = [{"src": "a"}]
    rows_b = [{"src": "b"}]
    node = MaterializeNode(
        name="mat", inputs=["a", "b"], config={"layout": "rows"}
    )
    ctx = _ctx(
        tmp_path,
        upstream={
            "a": _upstream_result(rows_a),
            "b": _upstream_result(rows_b),
        },
    )
    result = node.run(ctx)
    # Only the 'a' source rows should appear
    assert result.rows == rows_a
    assert result.extras["row_count"] == 1
