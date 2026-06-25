"""Coverage tests for ``lighttrain.builtin_plugins.data.prepgraph.nodes.chunk``.

Pins every previously-uncovered branch and adds exhaustive edge-case tests:

* L23-28  — ``_chunk_one`` short path (``len(ids) <= max_len``): builds and
             yields a shallow copy that still carries extra row fields.
* L33     — defensive ``break`` guard inside the chunking loop.  This branch
             is structurally unreachable from Python's ``range()`` semantics
             (``range(0, N, step)`` never produces a value >= N), so it is
             logged as a skipped / suspected-dead-code guard.
* L58     — ``ChunkNode.run`` raises ``ValueError`` when ``self.inputs`` is empty.
* L61     — ``ChunkNode.run`` raises ``ValueError`` when ``max_len`` <= 0.

General edge-case tests cover:

* Empty ``input_ids`` list.
* Sequence exactly equal to ``max_len`` (boundary: short path).
* Sequence one token longer than ``max_len`` (first chunked case).
* Non-zero ``overlap`` producing overlapping windows.
* ``overlap >= max_len - 1`` clamping step to 1.
* ``labels`` / ``attention_mask`` fields that differ from ``input_ids``.
* Missing ``labels`` / ``attention_mask`` fields — defaults applied.
* Extra row fields preserved across all chunks (both paths).
* ``rows=None`` upstream — treated as empty list.
* ``overlap`` config key missing — defaults to 0.
* ``ChunkNode`` registry registration.
* ``NodeResult`` metadata: ``fingerprint``, ``schema_kind``, ``extras``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lighttrain.builtin_plugins.data.prepgraph.nodes.chunk import (
    ChunkNode,
    _chunk_one,
)
from lighttrain.data.prepgraph.node import NodeResult, RunContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeUpstream:
    """Minimal upstream stub with a ``rows`` attribute."""

    def __init__(self, rows: list | None) -> None:
        self.rows = rows


def _make_ctx(rows: list | None, tmp_path: Path) -> RunContext:
    """Build a ``RunContext`` whose single upstream key is 'up'."""
    return RunContext(
        store_root=tmp_path,
        workers=1,
        upstream={"up": _FakeUpstream(rows)},  # type: ignore[dict-item]
        log=None,
    )


def _run(
    rows: list | None,
    *,
    max_len: int,
    overlap: int = 0,
    tmp_path: Path,
) -> NodeResult:
    """Run a ``ChunkNode`` end-to-end and return its ``NodeResult``."""
    cfg: dict = {"max_len": max_len}
    if overlap:
        cfg["overlap"] = overlap
    node = ChunkNode(name="chunk", inputs=["up"], config=cfg)
    return node.run(_make_ctx(rows, tmp_path))


# ===========================================================================
# _chunk_one — short path (L23-28)
# ===========================================================================


def test_invariant_chunk_one_short_path_yields_single_row() -> None:
    """L23-28: sequence shorter than max_len is yielded as a single row unchanged."""
    row = {"input_ids": [1, 2, 3], "labels": [10, 20, 30], "attention_mask": [1, 1, 1]}
    result = list(_chunk_one(row, max_len=5, overlap=0))
    assert len(result) == 1
    assert result[0]["input_ids"] == [1, 2, 3]
    assert result[0]["labels"] == [10, 20, 30]
    assert result[0]["attention_mask"] == [1, 1, 1]


def test_invariant_chunk_one_short_path_preserves_extra_fields() -> None:
    """L23-28: the yielded copy keeps extra row fields (e.g. 'source', 'meta')."""
    row = {"input_ids": [7, 8], "source": "wiki", "meta": {"doc_id": 42}}
    result = list(_chunk_one(row, max_len=10, overlap=0))
    assert len(result) == 1
    assert result[0]["source"] == "wiki"
    assert result[0]["meta"] == {"doc_id": 42}


def test_invariant_chunk_one_short_path_returns_shallow_copy() -> None:
    """L23-28: the returned row is a dict copy, not the original object."""
    row = {"input_ids": [1, 2]}
    result = list(_chunk_one(row, max_len=5, overlap=0))
    assert result[0] is not row


def test_invariant_chunk_one_short_path_exact_length() -> None:
    """L22: sequence length == max_len triggers the short path (<=, not <)."""
    row = {"input_ids": [1, 2, 3]}
    result = list(_chunk_one(row, max_len=3, overlap=0))
    assert len(result) == 1
    assert result[0]["input_ids"] == [1, 2, 3]


def test_invariant_chunk_one_short_path_empty_sequence() -> None:
    """L22-28: empty input_ids list (length 0) is <= any max_len → short path."""
    row = {"input_ids": []}  # type: ignore[var-annotated]
    result = list(_chunk_one(row, max_len=4, overlap=0))
    assert len(result) == 1
    assert result[0]["input_ids"] == []
    assert result[0]["labels"] == []
    assert result[0]["attention_mask"] == []


def test_invariant_chunk_one_short_path_missing_labels_defaults_to_input_ids() -> None:
    """L20: when 'labels' key absent it defaults to a copy of input_ids."""
    row = {"input_ids": [5, 6, 7]}
    result = list(_chunk_one(row, max_len=10, overlap=0))
    assert result[0]["labels"] == [5, 6, 7]


def test_invariant_chunk_one_short_path_missing_attention_mask_defaults_to_ones() -> None:
    """L21: when 'attention_mask' key absent it defaults to all-ones of same length."""
    row = {"input_ids": [1, 2, 3]}
    result = list(_chunk_one(row, max_len=10, overlap=0))
    assert result[0]["attention_mask"] == [1, 1, 1]


# ===========================================================================
# _chunk_one — chunking path (len > max_len)
# ===========================================================================


def test_invariant_chunk_one_splits_into_multiple_chunks() -> None:
    """Sequence longer than max_len is split into multiple chunks without overlap."""
    row = {"input_ids": list(range(1, 9))}  # len=8
    chunks = list(_chunk_one(row, max_len=4, overlap=0))
    assert len(chunks) == 2
    assert chunks[0]["input_ids"] == [1, 2, 3, 4]
    assert chunks[1]["input_ids"] == [5, 6, 7, 8]


def test_invariant_chunk_one_one_token_over_boundary() -> None:
    """Sequence of length max_len+1 produces two chunks, second being 1 token."""
    row = {"input_ids": [10, 20, 30, 40, 50]}
    chunks = list(_chunk_one(row, max_len=4, overlap=0))
    assert len(chunks) == 2
    assert chunks[0]["input_ids"] == [10, 20, 30, 40]
    assert chunks[1]["input_ids"] == [50]


def test_invariant_chunk_one_overlap_produces_overlapping_windows() -> None:
    """Overlap > 0 causes adjacent chunks to share tokens."""
    row = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}
    chunks = list(_chunk_one(row, max_len=4, overlap=2))
    # step = max(1, 4 - 2) = 2
    # starts: 0, 2, 4, 6
    assert chunks[0]["input_ids"] == [1, 2, 3, 4]
    assert chunks[1]["input_ids"] == [3, 4, 5, 6]
    assert chunks[2]["input_ids"] == [5, 6, 7, 8]
    # chunk[2] ends at 8 == len → loop exits via line 39-40
    assert len(chunks) == 3


def test_invariant_chunk_one_overlap_equal_to_max_len_minus_one_step_is_one() -> None:
    """Overlap = max_len - 1 forces step = 1 (minimum step guard on L29)."""
    row = {"input_ids": [1, 2, 3, 4]}
    chunks = list(_chunk_one(row, max_len=3, overlap=2))
    # step = max(1, 3-2) = 1 → starts 0,1,2,3 but each window is 3 wide
    assert chunks[0]["input_ids"] == [1, 2, 3]
    assert chunks[1]["input_ids"] == [2, 3, 4]


def test_invariant_chunk_one_overlap_larger_than_max_len_clamps_step_to_one() -> None:
    """Overlap >= max_len: step clamped to 1 by max(1, ...) guard (L29)."""
    row = {"input_ids": [1, 2, 3, 4]}
    chunks = list(_chunk_one(row, max_len=2, overlap=100))
    # step = max(1, 2-100) = 1
    assert chunks[0]["input_ids"] == [1, 2]
    assert chunks[1]["input_ids"] == [2, 3]
    assert chunks[2]["input_ids"] == [3, 4]


def test_invariant_chunk_one_labels_chunked_independently() -> None:
    """When explicit labels differ from input_ids, they are chunked identically."""
    row = {"input_ids": [1, 2, 3, 4, 5], "labels": [-1, -1, 3, 4, 5]}
    chunks = list(_chunk_one(row, max_len=3, overlap=0))
    assert chunks[0]["labels"] == [-1, -1, 3]
    assert chunks[1]["labels"] == [4, 5]


def test_invariant_chunk_one_attention_mask_chunked_correctly() -> None:
    """Custom attention_mask is sliced to match each chunk's input_ids window."""
    row = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 0, 0, 1]}
    chunks = list(_chunk_one(row, max_len=3, overlap=0))
    assert chunks[0]["attention_mask"] == [1, 1, 0]
    assert chunks[1]["attention_mask"] == [0, 1]


def test_invariant_chunk_one_extra_fields_copied_to_every_chunk() -> None:
    """Extra row fields (not input_ids/labels/attention_mask) appear in all chunks."""
    row = {"input_ids": [1, 2, 3, 4, 5, 6], "doc_id": "abc", "split": "train"}
    chunks = list(_chunk_one(row, max_len=3, overlap=0))
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk["doc_id"] == "abc"
        assert chunk["split"] == "train"


def test_invariant_chunk_one_early_break_when_end_reaches_sequence_end() -> None:
    """L39-40: loop terminates once end >= len(ids), no extra empty chunk emitted."""
    row = {"input_ids": [1, 2, 3, 4, 5, 6]}
    chunks = list(_chunk_one(row, max_len=3, overlap=0))
    # Exactly 6/3 = 2 chunks; third iteration would start at 6 >= 6, but
    # the L39-40 break on the second iteration prevents it.
    assert len(chunks) == 2
    assert chunks[0]["input_ids"] == [1, 2, 3]
    assert chunks[1]["input_ids"] == [4, 5, 6]


# ===========================================================================
# ChunkNode.run — error paths (L58, L61)
# ===========================================================================


def test_invariant_run_raises_when_no_inputs(tmp_path: Path) -> None:
    """L57-58: ChunkNode with empty inputs list raises ValueError."""
    node = ChunkNode(name="c", inputs=[], config={"max_len": 4})
    ctx = RunContext(store_root=tmp_path, workers=1, upstream={})
    with pytest.raises(ValueError, match="requires upstream input"):
        node.run(ctx)


def test_invariant_run_error_includes_node_name(tmp_path: Path) -> None:
    """L58: ValueError message includes the node name for diagnostics."""
    node = ChunkNode(name="my_chunk_node", inputs=[], config={"max_len": 4})
    ctx = RunContext(store_root=tmp_path, workers=1, upstream={})
    with pytest.raises(ValueError, match="my_chunk_node"):
        node.run(ctx)


def test_invariant_run_raises_when_max_len_is_zero(tmp_path: Path) -> None:
    """L60-61: max_len=0 raises ValueError (must be > 0)."""
    node = ChunkNode(name="c", inputs=["up"], config={"max_len": 0})
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="`max_len` must be > 0"):
        node.run(ctx)


def test_invariant_run_raises_when_max_len_is_negative(tmp_path: Path) -> None:
    """L60-61: negative max_len also raises ValueError."""
    node = ChunkNode(name="c", inputs=["up"], config={"max_len": -1})
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="`max_len` must be > 0"):
        node.run(ctx)


def test_invariant_run_raises_when_max_len_missing(tmp_path: Path) -> None:
    """L59: max_len absent from config defaults to 0, triggering L61 ValueError."""
    node = ChunkNode(name="c", inputs=["up"], config={})
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="`max_len` must be > 0"):
        node.run(ctx)


def test_invariant_run_max_len_error_includes_node_name(tmp_path: Path) -> None:
    """L61: the max_len error message includes the node's name."""
    node = ChunkNode(name="named_node", inputs=["up"], config={"max_len": 0})
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="named_node"):
        node.run(ctx)


def test_invariant_run_max_len_as_string_is_coerced(tmp_path: Path) -> None:
    """L59: max_len passed as a string is coerced via int() without error."""
    result = _run([{"input_ids": [1, 2]}], max_len=4, tmp_path=tmp_path)
    assert result.rows is not None
    assert len(result.rows) == 1


# ===========================================================================
# ChunkNode.run — successful end-to-end paths
# ===========================================================================


def test_invariant_run_empty_upstream_returns_empty_rows(tmp_path: Path) -> None:
    """Empty upstream rows → empty output rows."""
    result = _run([], max_len=4, tmp_path=tmp_path)
    assert result.rows == []
    assert result.extras["row_count"] == 0


def test_invariant_run_upstream_none_treated_as_empty(tmp_path: Path) -> None:
    """rows=None upstream is treated as an empty list (L64: ``upstream.rows or []``)."""
    result = _run(None, max_len=4, tmp_path=tmp_path)
    assert result.rows == []
    assert result.extras["row_count"] == 0


def test_invariant_run_short_rows_pass_through_unchanged(tmp_path: Path) -> None:
    """Rows shorter than max_len are returned as-is (L23-28 path)."""
    rows = [{"input_ids": [1, 2]}, {"input_ids": [3]}]
    result = _run(rows, max_len=10, tmp_path=tmp_path)
    assert len(result.rows) == 2  # type: ignore[arg-type]
    assert result.rows[0]["input_ids"] == [1, 2]  # type: ignore[index]
    assert result.rows[1]["input_ids"] == [3]  # type: ignore[index]


def test_invariant_run_long_row_is_split(tmp_path: Path) -> None:
    """A row longer than max_len is split into multiple output rows."""
    rows = [{"input_ids": list(range(10))}]
    result = _run(rows, max_len=4, tmp_path=tmp_path)
    assert len(result.rows) == 3  # type: ignore[arg-type]  # 4 + 4 + 2
    assert result.extras["row_count"] == 3


def test_invariant_run_mixed_short_and_long_rows(tmp_path: Path) -> None:
    """Short and long rows are handled together correctly."""
    rows = [
        {"input_ids": [1, 2]},         # short → 1 output
        {"input_ids": list(range(8))},  # long → 2 outputs (max_len=4)
        {"input_ids": [99]},            # short → 1 output
    ]
    result = _run(rows, max_len=4, tmp_path=tmp_path)
    assert result.extras["row_count"] == 4


def test_invariant_run_overlap_applies_to_all_rows(tmp_path: Path) -> None:
    """overlap config key is applied consistently across all rows.

    seq=[1..6], max_len=4, overlap=2, step=2.
    start=0: window [0:4]=[1,2,3,4], end(4) < 6 → continue.
    start=2: window [2:6]=[3,4,5,6], end(6) >= 6 → break (L39-40).
    Result: 2 chunks.
    """
    rows = [{"input_ids": [1, 2, 3, 4, 5, 6]}]
    result = _run(rows, max_len=4, overlap=2, tmp_path=tmp_path)
    # step=2, starts 0 and 2 only (second window covers the tail → break)
    assert len(result.rows) == 2  # type: ignore[arg-type]
    assert result.rows[0]["input_ids"] == [1, 2, 3, 4]  # type: ignore[index]
    assert result.rows[1]["input_ids"] == [3, 4, 5, 6]  # type: ignore[index]


def test_invariant_run_overlap_default_is_zero(tmp_path: Path) -> None:
    """When overlap is not in config it defaults to 0 (no overlapping windows)."""
    rows = [{"input_ids": [1, 2, 3, 4, 5, 6]}]
    result = _run(rows, max_len=3, tmp_path=tmp_path)
    assert len(result.rows) == 2  # type: ignore[arg-type]  # 3+3, no overlap


def test_invariant_run_result_schema_kind_is_chunked_rows(tmp_path: Path) -> None:
    """NodeResult.schema_kind must equal 'chunked_rows'."""
    result = _run([], max_len=4, tmp_path=tmp_path)
    assert result.schema_kind == "chunked_rows"


def test_invariant_run_result_fingerprint_is_empty_string(tmp_path: Path) -> None:
    """NodeResult.fingerprint is always the empty string (no hash computed)."""
    result = _run([], max_len=4, tmp_path=tmp_path)
    assert result.fingerprint == ""


def test_invariant_run_extras_row_count_matches_len_of_rows(tmp_path: Path) -> None:
    """extras['row_count'] must exactly equal len(result.rows)."""
    rows = [{"input_ids": list(range(6))}]
    result = _run(rows, max_len=4, tmp_path=tmp_path)
    assert result.extras["row_count"] == len(result.rows)  # type: ignore[arg-type]


# ===========================================================================
# ChunkNode class attributes and registry
# ===========================================================================


def test_invariant_chunk_node_kind() -> None:
    """ChunkNode.kind class attribute equals 'chunk'."""
    assert ChunkNode.kind == "chunk"


def test_invariant_chunk_node_schema_kind() -> None:
    """ChunkNode.schema_kind class attribute equals 'chunked_rows'."""
    assert ChunkNode.schema_kind == "chunked_rows"


def test_invariant_chunk_node_registered_in_registry() -> None:
    """ChunkNode is registered under ``prep_node / chunk`` in the global registry."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.chunk  # noqa: F401
    from lighttrain.registry import get_registry

    entry = get_registry().get("prep_node", "chunk")
    assert entry is ChunkNode


def test_invariant_all_exports_chunk_node() -> None:
    """``__all__`` contains 'ChunkNode'."""
    from lighttrain.builtin_plugins.data.prepgraph.nodes.chunk import __all__

    assert "ChunkNode" in __all__


# ===========================================================================
# Parametrized boundary / property tests
# ===========================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 10])
def test_invariant_chunk_one_total_tokens_accounted_for(n: int) -> None:
    """All input tokens appear across chunks exactly once (no overlap)."""
    ids = list(range(n))
    row = {"input_ids": ids}
    chunks = list(_chunk_one(row, max_len=3, overlap=0))
    recovered = [tok for c in chunks for tok in c["input_ids"]]
    assert recovered == ids


@pytest.mark.parametrize(
    "seq_len, max_len, expected_chunks",
    [
        (3, 4, 1),   # short path: 3 < 4
        (4, 4, 1),   # short path: 4 == 4
        (5, 4, 2),   # first chunked case
        (8, 4, 2),   # exactly two full chunks
        (9, 4, 3),   # three chunks, last has 1 token
    ],
)
def test_invariant_chunk_one_chunk_count_no_overlap(
    seq_len: int, max_len: int, expected_chunks: int
) -> None:
    """Verify exact chunk count for several (seq_len, max_len) pairs without overlap."""
    row = {"input_ids": list(range(seq_len))}
    chunks = list(_chunk_one(row, max_len=max_len, overlap=0))
    assert len(chunks) == expected_chunks


@pytest.mark.parametrize("name", ["chunker_a", "step_chunk", "chunk_final"])
def test_invariant_run_error_uses_given_name(name: str, tmp_path: Path) -> None:
    """Both error messages (no inputs, bad max_len) include the caller-given node name."""
    node_no_in = ChunkNode(name=name, inputs=[], config={"max_len": 4})
    ctx = RunContext(store_root=tmp_path, workers=1, upstream={})
    with pytest.raises(ValueError, match=name):
        node_no_in.run(ctx)

    node_bad_len = ChunkNode(name=name, inputs=["up"], config={"max_len": 0})
    with pytest.raises(ValueError, match=name):
        node_bad_len.run(_make_ctx([], tmp_path))
