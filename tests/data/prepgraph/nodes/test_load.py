"""Tests for ``lighttrain.builtin_plugins.data.prepgraph.nodes.load``.

Coverage targets (previously uncovered lines):
  30   — _iter_jsonl: blank/whitespace-only line is skipped
  45   — _iter_parquet: import pyarrow.parquet
  47   — _iter_parquet: read_table
  48   — _iter_parquet: yield rows
  52   — _iter_dir: sorted(path.iterdir())
  53   — _iter_dir: loop over files
  54   — _iter_dir: dispatch .jsonl
  55   — _iter_dir: yield from _iter_jsonl
  56   — _iter_dir: dispatch .parquet
  57   — _iter_dir: yield from _iter_parquet
  61   — _iter_hf: import datasets.load_dataset
  63   — _iter_hf: load_dataset call (with/without subset)
  64   — _iter_hf: loop over dataset
  65   — _iter_hf: yield dict(ex)
  87   — LoadNode.estimate: returns NodeEstimate with note
  92   — LoadNode._iter: missing source raises ValueError
  96   — LoadNode._iter: plain-path fallback to "auto" scheme
  101  — LoadNode._iter: "lines:" scheme dispatches to _iter_lines
  102  — LoadNode._iter: "parquet:" scheme dispatches to _iter_parquet
  103  — LoadNode._iter: "dir:" scheme dispatches to _iter_dir
  104  — LoadNode._iter: "hf:" scheme entry
  105  — LoadNode._iter: hf payload partition on ":"
  106  — LoadNode._iter: hf name/subset split
  107  — LoadNode._iter: _iter_hf call
  112  — LoadNode._iter: "auto" scheme → p.is_dir() branch
  113  — LoadNode._iter: p = Path(payload)
  114  — LoadNode._iter: is_dir() → _iter_dir
  115  — LoadNode._iter: return _iter_dir
  116  — LoadNode._iter: .jsonl suffix branch
  117  — LoadNode._iter: return _iter_jsonl(p)
  118  — LoadNode._iter: .parquet suffix branch
  119  — LoadNode._iter: return _iter_parquet(p)
  120  — LoadNode._iter: fallthrough → unknown scheme ValueError
  126  — LoadNode.run: limit is not None → rows[:int(limit)]
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lighttrain.builtin_plugins.data.prepgraph.nodes.load import (
    LoadNode,
    _iter_dir,
    _iter_jsonl,
    _iter_lines,
    _iter_parquet,
)
from lighttrain.data.prepgraph.node import NodeEstimate, NodeResult, RunContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(store_root=tmp_path, workers=1)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _write_parquet(path: Path, data: dict) -> Path:
    table = pa.table(data)
    pq.write_table(table, str(path))
    return path


# ---------------------------------------------------------------------------
# _iter_jsonl
# ---------------------------------------------------------------------------


def test_invariant_iter_jsonl_yields_all_rows(tmp_path: Path) -> None:
    """All non-blank JSONL lines are parsed and yielded."""
    p = _write_jsonl(tmp_path / "d.jsonl", [{"a": 1}, {"b": 2}, {"c": 3}])
    result = list(_iter_jsonl(p))
    assert result == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_invariant_iter_jsonl_skips_blank_and_whitespace_lines(tmp_path: Path) -> None:
    """Blank and whitespace-only lines (line 30 branch) are skipped without error."""
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"x": 1}) + "\n\n   \n" + json.dumps({"x": 2}) + "\n",
        encoding="utf-8",
    )
    result = list(_iter_jsonl(p))
    assert result == [{"x": 1}, {"x": 2}]


def test_invariant_iter_jsonl_empty_file_yields_nothing(tmp_path: Path) -> None:
    """An empty JSONL file returns an empty iterator."""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert list(_iter_jsonl(p)) == []


# ---------------------------------------------------------------------------
# _iter_lines
# ---------------------------------------------------------------------------


def test_invariant_iter_lines_wraps_each_line_in_text_key(tmp_path: Path) -> None:
    """Each non-empty line becomes ``{"text": stripped_line}``."""
    p = tmp_path / "corpus.txt"
    p.write_text("  hello  \nworld\n\n  \n", encoding="utf-8")
    result = list(_iter_lines(p))
    assert result == [{"text": "hello"}, {"text": "world"}]


def test_invariant_iter_lines_empty_file_yields_nothing(tmp_path: Path) -> None:
    """An empty .txt file returns an empty iterator."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    assert list(_iter_lines(p)) == []


# ---------------------------------------------------------------------------
# _iter_parquet — lines 45, 47, 48
# ---------------------------------------------------------------------------


def test_invariant_iter_parquet_yields_rows(tmp_path: Path) -> None:
    """``_iter_parquet`` reads a Parquet file and yields one dict per row (lines 45-48)."""
    p = tmp_path / "data.parquet"
    _write_parquet(p, {"name": ["alice", "bob"], "score": [10, 20]})
    result = list(_iter_parquet(p))
    assert result == [{"name": "alice", "score": 10}, {"name": "bob", "score": 20}]


def test_invariant_iter_parquet_single_row(tmp_path: Path) -> None:
    """A single-row Parquet file is handled correctly."""
    p = tmp_path / "one.parquet"
    _write_parquet(p, {"val": [42]})
    result = list(_iter_parquet(p))
    assert result == [{"val": 42}]


def test_invariant_iter_parquet_empty_table(tmp_path: Path) -> None:
    """An empty Parquet table yields no rows."""
    pa.schema([("col", pa.int64())])
    table = pa.table({"col": pa.array([], type=pa.int64())})
    p = tmp_path / "empty.parquet"
    pq.write_table(table, str(p))
    assert list(_iter_parquet(p)) == []


# ---------------------------------------------------------------------------
# _iter_dir — lines 52–57
# ---------------------------------------------------------------------------


def test_invariant_iter_dir_dispatches_jsonl_and_parquet(tmp_path: Path) -> None:
    """``_iter_dir`` processes .jsonl and .parquet files sorted by name (lines 52-57)."""
    d = tmp_path / "mixed"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", [{"src": "jsonl"}])
    _write_parquet(d / "b.parquet", {"src": ["parquet"]})
    (d / "z.txt").write_text("should be ignored", encoding="utf-8")
    result = list(_iter_dir(d))
    assert result == [{"src": "jsonl"}, {"src": "parquet"}]


def test_invariant_iter_dir_sorted_order(tmp_path: Path) -> None:
    """Files are processed in alphabetical order (line 52)."""
    d = tmp_path / "ordered"
    d.mkdir()
    _write_jsonl(d / "c.jsonl", [{"order": 3}])
    _write_jsonl(d / "a.jsonl", [{"order": 1}])
    _write_jsonl(d / "b.jsonl", [{"order": 2}])
    result = list(_iter_dir(d))
    assert [r["order"] for r in result] == [1, 2, 3]


def test_invariant_iter_dir_skips_unsupported_extensions(tmp_path: Path) -> None:
    """Files with .txt, .csv etc. are silently skipped."""
    d = tmp_path / "unsupported"
    d.mkdir()
    (d / "readme.txt").write_text("ignore", encoding="utf-8")
    (d / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert list(_iter_dir(d)) == []


def test_invariant_iter_dir_only_parquet(tmp_path: Path) -> None:
    """A directory with only Parquet files works correctly (line 56-57)."""
    d = tmp_path / "parquet_only"
    d.mkdir()
    _write_parquet(d / "data.parquet", {"v": [5, 6]})
    result = list(_iter_dir(d))
    assert result == [{"v": 5}, {"v": 6}]


# ---------------------------------------------------------------------------
# _iter_hf — lines 61, 63, 64, 65 (mocked to avoid network)
# ---------------------------------------------------------------------------


def test_invariant_iter_hf_no_subset_passes_none(tmp_path: Path) -> None:
    """``_iter_hf`` calls ``load_dataset`` without a config when subset is None (line 63).

    ``load_dataset`` is a lazy import inside the function body, so we patch it
    at the ``datasets`` module level.
    """
    fake_rows = [{"text": "hello"}, {"text": "world"}]
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter(fake_rows)
        from lighttrain.builtin_plugins.data.prepgraph.nodes.load import _iter_hf

        result = list(_iter_hf(name="org/repo", split="train", subset=None))
    mock_ld.assert_called_once_with("org/repo", split="train")
    assert result == [{"text": "hello"}, {"text": "world"}]


def test_invariant_iter_hf_with_subset_passes_config(tmp_path: Path) -> None:
    """``_iter_hf`` calls ``load_dataset`` with the subset name when given (line 63).

    ``load_dataset`` is a lazy import inside the function body, so we patch it
    at the ``datasets`` module level.
    """
    fake_rows = [{"q": "what?"}]
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter(fake_rows)
        from lighttrain.builtin_plugins.data.prepgraph.nodes.load import _iter_hf

        result = list(_iter_hf(name="org/repo", split="validation", subset="myconfig"))
    mock_ld.assert_called_once_with("org/repo", "myconfig", split="validation")
    assert result == [{"q": "what?"}]


def test_invariant_iter_hf_converts_examples_to_plain_dicts() -> None:
    """Each element from the dataset is coerced to a plain dict via dict() (line 65).

    ``load_dataset`` is a lazy import inside the function body, so we patch it
    at the ``datasets`` module level.
    """

    class _MockRow:
        """Simulate a HuggingFace dataset row (mapping-like but not dict)."""

        def keys(self):
            return ["a", "b"]

        def __iter__(self):
            return iter(["a", "b"])

        def __getitem__(self, k):
            return {"a": 1, "b": 2}[k]

    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = [_MockRow()]
        from lighttrain.builtin_plugins.data.prepgraph.nodes.load import _iter_hf

        result = list(_iter_hf(name="x/y", split="train", subset=None))

    assert isinstance(result[0], dict)
    assert result[0] == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# LoadNode.estimate — line 87
# ---------------------------------------------------------------------------


def test_invariant_estimate_returns_node_estimate_with_source_note(tmp_path: Path) -> None:
    """``estimate()`` returns a ``NodeEstimate`` whose note contains the source (line 87)."""
    node = LoadNode(name="loader", config={"source": "jsonl:/tmp/x.jsonl"})
    ctx = _ctx(tmp_path)
    est = node.estimate(ctx)
    assert isinstance(est, NodeEstimate)
    assert "jsonl:/tmp/x.jsonl" in (est.note or "")


def test_invariant_estimate_note_when_source_is_none(tmp_path: Path) -> None:
    """``estimate()`` with no source: note contains repr of None."""
    node = LoadNode(name="no_src", config={})
    ctx = _ctx(tmp_path)
    est = node.estimate(ctx)
    assert "None" in (est.note or "")


# ---------------------------------------------------------------------------
# LoadNode._iter — error paths and scheme dispatch
# ---------------------------------------------------------------------------


def test_invariant_iter_raises_for_missing_source() -> None:
    """``_iter()`` raises ``ValueError`` when ``source`` config key is absent (line 92)."""
    node = LoadNode(name="n", config={})
    with pytest.raises(ValueError, match="missing `source`"):
        list(node._iter())


def test_invariant_iter_raises_for_empty_source() -> None:
    """``_iter()`` raises ``ValueError`` when ``source`` is an empty string."""
    node = LoadNode(name="n", config={"source": ""})
    with pytest.raises(ValueError, match="missing `source`"):
        list(node._iter())


def test_invariant_iter_raises_for_unknown_scheme() -> None:
    """``_iter()`` raises ``ValueError`` for an unrecognised scheme (line 120)."""
    node = LoadNode(name="n", config={"source": "bogus:something"})
    with pytest.raises(ValueError, match="unknown source scheme 'bogus'"):
        list(node._iter())


@pytest.mark.parametrize(
    "scheme, suffix",
    [
        ("jsonl", ".jsonl"),
        ("lines", ".txt"),
        ("parquet", ".parquet"),
    ],
)
def test_invariant_iter_explicit_schemes_read_files(
    tmp_path: Path, scheme: str, suffix: str
) -> None:
    """Explicit ``jsonl:``, ``lines:``, and ``parquet:`` schemes load the file (lines 98-102)."""
    if scheme == "parquet":
        p = tmp_path / f"data{suffix}"
        _write_parquet(p, {"n": [7]})
    else:
        p = tmp_path / f"data{suffix}"
        if scheme == "lines":
            p.write_text("row one\nrow two\n", encoding="utf-8")
        else:
            _write_jsonl(p, [{"v": 42}])

    node = LoadNode(name="n", config={"source": f"{scheme}:{p}"})
    rows = list(node._iter())
    assert len(rows) >= 1


def test_invariant_iter_dir_scheme_reads_directory(tmp_path: Path) -> None:
    """The ``dir:`` scheme dispatches to ``_iter_dir`` (line 103-104)."""
    d = tmp_path / "mydir"
    d.mkdir()
    _write_jsonl(d / "f.jsonl", [{"k": "v"}])
    node = LoadNode(name="n", config={"source": f"dir:{d}"})
    assert list(node._iter()) == [{"k": "v"}]


def test_invariant_iter_hf_scheme_dispatches_correctly(tmp_path: Path) -> None:
    """The ``hf:`` scheme parses name/subset and calls ``_iter_hf`` (lines 105-111)."""
    with patch(
        "lighttrain.builtin_plugins.data.prepgraph.nodes.load._iter_hf"
    ) as mock_hf:
        mock_hf.return_value = iter([{"text": "hello"}])
        node = LoadNode(
            name="hf_node",
            config={"source": "hf:org/dataset:myconfig", "split": "validation"},
        )
        rows = list(node._iter())

    mock_hf.assert_called_once_with(
        name="org/dataset", split="validation", subset="myconfig"
    )
    assert rows == [{"text": "hello"}]


def test_invariant_iter_hf_scheme_no_subset(tmp_path: Path) -> None:
    """``hf:<name>`` (no subset) passes ``subset=None`` to ``_iter_hf``."""
    with patch(
        "lighttrain.builtin_plugins.data.prepgraph.nodes.load._iter_hf"
    ) as mock_hf:
        mock_hf.return_value = iter([])
        node = LoadNode(name="hf", config={"source": "hf:org/repo"})
        list(node._iter())

    mock_hf.assert_called_once_with(name="org/repo", split="train", subset=None)


def test_invariant_iter_hf_default_split_is_train() -> None:
    """When ``split`` is not in config, ``_iter_hf`` receives split='train'."""
    with patch(
        "lighttrain.builtin_plugins.data.prepgraph.nodes.load._iter_hf"
    ) as mock_hf:
        mock_hf.return_value = iter([])
        node = LoadNode(name="hf", config={"source": "hf:x/y"})
        list(node._iter())
    _, kwargs = mock_hf.call_args
    assert kwargs["split"] == "train"


# ---------------------------------------------------------------------------
# LoadNode._iter — auto scheme (lines 112-119)
# ---------------------------------------------------------------------------


def test_invariant_iter_auto_scheme_detects_jsonl(tmp_path: Path) -> None:
    """A plain path with ``.jsonl`` extension is picked up by the auto scheme (line 116-117)."""
    p = _write_jsonl(tmp_path / "corpus.jsonl", [{"auto": True}])
    node = LoadNode(name="n", config={"source": str(p)})
    assert list(node._iter()) == [{"auto": True}]


def test_invariant_iter_auto_scheme_detects_parquet(tmp_path: Path) -> None:
    """A plain path with ``.parquet`` extension is picked up by the auto scheme (line 118-119)."""
    p = tmp_path / "corpus.parquet"
    _write_parquet(p, {"x": [99]})
    node = LoadNode(name="n", config={"source": str(p)})
    assert list(node._iter()) == [{"x": 99}]


def test_invariant_iter_auto_scheme_detects_directory(tmp_path: Path) -> None:
    """A plain directory path triggers the auto→dir branch (lines 113-115)."""
    d = tmp_path / "autodir"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", [{"from": "auto_dir"}])
    node = LoadNode(name="n", config={"source": str(d)})
    assert list(node._iter()) == [{"from": "auto_dir"}]


def test_invariant_iter_auto_scheme_unknown_extension_raises(tmp_path: Path) -> None:
    """A plain path with an unsupported extension falls through to ValueError (line 120)."""
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    node = LoadNode(name="n", config={"source": str(p)})
    with pytest.raises(ValueError, match="unknown source scheme 'auto'"):
        list(node._iter())


# ---------------------------------------------------------------------------
# LoadNode.run — limit behaviour (lines 122-132)
# ---------------------------------------------------------------------------


def test_invariant_run_returns_node_result(tmp_path: Path) -> None:
    """``run()`` returns a ``NodeResult`` with rows and extras."""
    p = _write_jsonl(tmp_path / "d.jsonl", [{"i": 0}, {"i": 1}])
    node = LoadNode(name="n", config={"source": f"jsonl:{p}"})
    result = node.run(_ctx(tmp_path))
    assert isinstance(result, NodeResult)
    assert result.rows == [{"i": 0}, {"i": 1}]
    assert result.extras["row_count"] == 2
    assert result.schema_kind == "rows"
    assert result.fingerprint == ""


def test_invariant_run_without_limit_returns_all_rows(tmp_path: Path) -> None:
    """``run()`` without a ``limit`` config key returns every row."""
    rows = [{"n": i} for i in range(10)]
    p = _write_jsonl(tmp_path / "d.jsonl", rows)
    node = LoadNode(name="n", config={"source": f"jsonl:{p}"})
    result = node.run(_ctx(tmp_path))
    assert len(result.rows) == 10  # type: ignore[arg-type]
    assert result.extras["row_count"] == 10


def test_invariant_run_with_limit_truncates_rows(tmp_path: Path) -> None:
    """``run()`` with ``limit`` clips the output (line 126)."""
    rows = [{"n": i} for i in range(10)]
    p = _write_jsonl(tmp_path / "d.jsonl", rows)
    node = LoadNode(name="n", config={"source": f"jsonl:{p}", "limit": 4})
    result = node.run(_ctx(tmp_path))
    assert result.rows == [{"n": i} for i in range(4)]
    assert result.extras["row_count"] == 4


def test_invariant_run_limit_zero_returns_empty(tmp_path: Path) -> None:
    """``limit=0`` produces an empty row list."""
    p = _write_jsonl(tmp_path / "d.jsonl", [{"n": 1}, {"n": 2}])
    node = LoadNode(name="n", config={"source": f"jsonl:{p}", "limit": 0})
    result = node.run(_ctx(tmp_path))
    assert result.rows == []
    assert result.extras["row_count"] == 0


def test_invariant_run_limit_larger_than_file_returns_all(tmp_path: Path) -> None:
    """``limit`` larger than the file size returns all rows without error."""
    p = _write_jsonl(tmp_path / "d.jsonl", [{"n": i} for i in range(3)])
    node = LoadNode(name="n", config={"source": f"jsonl:{p}", "limit": 9999})
    result = node.run(_ctx(tmp_path))
    assert len(result.rows) == 3  # type: ignore[arg-type]


def test_invariant_run_limit_as_string_is_coerced(tmp_path: Path) -> None:
    """``limit`` supplied as a string is coerced via ``int()`` (line 126)."""
    p = _write_jsonl(tmp_path / "d.jsonl", [{"n": i} for i in range(5)])
    node = LoadNode(name="n", config={"source": f"jsonl:{p}", "limit": "2"})
    result = node.run(_ctx(tmp_path))
    assert len(result.rows) == 2  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Class-level attributes and registration
# ---------------------------------------------------------------------------


def test_invariant_load_node_kind_and_schema_kind() -> None:
    """``LoadNode`` class attributes ``kind`` and ``schema_kind`` are correct."""
    assert LoadNode.kind == "load"
    assert LoadNode.schema_kind == "rows"


def test_invariant_load_node_registered_in_registry() -> None:
    """``LoadNode`` is registered under ``prep_node / load`` in the global registry."""
    # Importing the module triggers @register side-effect
    import lighttrain.builtin_plugins.data.prepgraph.nodes.load  # noqa: F401
    from lighttrain.registry import get_registry

    entry = get_registry().get("prep_node", "load")
    assert entry is LoadNode


def test_invariant_all_exports_loadnode() -> None:
    """``__all__`` contains ``LoadNode``."""
    from lighttrain.builtin_plugins.data.prepgraph.nodes.load import __all__

    assert "LoadNode" in __all__


# ---------------------------------------------------------------------------
# Parametrize error messages include node name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["myloader", "step1_raw"])
def test_invariant_error_messages_include_node_name(name: str) -> None:
    """ValueError messages contain the node name for easier diagnosis."""
    node = LoadNode(name=name, config={})
    with pytest.raises(ValueError, match=name):
        list(node._iter())


def test_invariant_unknown_scheme_error_includes_scheme() -> None:
    """The unknown-scheme error message names the offending scheme."""
    node = LoadNode(name="n", config={"source": "ftp:something"})
    with pytest.raises(ValueError, match="'ftp'"):
        list(node._iter())
