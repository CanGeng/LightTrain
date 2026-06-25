"""Edge-case tests for ``lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize``.

Coverage targets (previously uncovered lines):

* L37  — ``_label_with_mask``: mismatched lengths → labels replaced by list(ids)
* L80  — ``TokenizeNode._ensure``: both processor+tokenizer raises ValueError
* L89  — ``TokenizeNode._ensure``: neither processor nor tokenizer raises ValueError
* L105 — ``_iter_tokenized`` (processor path): missing turns field → synthetic turn
* L108 — ``_iter_tokenized`` (processor path): response_only_mask forwarded to proc
* L125 — ``_iter_tokenized`` (processor path): modality_inputs carried through
* L141 — ``TokenizeNode.run``: no inputs raises ValueError
* L145 — ``TokenizeNode.run``: upstream rows=None raises RuntimeError

Also covers general happy-paths for both the tokenizer and processor flavours,
plus invariants for ``NodeResult`` shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Ensure the built-in prep_node kinds (including "tokenize") are registered.
from lighttrain.builtin_plugins.data.prepgraph import nodes as _nodes_pkg  # noqa: F401
from lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize import (
    TokenizeNode,
    _label_with_mask,
)
from lighttrain.data.prepgraph.node import NodeResult, RunContext

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer stub: encode returns one int per character."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class _FakeProc:
    """Minimal processor stub.

    Returns a configurable output dict so we can exercise all branches.
    """

    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self._output = output or {}
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, turns: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((turns, kwargs))
        return dict(self._output)


def _make_ctx(rows: list[dict[str, Any]], tmp_path: Path, upstream_name: str = "up") -> RunContext:
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {
        upstream_name: NodeResult(fingerprint="fp", schema_kind="rows", rows=list(rows))
    }
    return ctx


def _make_ctx_no_rows(tmp_path: Path, upstream_name: str = "up") -> RunContext:
    """Context where the upstream NodeResult has rows=None."""
    ctx = RunContext(store_root=tmp_path / "store")
    ctx.store_root.mkdir(parents=True, exist_ok=True)
    ctx.upstream = {
        upstream_name: NodeResult(fingerprint="fp", schema_kind="rows", rows=None)
    }
    return ctx


# ---------------------------------------------------------------------------
# _label_with_mask  (L33-38)
# ---------------------------------------------------------------------------


def test_invariant_label_with_mask_same_length_returns_labels_unchanged() -> None:
    """When len(labels) == len(ids), ``_label_with_mask`` returns ``labels`` as-is."""
    ids = [1, 2, 3]
    labels = [-100, 2, 3]
    result = _label_with_mask(ids, labels, label_ignore=-100)
    assert result == labels


def test_invariant_label_with_mask_mismatched_length_returns_list_of_ids() -> None:
    """L37: When len(labels) != len(ids), labels is replaced with ``list(ids)``.

    This covers the uncovered branch on line 37.
    """
    ids = [10, 20, 30]
    labels = [-100]  # shorter — triggers the branch
    result = _label_with_mask(ids, labels, label_ignore=-100)
    assert result == list(ids)


def test_invariant_label_with_mask_longer_labels_also_replaced() -> None:
    """L37: labels longer than ids also triggers replacement."""
    ids = [1]
    labels = [-100, -100, -100]
    result = _label_with_mask(ids, labels, label_ignore=-100)
    assert result == list(ids)


# ---------------------------------------------------------------------------
# TokenizeNode._ensure — error paths  (L80, L89)
# ---------------------------------------------------------------------------


def test_invariant_ensure_raises_when_both_proc_and_tok_specified(monkeypatch: pytest.MonkeyPatch) -> None:
    """L80: specifying both ``processor`` and ``tokenizer`` raises ValueError."""
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={
            "processor": {"name": "dummy_proc"},
            "tokenizer": {"name": "dummy_tok"},
        },
    )
    # Patch _build_processor / _build_tokenizer so _resolve is never called.
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: _FakeProc())
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    with pytest.raises(ValueError, match="exactly one of"):
        node._ensure()


def test_invariant_ensure_raises_when_neither_proc_nor_tok_specified() -> None:
    """L89: specifying neither ``processor`` nor ``tokenizer`` raises ValueError."""
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={},
    )
    with pytest.raises(ValueError, match="must declare"):
        node._ensure()


# ---------------------------------------------------------------------------
# TokenizeNode.run — error paths  (L141, L145)
# ---------------------------------------------------------------------------


def test_invariant_run_raises_when_inputs_empty(tmp_path: Path) -> None:
    """L141: ``run()`` with no inputs raises ValueError."""
    node = TokenizeNode(
        name="tok",
        inputs=[],
        config={"tokenizer": {"name": "dummy"}},
    )
    ctx = _make_ctx([], tmp_path)
    with pytest.raises(ValueError, match="requires upstream input"):
        node.run(ctx)


def test_invariant_run_raises_when_upstream_rows_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L145: ``run()`` raises RuntimeError when upstream NodeResult has rows=None."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    ctx = _make_ctx_no_rows(tmp_path)
    with pytest.raises(RuntimeError, match="produced no rows in memory"):
        node.run(ctx)


# ---------------------------------------------------------------------------
# Tokenizer (plain) path — happy path
# ---------------------------------------------------------------------------


def test_invariant_tokenizer_path_produces_input_ids_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain tokenizer path: ``input_ids``, ``labels``, ``attention_mask``, ``modality`` all set."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    tok = _FakeTok()
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: tok)

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    rows = [{"text": "hello world"}, {"text": "foo"}]
    ctx = _make_ctx(rows, tmp_path)
    result = node.run(ctx)

    assert result.schema_kind == "tokenized_rows"
    assert result.extras["row_count"] == 2
    out = list(result.rows)
    assert len(out) == 2

    r0 = out[0]
    expected_ids = tok.encode("hello world")
    assert r0["input_ids"] == expected_ids
    assert r0["labels"] == expected_ids
    assert r0["attention_mask"] == [1] * len(expected_ids)
    assert r0["modality"] == "text"


def test_invariant_tokenizer_path_custom_text_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``text_field`` config key controls which row key is read."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    tok = _FakeTok()
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: tok)

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}, "text_field": "raw"},
    )
    rows = [{"raw": "a b c", "text": "should be ignored"}]
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["input_ids"] == tok.encode("a b c")


def test_invariant_tokenizer_path_empty_text_becomes_empty_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row with missing text_field yields empty token lists without error."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    result = node.run(_make_ctx([{"other_field": "x"}], tmp_path))
    out = list(result.rows)
    # encode("") → []
    assert out[0]["input_ids"] == []
    assert out[0]["labels"] == []
    assert out[0]["attention_mask"] == []


def test_invariant_tokenizer_path_preserves_extra_row_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extra row fields (e.g. ``id``) survive into the output row."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod
    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    rows = [{"text": "hi", "id": "sample-42", "meta": {"split": "train"}}]
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["id"] == "sample-42"
    assert out[0]["meta"] == {"split": "train"}


# ---------------------------------------------------------------------------
# Processor path — happy path
# ---------------------------------------------------------------------------


def test_invariant_processor_path_uses_messages_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processor path: messages from ``turns_field`` are passed to the processor."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1, 2, 3], "labels": [1, 2, 3]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    turns = [{"role": "user", "content": "hi"}]
    rows = [{"messages": turns}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["input_ids"] == [1, 2, 3]
    assert len(proc.calls) == 1
    assert proc.calls[0][0] == turns


def test_invariant_processor_path_fallback_to_turns_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processor path: ``turns`` key is tried when ``messages`` is absent."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [5, 6]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    turns = [{"role": "assistant", "content": "ok"}]
    rows = [{"turns": turns}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["input_ids"] == [5, 6]
    assert proc.calls[0][0] == turns


def test_invariant_processor_path_fallback_to_conversations_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processor path: ``conversations`` key is tried last before synthetic turn."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [7]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    convs = [{"role": "user", "content": "yes"}]
    rows = [{"conversations": convs}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["input_ids"] == [7]
    assert proc.calls[0][0] == convs


def test_invariant_processor_path_synthetic_turn_when_no_turns_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L105: when all turns-field lookups fail, a synthetic user-turn is created from text_field."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    captured: list[Any] = []

    class _CapProc:
        def __call__(self, turns: Any, **kwargs: Any) -> dict[str, Any]:
            captured.append(turns)
            return {"input_ids": [99], "labels": [99]}

    monkeypatch.setattr(_mod, "_build_processor", lambda spec: _CapProc())

    rows = [{"text": "synthetic content"}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    list(result.rows)  # consume to trigger processing
    assert len(captured) == 1
    synthetic = captured[0]
    assert isinstance(synthetic, list)
    assert len(synthetic) == 1
    assert synthetic[0]["role"] == "user"
    assert synthetic[0]["content"] == "synthetic content"


def test_invariant_processor_path_response_only_mask_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L108: ``response_only_mask`` from config is passed as kwarg to the processor."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1, 2], "labels": [1, 2]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "q"}]}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}, "response_only_mask": True},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    list(result.rows)
    assert len(proc.calls) == 1
    _, kwargs = proc.calls[0]
    assert kwargs.get("response_only_mask") is True


def test_invariant_processor_path_response_only_mask_false_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L108: ``response_only_mask=False`` is also forwarded (falsy value must not be dropped)."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "q"}]}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}, "response_only_mask": False},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    list(result.rows)
    _, kwargs = proc.calls[0]
    assert "response_only_mask" in kwargs
    assert kwargs["response_only_mask"] is False


def test_invariant_processor_path_no_response_only_mask_no_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processor is called without ``response_only_mask`` when not in config."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "q"}]}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    list(result.rows)
    _, kwargs = proc.calls[0]
    assert "response_only_mask" not in kwargs


def test_invariant_processor_path_modality_inputs_carried_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L125: when processor output includes ``modality_inputs``, it is copied into the row."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    pixel_values = [[1.0, 2.0], [3.0, 4.0]]
    proc = _FakeProc(
        output={
            "input_ids": [10, 20],
            "labels": [10, 20],
            "modality": "image",
            "modality_inputs": {"pixel_values": pixel_values},
        }
    )
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "what is this?"}]}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert "modality_inputs" in out[0]
    assert out[0]["modality_inputs"]["pixel_values"] == pixel_values
    assert out[0]["modality"] == "image"


def test_invariant_processor_path_no_modality_inputs_key_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When processor output has no ``modality_inputs``, the key must not appear in the row."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1], "labels": [1]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert "modality_inputs" not in out[0]


def test_invariant_processor_path_messages_field_removed_from_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processor path removes the turns_field key from the output row."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": [{"role": "user", "content": "q"}], "id": "s0"}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert "messages" not in out[0]
    assert out[0]["id"] == "s0"  # other fields preserved


def test_invariant_processor_path_custom_turns_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``turns_field`` config key controls which row key is read for turns."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [3, 4]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    turns = [{"role": "user", "content": "custom"}]
    rows = [{"chat": turns}]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}, "turns_field": "chat"},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    list(result.rows)
    assert proc.calls[0][0] == turns


# ---------------------------------------------------------------------------
# Processor path — attention_mask defaults + label_ignore fallback
# ---------------------------------------------------------------------------


def test_invariant_processor_path_attention_mask_defaults_to_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When processor output omits ``attention_mask``, defaults to all-ones of len(ids)."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [1, 2, 3]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": []}]  # type: ignore[var-annotated]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["attention_mask"] == [1, 1, 1]


def test_invariant_processor_path_labels_default_to_input_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When processor output omits ``labels``, they default to a copy of ``input_ids``."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    proc = _FakeProc(output={"input_ids": [10, 20]})
    monkeypatch.setattr(_mod, "_build_processor", lambda spec: proc)

    rows = [{"messages": []}]  # type: ignore[var-annotated]
    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"processor": {"name": "chat_template"}},
    )
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert out[0]["labels"] == [10, 20]


# ---------------------------------------------------------------------------
# _ensure caching: called only once
# ---------------------------------------------------------------------------


def test_invariant_ensure_called_only_once_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_ensure`` builds the tokenizer at most once even across multiple rows."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    build_calls: list[int] = []

    def _counting_build(spec: Any) -> _FakeTok:
        build_calls.append(1)
        return _FakeTok()

    monkeypatch.setattr(_mod, "_build_tokenizer", _counting_build)

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    node.run(_make_ctx(rows, tmp_path))
    assert sum(build_calls) == 1


# ---------------------------------------------------------------------------
# NodeResult contract
# ---------------------------------------------------------------------------


def test_invariant_run_result_extras_row_count_matches_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``result.extras['row_count']`` must match the number of output rows."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    rows = [{"text": "a"}, {"text": "b"}]
    result = node.run(_make_ctx(rows, tmp_path))
    out = list(result.rows)
    assert result.extras["row_count"] == len(out) == 2


def test_invariant_run_result_schema_kind() -> None:
    """``TokenizeNode.schema_kind`` is ``'tokenized_rows'``."""
    node = TokenizeNode(name="tok", inputs=["up"], config={})
    assert node.schema_kind == "tokenized_rows"


def test_invariant_run_result_fingerprint_is_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``NodeResult.fingerprint`` from ``run()`` is the empty string (placeholder)."""
    import lighttrain.builtin_plugins.data.prepgraph.nodes.tokenize as _mod

    monkeypatch.setattr(_mod, "_build_tokenizer", lambda spec: _FakeTok())

    node = TokenizeNode(
        name="tok",
        inputs=["up"],
        config={"tokenizer": {"name": "dummy"}},
    )
    result = node.run(_make_ctx([{"text": "hi"}], tmp_path))
    assert result.fingerprint == ""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_invariant_tokenize_node_is_registered() -> None:
    """``TokenizeNode`` must be registered under ``prep_node/tokenize``."""
    from lighttrain.registry import get as _reg_get
    cls = _reg_get("prep_node", "tokenize")
    assert cls is TokenizeNode
