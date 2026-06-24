"""Fingerprint stability + sensitivity tests (DESIGN §7.7.4)."""

from __future__ import annotations

from lighttrain.data.prepgraph._fp import (
    SCHEMA_VERSION,
    canonical_config,
    code_version_for,
    compose_fingerprint,
)


def test_canonical_config_order_invariant():
    a = canonical_config({"x": 1, "y": [3, 1, 2], "z": {"b": 2, "a": 1}})
    b = canonical_config({"y": [3, 1, 2], "z": {"a": 1, "b": 2}, "x": 1})
    assert a == b


def test_canonical_config_quantizes_floats():
    a = canonical_config({"lr": 1e-9})
    b = canonical_config({"lr": 1e-9 + 1e-15})  # below quantization granularity
    assert a == b


def test_compose_fingerprint_deterministic():
    fp1 = compose_fingerprint(
        kind="tokenize",
        schema_kind="tokenized_rows",
        code_version="abc",
        config={"max_len": 256},
        input_fps=["upstream-1"],
    )
    fp2 = compose_fingerprint(
        kind="tokenize",
        schema_kind="tokenized_rows",
        code_version="abc",
        config={"max_len": 256},
        input_fps=["upstream-1"],
    )
    assert fp1 == fp2
    assert len(fp1) == 64


def test_compose_fingerprint_sensitive_to_code_version():
    fp1 = compose_fingerprint(
        kind="tokenize",
        schema_kind="tokenized_rows",
        code_version="abc",
        config={},
        input_fps=[],
    )
    fp2 = compose_fingerprint(
        kind="tokenize",
        schema_kind="tokenized_rows",
        code_version="xyz",
        config={},
        input_fps=[],
    )
    assert fp1 != fp2


def test_compose_fingerprint_sensitive_to_upstream():
    base = dict(
        kind="tokenize",
        schema_kind="tokenized_rows",
        code_version="abc",
        config={},
    )
    fp1 = compose_fingerprint(**base, input_fps=["a", "b"])
    fp2 = compose_fingerprint(**base, input_fps=["a", "c"])
    assert fp1 != fp2


def test_compose_fingerprint_input_fp_order_irrelevant():
    """Upstream order shouldn't matter — only the set + sorted form."""
    base = dict(
        kind="mix",
        schema_kind="mixed_rows",
        code_version="abc",
        config={},
    )
    fp1 = compose_fingerprint(**base, input_fps=["a", "b"])
    fp2 = compose_fingerprint(**base, input_fps=["b", "a"])
    assert fp1 == fp2


def test_code_version_for_class_stable():
    class FakeNode:
        kind = "demo"

        def run(self, ctx):
            return None

    a = code_version_for(FakeNode)
    b = code_version_for(FakeNode)
    assert a == b
    assert isinstance(a, str)
    assert len(a) > 0


def test_schema_version_table_complete():
    """All schemas referenced by leaf nodes must appear in SCHEMA_VERSION."""
    expected = {
        "rows",
        "tokenized_rows",
        "packed_rows",
        "validate_report",
        "materialized",
        "mixed_rows",
        "chunked_rows",
    }
    for k in expected:
        assert k in SCHEMA_VERSION, f"missing schema_version for {k!r}"
