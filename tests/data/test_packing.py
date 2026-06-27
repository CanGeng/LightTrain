"""Edge-case tests for ``lighttrain.data.packing.SequencePacker``.

The raw ``SequencePacker.pack`` was previously only exercised indirectly via the
``PackNode`` wrapper (tests/data/prepgraph/test_pack.py). This pins the packer's
own contract directly:

* **__post_init__**: ``seq_len <= 0`` rejected; ``eos_id``/``pad_id`` int-coerced.
* **Row filtering**: empty / missing ``input_ids`` rows skipped.
* **EOS glue**: appended iff the doc doesn't already end in ``eos_id``.
* **Greedy flush**: overflow flushes the buffer; exact-fill flushes with no pad.
* **emit()**: padding (``pad_id`` / ``position_ids`` 0 / ``document_ids`` -1),
  per-doc position restart, per-doc ``document_ids``.
* **keep_short**: trailing partial buffer kept (True) or dropped (False).

One ``test_pin_current_behavior_*`` documents a debatable truncation choice
(over-long doc loses its appended EOS) — current behavior, NOT a design
contract. See SUSPECTED-BUGS note on that test.
"""

from __future__ import annotations

import pytest

from lighttrain.data.packing import SequencePacker


def _pack(rows, **kw) -> list[dict]:
    """Materialize the packer's lazy output for assertions."""
    return list(SequencePacker(**kw).pack(rows))


# ---------------------------------------------------------------------------
# __post_init__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -10])
def test_invariant_post_init_rejects_nonpositive_seq_len(bad):
    """``seq_len <= 0`` raises ValueError at construction."""
    with pytest.raises(ValueError, match="seq_len must be positive"):
        SequencePacker(seq_len=bad, eos_id=9)


def test_invariant_eos_and_pad_coerced_to_int():
    """Float ``eos_id`` / ``pad_id`` are coerced to int in __post_init__."""
    p = SequencePacker(seq_len=4, eos_id=2.0, pad_id=1.0)  # type: ignore[arg-type]
    assert p.eos_id == 2 and isinstance(p.eos_id, int)
    assert p.pad_id == 1 and isinstance(p.pad_id, int)


# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------

def test_invariant_empty_rows_iterable_yields_nothing():
    """No rows → no output (loop body never runs, final buffer empty)."""
    assert _pack([], seq_len=4, eos_id=9) == []


def test_invariant_blank_or_missing_input_ids_skipped():
    """Rows with empty / None / missing ``input_ids`` are skipped entirely."""
    rows = [{"input_ids": []}, {}, {"input_ids": None}]
    assert _pack(rows, seq_len=4, eos_id=9) == []


# ---------------------------------------------------------------------------
# EOS glue
# ---------------------------------------------------------------------------

def test_invariant_eos_appended_when_doc_does_not_end_in_eos():
    """A doc not ending in ``eos_id`` gets one appended before packing."""
    out = _pack([{"input_ids": [1, 2, 3]}], seq_len=8, eos_id=9)
    assert len(out) == 1
    assert out[0]["input_ids"][:4] == [1, 2, 3, 9]


def test_invariant_eos_not_duplicated_when_doc_already_ends_in_eos():
    """A doc already ending in ``eos_id`` is left as-is (no second EOS)."""
    out = _pack([{"input_ids": [1, 2, 9]}], seq_len=8, eos_id=9)
    assert out[0]["input_ids"][:3] == [1, 2, 9]
    assert out[0]["input_ids"].count(9) == 1


# ---------------------------------------------------------------------------
# emit(): padding, position restart, document ids  (full closed-form pin)
# ---------------------------------------------------------------------------

def test_invariant_two_short_docs_glue_into_one_window_with_padding():
    """Two short docs glue into one window; the trailing gap is padded.

    Pins all three emit() vectors at once: ``input_ids`` padded with ``pad_id``,
    ``position_ids`` restarts at 0 per doc then 0-pads, ``document_ids`` numbers
    docs 0,1 then -1 for pad.
    """
    rows = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
    out = _pack(rows, seq_len=8, eos_id=9, pad_id=0)
    assert len(out) == 1
    assert out[0]["input_ids"] == [1, 2, 9, 3, 4, 9, 0, 0]
    assert out[0]["position_ids"] == [0, 1, 2, 0, 1, 2, 0, 0]
    assert out[0]["document_ids"] == [0, 0, 0, 1, 1, 1, -1, -1]


def test_invariant_custom_pad_id_used_for_filler():
    """``pad_id`` (not 0) fills ``input_ids`` padding positions."""
    out = _pack([{"input_ids": [1, 2]}], seq_len=5, eos_id=9, pad_id=7)
    assert out[0]["input_ids"] == [1, 2, 9, 7, 7]
    # position/document padding markers are fixed regardless of pad_id.
    assert out[0]["position_ids"] == [0, 1, 2, 0, 0]
    assert out[0]["document_ids"] == [0, 0, 0, -1, -1]


def test_invariant_exact_fill_window_has_no_padding():
    """A doc that exactly fills ``seq_len`` flushes with pad==0 (no pad branch)."""
    out = _pack([{"input_ids": [1, 2, 3]}], seq_len=4, eos_id=9)
    assert out == [
        {"input_ids": [1, 2, 3, 9], "position_ids": [0, 1, 2, 3], "document_ids": [0, 0, 0, 0]}
    ]


# ---------------------------------------------------------------------------
# Greedy flush boundaries
# ---------------------------------------------------------------------------

def test_invariant_overflow_flushes_existing_buffer_then_starts_fresh():
    """When the next doc would overflow, the non-empty buffer flushes first
    and the new doc opens the next window."""
    rows = [{"input_ids": [1, 2]}, {"input_ids": [3, 4, 5]}]
    out = _pack(rows, seq_len=5, eos_id=9)
    assert len(out) == 2
    # window 1 = doc A ([1,2,9]) padded; window 2 = doc B ([3,4,5,9]) padded.
    assert out[0]["input_ids"] == [1, 2, 9, 0, 0]
    assert out[1]["input_ids"] == [3, 4, 5, 9, 0]


def test_invariant_buffer_reaching_seq_len_flushes_immediately():
    """Reaching exactly ``seq_len`` after extend flushes within the loop."""
    rows = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
    out = _pack(rows, seq_len=4, eos_id=9)
    # each 3-token doc + EOS == 4 == seq_len → one window per doc.
    assert [r["input_ids"] for r in out] == [[1, 2, 3, 9], [4, 5, 6, 9]]


# ---------------------------------------------------------------------------
# keep_short
# ---------------------------------------------------------------------------

def test_invariant_keep_short_false_discards_trailing_partial():
    """``keep_short=False`` drops a final under-full buffer."""
    out = _pack([{"input_ids": [1, 2]}], seq_len=8, eos_id=9, keep_short=False)
    assert out == []


def test_invariant_keep_short_true_emits_trailing_partial():
    """``keep_short=True`` (default) keeps the final under-full buffer (padded)."""
    out = _pack([{"input_ids": [1, 2]}], seq_len=8, eos_id=9, keep_short=True)
    assert len(out) == 1
    assert out[0]["input_ids"][:3] == [1, 2, 9]


# ---------------------------------------------------------------------------
# SUSPECTED-BUG pin (debatable design choice — surface, do not "fix")
# ---------------------------------------------------------------------------

def test_pin_current_behavior_oversized_doc_truncation_drops_trailing_eos():
    """An over-long doc is head-truncated to ``seq_len`` AFTER appending EOS,
    so the appended EOS at the tail is dropped — the packed window has no EOS
    terminator.

    SUSPECTED-BUG (design choice, surfaced for decision): truncation semantics
    for docs longer than ``seq_len`` are debatable — one could truncate to
    ``seq_len - 1`` and keep the EOS. This pins TODAY's behavior, it is NOT a
    contract. See lighttrain/data/packing/_packer.py:64,71.
    """
    out = _pack([{"input_ids": [1, 2, 3, 4]}], seq_len=3, eos_id=9)
    assert len(out) == 1
    assert out[0]["input_ids"] == [1, 2, 3]
    assert 9 not in out[0]["input_ids"]  # the appended EOS was truncated away
    assert out[0]["document_ids"] == [0, 0, 0]
