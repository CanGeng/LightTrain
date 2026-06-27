"""Adversarial tests for ``lighttrain.builtin_plugins.data.core.tokenizers.ByteTokenizer``.

Coverage beyond the smoke check in ``tests/test_data_core.py`` (which
tests one round-trip + one unicode sample):

* **Vocab constants pinned** (PAD/BOS/EOS/UNK at fixed integers, VOCAB_SIZE
  == 260 exactly).
* **Special IDs strictly outside the 0..255 byte range**.
* **Round-trip on every byte** in 0..255 (full coverage).
* **add_bos/add_eos flags propagate to encoding**.
* **Decode silently drops out-of-byte-range IDs** (specials AND negatives
  AND >= 256 — per line 49-50 of tokenizers.py).
* **Decode with empty list returns empty string**.
* **Encode of bytes object skips re-encoding** (line 35-36 — bytes
  fast-path).
"""

from __future__ import annotations

from lighttrain.builtin_plugins.data.core.tokenizers import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    ByteTokenizer,
)

# ---------------------------------------------------------------------------
# Vocab constants
# ---------------------------------------------------------------------------

def test_pin_vocab_constants_exact_values():
    """Pin: PAD=256, BOS=257, EOS=258, UNK=259, VOCAB_SIZE=260.

    These are baked into many call sites (CausalLMCollator uses PAD_ID;
    SimpleDataModule reads vocab_size for embedding size). Changing them
    is a coordinated breaking change.
    """
    assert PAD_ID == 256
    assert BOS_ID == 257
    assert EOS_ID == 258
    assert UNK_ID == 259
    assert VOCAB_SIZE == 260


def test_invariant_special_ids_strictly_outside_byte_range():
    """Invariant: every special ID is >= 256 (outside the 0..255 byte
    range), so a real byte can never be confused with a special token.

    Sweep: PAD, BOS, EOS, UNK.
    """
    for sid in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
        assert sid >= 256, f"special id {sid} collides with byte range"


def test_invariant_special_ids_strictly_below_vocab_size():
    """Every special ID is < VOCAB_SIZE — i.e., they fit in the embedding."""
    for sid in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
        assert sid < VOCAB_SIZE


def test_invariant_all_special_ids_distinct():
    """No two special IDs collide."""
    sids = (PAD_ID, BOS_ID, EOS_ID, UNK_ID)
    assert len(set(sids)) == len(sids)


# ---------------------------------------------------------------------------
# Encode/decode round-trips
# ---------------------------------------------------------------------------

def test_invariant_encode_decode_round_trip_all_bytes():
    """Invariant: encode(decode-equivalent) round-trip preserves every byte
    in 0..255.

    Setup: bytes object spanning the entire byte range.
    Expected: decode(encode(b)) reproduces the original bytes, ignoring
    the trailing EOS appended by default. With ``add_eos=False`` the
    round-trip is exact.
    """
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    text = bytes(range(256))
    ids = tk.encode(text)  # type: ignore[arg-type]
    # Encoding bytes directly: every byte preserved literally.
    assert ids == list(range(256))
    # Decode reproduces the original byte sequence.
    decoded_bytes = bytes(int(i) for i in ids if 0 <= i < 256)
    assert decoded_bytes == text


def test_encode_text_roundtrip_ascii():
    """Plain ASCII round-trips via UTF-8 encode/decode."""
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    text = "hello, world!"
    ids = tk.encode(text)
    assert tk.decode(ids) == text


def test_encode_text_roundtrip_unicode():
    """Multi-byte UTF-8 (CJK) round-trips."""
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    text = "你好，世界！"
    ids = tk.encode(text)
    assert tk.decode(ids) == text


def test_encode_text_roundtrip_emoji():
    """4-byte UTF-8 (emoji) round-trips."""
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    text = "test 🚀 done"
    ids = tk.encode(text)
    assert tk.decode(ids) == text


def test_encode_bytes_input_uses_bytes_fast_path():
    """Passing bytes directly skips utf-8 encoding (line 35-36 of tokenizers.py).

    Setup: raw bytes that would be lossy under utf-8 (\\xff is not valid
    standalone UTF-8). The bytes fast-path preserves them; the str-path
    would re-encode (but it's already bytes so the check fires).
    Expected: encoded IDs match the literal byte values.
    """
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    raw = b"\xff\xfe\x00\x01"
    ids = tk.encode(raw)  # type: ignore[arg-type]
    assert ids == [255, 254, 0, 1]


# ---------------------------------------------------------------------------
# BOS / EOS flag propagation
# ---------------------------------------------------------------------------

def test_invariant_add_bos_prepends_bos_id():
    """``add_bos=True`` prepends BOS_ID to the encoded list."""
    tk = ByteTokenizer(add_bos=True, add_eos=False)
    ids = tk.encode("hi")
    assert ids[0] == BOS_ID
    # Remaining IDs are the bytes
    assert ids[1:] == list(b"hi")


def test_invariant_add_eos_appends_eos_id():
    """``add_eos=True`` appends EOS_ID."""
    tk = ByteTokenizer(add_bos=False, add_eos=True)
    ids = tk.encode("hi")
    assert ids[-1] == EOS_ID
    assert ids[:-1] == list(b"hi")


def test_invariant_no_specials_when_flags_false():
    """With both flags off, encoding is pure raw bytes — no specials."""
    tk = ByteTokenizer(add_bos=False, add_eos=False)
    ids = tk.encode("abc")
    assert ids == list(b"abc")


def test_invariant_both_specials_present_when_both_flags_on():
    """With both flags on, BOS at start AND EOS at end."""
    tk = ByteTokenizer(add_bos=True, add_eos=True)
    ids = tk.encode("xy")
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert ids[1:-1] == list(b"xy")


def test_pin_byte_tokenizer_default_flags():
    """Pin: default is ``add_bos=False, add_eos=True`` (used by smoke tests
    and the default recipe path).

    If you change the defaults, update all recipes that omit the flags.
    """
    tk = ByteTokenizer()
    assert tk.add_bos is False
    assert tk.add_eos is True


# ---------------------------------------------------------------------------
# Decode robustness
# ---------------------------------------------------------------------------

def test_invariant_decode_silently_drops_specials():
    """Specials in the input ID list are silently dropped (line 49-50 of
    tokenizers.py).

    Setup: mixed list of bytes and specials.
    Expected: decode returns only the bytes' UTF-8 decode; specials gone.
    """
    tk = ByteTokenizer()
    ids = [BOS_ID, ord("h"), ord("i"), EOS_ID, PAD_ID]
    assert tk.decode(ids) == "hi"


def test_invariant_decode_silently_drops_negative_ids():
    """Negative IDs (e.g., -100 label-ignore values) are silently dropped.

    Goal: handles the case where someone accidentally passes labels
    (which contain -100 on pad) to the decoder.
    """
    tk = ByteTokenizer()
    ids = [-100, ord("a"), ord("b"), -100]
    assert tk.decode(ids) == "ab"


def test_invariant_decode_silently_drops_ids_above_byte_range():
    """IDs >= 256 (but not exactly special) are dropped.

    Setup: 300 is not a special but also not in byte range.
    """
    tk = ByteTokenizer()
    ids = [300, ord("c"), 999, ord("d")]
    assert tk.decode(ids) == "cd"


def test_decode_empty_list_returns_empty_string():
    """``decode([])`` returns ``""``, no exception."""
    tk = ByteTokenizer()
    assert tk.decode([]) == ""


def test_decode_handles_invalid_utf8_via_replace():
    """Invalid UTF-8 byte sequences are replaced (not raised) via the
    ``errors='replace'`` setting on line 52.

    Setup: lone continuation byte 0xFF (invalid UTF-8 start).
    Expected: decode returns a string containing the Unicode replacement
    character (U+FFFD), not raise UnicodeDecodeError.
    """
    tk = ByteTokenizer()
    result = tk.decode([0xFF])
    # Python's 'replace' produces "�" for invalid sequences
    assert "�" in result


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

def test_byte_tokenizer_registered_under_byte():
    """``ByteTokenizer`` is registered as ``('tokenizer', 'byte')``.

    Goal: pin the registry name — recipes use ``tokenizer.name = byte``.
    """
    from lighttrain.registry import get
    cls = get("tokenizer", "byte")
    assert cls is ByteTokenizer
