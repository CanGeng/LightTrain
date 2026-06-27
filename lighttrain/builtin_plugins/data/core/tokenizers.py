"""Byte-level tokenizer + HuggingFace AutoTokenizer bridge.

Vocab = 260: ids 0..255 are raw bytes, plus 4 specials (PAD=256, BOS=257,
EOS=258, UNK=259). Hermetic: no network, no model files. Smoke tests use
this tokenizer to avoid depending on a HuggingFace tokenizer.

``HFAutoTokenizer`` is the built-in ``hf_auto`` short name: a thin bridge to
``transformers.AutoTokenizer`` that explicitly satisfies
``TokenizerProtocol`` (encode/decode) plus exposes the common ``vocab_size``
/ ``pad_id`` / ``bos_id`` / ``eos_id`` properties. It replaces the older
``__getattr__``-magic adapter that lived in ``examples/MiniMind``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lighttrain.registry import register

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259
VOCAB_SIZE = 260

# Directory of the vendored Qwen3-0.6B tokenizer files (see
# ``builtin_plugins/data/_q3_tok_baseline/``). The CLI ``prune-tokenizer``
# command uses this as the default ``--tokenizer`` path.
QWEN3_BASELINE_DIR: Path = Path(__file__).resolve().parents[1] / "_q3_tok_baseline"


@register("tokenizer", "byte")
class ByteTokenizer:
    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    unk_id: int = UNK_ID
    vocab_size: int = VOCAB_SIZE

    def __init__(self, add_bos: bool = False, add_eos: bool = True) -> None:
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)

    def encode(self, text: str, **_: Any) -> list[int]:
        if isinstance(text, bytes):
            data = text
        else:
            data = text.encode("utf-8", errors="replace")
        ids = list(data)
        if self.add_bos:
            ids = [self.bos_id, *ids]
        if self.add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], **_: Any) -> str:
        out = bytearray()
        for i in ids:
            if i < 0 or i >= 256:
                continue  # drop specials silently
            out.append(int(i))
        return out.decode("utf-8", errors="replace")


@register("tokenizer", "hf_auto")
class HFAutoTokenizer:
    """Built-in bridge to ``transformers.AutoTokenizer``.

    Explicitly implements ``encode`` / ``decode`` so the
    ``TokenizerProtocol`` check is structural (not ``__getattr__`` magic),
    and caches the common scalar properties once at construction so callers
    don't pay HF's internal lookup on every read.

    Usage in a recipe::

        tokenizer:
          name: hf_auto
          path: path/to/tokenizer_dir
          use_fast: true
    """

    def __init__(self, path: str, *, use_fast: bool = True, **kwargs: Any) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(path, use_fast=use_fast, **kwargs)
        self._vocab_size = int(self._tok.vocab_size)
        pad = self._tok.pad_token_id
        self._pad_id = int(pad) if pad is not None else 0
        self._bos_id = self._tok.bos_token_id
        self._eos_id = self._tok.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_id(self) -> int | None:
        return self._eos_id

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self._tok.encode(text, **kwargs)

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return self._tok.decode(ids, **kwargs)  # type: ignore[return-value]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tok(*args, **kwargs)


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "HFAutoTokenizer",
    "ByteTokenizer",
    "PAD_ID",
    "QWEN3_BASELINE_DIR",
    "UNK_ID",
    "VOCAB_SIZE",
]
