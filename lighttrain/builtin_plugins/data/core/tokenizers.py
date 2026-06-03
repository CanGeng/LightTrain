"""Byte-level tokenizer.

Vocab = 260: ids 0..255 are raw bytes, plus 4 specials (PAD=256, BOS=257,
EOS=258, UNK=259). Hermetic: no network, no model files. Smoke tests use
this tokenizer to avoid depending on a HuggingFace tokenizer.
"""

from __future__ import annotations

from typing import Any

from lighttrain.registry import register


PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259
VOCAB_SIZE = 260


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


__all__ = ["BOS_ID", "ByteTokenizer", "EOS_ID", "PAD_ID", "UNK_ID", "VOCAB_SIZE"]
