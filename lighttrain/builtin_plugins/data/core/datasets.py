"""Concrete datasets — minimal, deterministic, in-tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lighttrain.registry import register
from lighttrain.data.core._schema import Sample


@register("dataset", "line_file_text")
class LineFileTextDataset:
    """Map-style dataset reading newline-separated text from a file.

    Loads the whole file at construction time. Tokenization happens up front;
    ``__getitem__`` is a list lookup. Empty lines are dropped.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: Any,
        max_len: int = 256,
        chunk_size: int | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        # Opt-in document chunking for stateful (RWKV/Mamba) streaming: a long
        # document is split into fixed-size chunks and the *first* chunk of each
        # document carries ``_doc_boundary=True`` (the recurrent-state reset
        # point). Chunk first, then cap each chunk at ``max_len`` (keep
        # chunk_size <= max_len). ``None`` keeps the one-line-per-sample default.
        self.chunk_size = int(chunk_size) if chunk_size else None

        text = self.path.read_text(encoding=encoding, errors="replace")
        self.samples: list[Sample] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            if not ids:
                continue
            if self.chunk_size:
                for ci in range(0, len(ids), self.chunk_size):
                    chunk = ids[ci : ci + self.chunk_size][: self.max_len]
                    if not chunk:
                        continue
                    self.samples.append(
                        {
                            "input_ids": chunk,
                            "attention_mask": [1] * len(chunk),
                            "labels": list(chunk),
                            "_doc_boundary": ci == 0,
                        }
                    )
            else:
                ids = ids[: self.max_len]
                self.samples.append(
                    {
                        "input_ids": ids,
                        "attention_mask": [1] * len(ids),
                        "labels": list(ids),
                    }
                )

        if not self.samples:
            raise ValueError(f"No usable lines in {self.path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[int(idx)]

    def __iter__(self):
        return iter(self.samples)


@register("dataset", "preference_jsonl")
class PreferenceJsonlDataset:
    """Map-style dataset reading preference pairs from a JSONL file.

    Each line must be a JSON object containing at minimum:
    ``chosen_input_ids``, ``chosen_labels``,
    ``rejected_input_ids``, ``rejected_labels``.
    An ``id`` field is recommended so artifact stores can join by sample id.

    ``tokenizer`` is accepted but ignored (data is pre-tokenized).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_len: int = 1024,
        tokenizer: Any = None,  # injected by SimpleDataModule / _resolve_base; unused
        encoding: str = "utf-8",
    ) -> None:
        import json

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.max_len = int(max_len)
        self.samples: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding=encoding, errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            obj: dict[str, Any] = json.loads(raw)
            self.samples.append({
                "chosen_input_ids": list(obj["chosen_input_ids"])[: self.max_len],
                "chosen_labels": list(obj["chosen_labels"])[: self.max_len],
                "rejected_input_ids": list(obj["rejected_input_ids"])[: self.max_len],
                "rejected_labels": list(obj["rejected_labels"])[: self.max_len],
                "id": str(obj.get("id", len(self.samples))),
            })
        if not self.samples:
            raise ValueError(f"No usable lines in {self.path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[int(idx)]

    def __iter__(self):
        return iter(self.samples)


__all__ = ["LineFileTextDataset", "PreferenceJsonlDataset"]
