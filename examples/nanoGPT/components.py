"""nanoGPT dataset + collator components for lighttrain.

Registered via ``user_modules`` in the recipe YAML:
  user_modules: [examples/nanoGPT/components.py]

BinaryMemmapDataset: reads the uint16 .bin token files produced by the
  data/*/prepare.py scripts (same format as the original nanoGPT).
StackCollator: stacks (input_ids, labels) pairs — no padding needed since
  all windows are exactly block_size tokens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lighttrain.registry import register


@register("dataset", "memmap_bin")
class BinaryMemmapDataset(Dataset):
    """Random-access windows over a uint16 token binary file.

    Each sample is a (input_ids, labels) pair of length ``block_size`` where
    labels = input_ids shifted left by one (next-token prediction targets).

    Args:
        path: Path to the .bin file (uint16 numpy memmap).
        block_size: Context window length.
        tokenizer: Injected by the data module; unused (data is pre-tokenized).
    """

    def __init__(self, path: str, block_size: int = 1024, tokenizer: Any = None) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = int(block_size)
        if len(self.data) <= self.block_size:
            raise ValueError(
                f"{path}: file has {len(self.data)} tokens but block_size={self.block_size};"
                " run the prepare.py script first."
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


@register("collator", "stack_xy")
class StackCollator:
    """Stack fixed-length (input_ids, labels) samples — no padding."""

    def __init__(self, pad_id: int | None = None) -> None:  # pad_id injected, unused
        pass

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([s["input_ids"] for s in samples]),
            "labels": torch.stack([s["labels"] for s in samples]),
        }


__all__ = ["BinaryMemmapDataset", "StackCollator"]
