"""Map-style dataset over an on-disk shard directory.

Extracted from the ``materialize`` PrepNode so the core ``PrepRunner`` /
``PrepGraphDataModule`` can read a shard cache without importing the (relocated)
node implementations in ``lighttrain.builtin_plugins.prepgraph.nodes``
(DESIGN §3.3: the graph framework stays in core, node impls are builtin_plugins).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._shards import iter_rows


class _RowsDataset:
    """Map-style dataset over an on-disk shard directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._rows: list[dict[str, Any]] | None = None

    def _materialize(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = list(iter_rows(self.root))
        return self._rows

    def __len__(self) -> int:
        return len(self._materialize())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._materialize()[int(idx)]

    def __iter__(self):
        for r in self._materialize():
            yield r


__all__ = ["_RowsDataset"]
