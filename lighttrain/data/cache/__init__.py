"""Tokenized cache backends.

Two flavours:

* ``_shards`` — variable-length JSONL/Parquet shards. The general path used by
  ``tokenize`` / ``mix`` / ``validate`` PrepGraph nodes.
* ``_memmap`` — fixed-shape, mmap-friendly storage for ``packed`` data. Used
  by ``materialize`` when downstream sees uniform ``seq_len`` rows.
"""

from ._memmap import MemmapDataset, MemmapHeader, read_header, write_memmap
from ._shards import (
    ShardWriter,
    cache_key,
    count_rows,
    iter_rows,
    read_manifest,
    shard_state,
)

__all__ = [
    "MemmapDataset",
    "MemmapHeader",
    "ShardWriter",
    "cache_key",
    "count_rows",
    "iter_rows",
    "read_header",
    "read_manifest",
    "shard_state",
    "write_memmap",
]
