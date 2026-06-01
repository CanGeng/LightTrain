"""NvmeStorage — DESIGN §14.1 (M5 — thread-pool fallback path).

True io_uring is M5 deferred (M5 interface doc); these tests verify the
thread-pool fallback round-trips a tiny module to disk and back.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lighttrain.plugins.layer_offload import NvmeStorage


def test_nvme_storage_round_trip(tmp_path):
    src = nn.Linear(8, 16)
    dst = nn.Linear(8, 16)
    storage = NvmeStorage(root=tmp_path / "weights", device=torch.device("cpu"))
    storage.init_from_layer("layer.0", src)
    storage.swap_in("layer.0", dst)
    assert torch.allclose(src.weight, dst.weight)
    assert torch.allclose(src.bias, dst.bias)
    storage.close()


def test_nvme_storage_swap_out_writes_modified_weights(tmp_path):
    src = nn.Linear(8, 16)
    storage = NvmeStorage(root=tmp_path / "weights", device=torch.device("cpu"))
    storage.init_from_layer("layer.0", src)
    # Mutate, swap_out, swap_in into a fresh module — should reflect mutation.
    with torch.no_grad():
        src.weight.data += 1.0
    storage.swap_out("layer.0", src)
    dst = nn.Linear(8, 16)
    storage.swap_in("layer.0", dst)
    assert torch.allclose(src.weight, dst.weight)
    storage.close()
