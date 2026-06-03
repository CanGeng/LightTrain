"""OptimizerCPUOffloadWrapper — DESIGN §14.2 step 5 (M5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import lighttrain.builtin_plugins.layer_offload  # noqa: F401 — register cpu_offload
import lighttrain.builtin_plugins.optim.wrappers  # noqa: F401 — register adamw / lion

from lighttrain.registry import get as _registry_get


def test_cpu_offload_registered():
    cls = _registry_get("optimizer", "cpu_offload")
    assert cls is not None


def test_cpu_offload_step_matches_plain_adamw():
    """Two copies of the same model: one trained with adamw, one with the
    cpu_offload wrapper around adamw. After one step, params must match
    closely (host fp32 master + bf16 cast roundtrip may introduce small
    drift, so we use atol=1e-4)."""
    torch.manual_seed(0)
    m1 = nn.Linear(8, 16)
    torch.manual_seed(0)
    m2 = nn.Linear(8, 16)
    assert torch.allclose(m1.weight, m2.weight)

    # Forward + loss + backward on both, same batch.
    x = torch.randn(4, 8)
    y = torch.randn(4, 16)
    for m in (m1, m2):
        out = m(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()

    # Train m1 with plain AdamW
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    opt1.step()

    # Train m2 with cpu_offload(adamw)
    CpuOffload = _registry_get("optimizer", "cpu_offload")
    opt2_wrapper = CpuOffload(base={"name": "adamw"}, lr=1e-3)
    opt2_wrapper.build(m2)
    opt2_wrapper.step()

    assert torch.allclose(m1.weight, m2.weight, atol=1e-5, rtol=1e-5), (
        f"max diff = {(m1.weight - m2.weight).abs().max()}"
    )


def test_cpu_offload_zero_grad_clears_grads():
    m = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    (m(x).sum()).backward()
    CpuOffload = _registry_get("optimizer", "cpu_offload")
    opt = CpuOffload(base={"name": "adamw"}, lr=1e-3)
    opt.build(m)
    assert m.weight.grad is not None
    opt.zero_grad(set_to_none=True)
    assert m.weight.grad is None


def test_cpu_offload_state_dict_round_trips():
    m = nn.Linear(4, 4)
    (m(torch.randn(1, 4)).sum()).backward()
    CpuOffload = _registry_get("optimizer", "cpu_offload")
    opt = CpuOffload(base={"name": "adamw"}, lr=1e-3)
    opt.build(m)
    opt.step()
    sd = opt.state_dict()
    assert "inner" in sd
    m2 = nn.Linear(4, 4)
    opt2 = CpuOffload(base={"name": "adamw"}, lr=1e-3)
    opt2.build(m2)
    opt2.load_state_dict(sd)
