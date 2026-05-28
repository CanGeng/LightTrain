"""LR schedulers: shape (warmup → cosine / wsd / linear) + state_dict round-trip."""

from __future__ import annotations

import math

import torch

from lighttrain.optim.schedulers import (
    ConstantScheduler,
    LinearScheduler,
    WarmupCosineScheduler,
    WSDScheduler,
)


def _toy_optim(lr: float = 1.0) -> torch.optim.Optimizer:
    p = torch.nn.Linear(2, 2)
    return torch.optim.SGD(p.parameters(), lr=lr)


def _trace(sched, optim, n: int) -> list[float]:
    out = [optim.param_groups[0]["lr"]]
    for _ in range(n):
        sched.step()
        out.append(optim.param_groups[0]["lr"])
    return out


def test_constant_scheduler_keeps_lr():
    o = _toy_optim(0.1)
    s = ConstantScheduler(o)
    lrs = _trace(s, o, 10)
    assert all(abs(v - 0.1) < 1e-6 for v in lrs)


def test_warmup_cosine_warms_then_decays():
    o = _toy_optim(1.0)
    s = WarmupCosineScheduler(o, warmup_steps=5, total_steps=20, min_lr_ratio=0.0)
    lrs = _trace(s, o, 20)
    # Warmup: ramp 0..1 across first 5 steps.
    assert lrs[1] < lrs[2] < lrs[5]
    assert lrs[5] == 1.0 or lrs[6] == 1.0
    # Decay below 1.0 thereafter, ending near min_lr_ratio.
    assert lrs[-1] < 0.05
    assert lrs[-1] >= 0.0


def test_wsd_decays_after_stable_window():
    o = _toy_optim(2.0)
    s = WSDScheduler(o, warmup_steps=2, stable_steps=5, decay_steps=3, min_lr_ratio=0.5)
    lrs = _trace(s, o, 12)
    # During stable we should hit the base lr.
    assert any(abs(lr - 2.0) < 1e-6 for lr in lrs[2:7])
    # End below stable, above min.
    assert lrs[-1] < 2.0
    assert lrs[-1] >= 1.0 - 1e-6  # 0.5 * 2.0


def test_linear_decay_endpoints():
    o = _toy_optim(1.0)
    s = LinearScheduler(o, total_steps=10, end_factor=0.0, warmup_steps=0)
    lrs = _trace(s, o, 10)
    assert lrs[-1] < 1e-6
    assert lrs[0] == 1.0


def test_state_dict_round_trip():
    o1 = _toy_optim(1.0)
    s1 = WarmupCosineScheduler(o1, warmup_steps=5, total_steps=20)
    for _ in range(7):
        s1.step()
    state = s1.state_dict()

    o2 = _toy_optim(1.0)
    s2 = WarmupCosineScheduler(o2, warmup_steps=5, total_steps=20)
    s2.load_state_dict(state)
    assert s2.last_step == s1.last_step
