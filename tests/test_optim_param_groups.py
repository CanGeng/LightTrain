"""Optim wrapper param-group DSL: regex match + first-match-wins + options."""

from __future__ import annotations

import torch

from lighttrain.optim.wrappers import AdamWWrapper, ParamGroupSpec, _split_param_groups


def _toy_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=True),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4, bias=True),
    )


def test_no_specs_collapses_to_single_group():
    model = _toy_model()
    groups = _split_param_groups(model, None, {"lr": 0.01})
    assert len(groups) == 1
    assert groups[0]["lr"] == 0.01
    assert sum(p.numel() for p in groups[0]["params"]) == sum(
        p.numel() for p in model.parameters()
    )


def test_first_match_wins():
    model = _toy_model()
    specs = [
        ParamGroupSpec(pattern=r"\.bias$", options={"weight_decay": 0.0}),
        ParamGroupSpec(pattern=r"\.weight$", options={"weight_decay": 0.1}),
    ]
    groups = _split_param_groups(model, specs, {"lr": 1e-3})
    by_wd = {g["weight_decay"]: g for g in groups}
    bias_params = sum(p.numel() for p in by_wd[0.0]["params"])
    weight_params = sum(p.numel() for p in by_wd[0.1]["params"])
    # Total partition: every param is in exactly one bucket.
    assert bias_params + weight_params == sum(p.numel() for p in model.parameters())
    # The bias bucket is non-empty.
    assert bias_params > 0


def test_unmatched_params_fall_back():
    model = _toy_model()
    specs = [ParamGroupSpec(pattern=r"never_matches", options={"lr": 999.0})]
    groups = _split_param_groups(model, specs, {"lr": 1e-3})
    # Spec bucket pruned (empty); fallback bucket kept.
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3


def test_adamw_wrapper_builds_and_steps():
    model = _toy_model()
    w = AdamWWrapper(lr=1e-3)
    opt = w.build(model)
    assert isinstance(opt, torch.optim.AdamW)

    x = torch.randn(2, 8)
    y = model(x).sum()
    y.backward()
    w.step()
    w.zero_grad()
    assert w.optimizer.param_groups[0]["lr"] == 1e-3
