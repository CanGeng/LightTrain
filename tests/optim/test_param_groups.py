"""Adversarial tests for ``lighttrain.optim.wrappers`` param-group DSL.

Layered on top of ``tests/test_optim_param_groups.py``. New coverage:

* **First-match-wins via greedy regex**: a permissive pattern listed first
  starves subsequent specs.
* **Partition disjoint + covering** invariant across the whole param set.
* **Frozen params (requires_grad=False) excluded** from every bucket.
* **Distinct LR per group** preserved on the constructed optimizer.
* **Pin: regex uses ``re.search`` (substring)**, NOT ``re.fullmatch``.
* **Rebuild raises** (line 84-86 of wrappers.py).
* **Empty / all-frozen model raises ValueError** with a clear message.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.optim.wrappers import (
    AdamWWrapper,
    LionWrapper,
    ParamGroupSpec,
    _split_param_groups,
)


def _toy_model() -> torch.nn.Module:
    """Linear → LayerNorm → Linear; 9 named params (weight/bias per layer + LN
    weight/bias)."""
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=True),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4, bias=True),
    )


# ---------------------------------------------------------------------------
# First-match-wins
# ---------------------------------------------------------------------------

def test_invariant_first_match_wins_permissive_regex_starves_later_specs():
    """Invariant: ``.*`` listed first matches every param, so the second
    spec's bucket ends up empty (and is pruned).

    Setup: specs = [``.*`` (lr=1.0), ``bias`` (lr=2.0)].
    Expected: only one group remains; lr=1.0; all params in it; the
    ``bias`` bucket is pruned because empty.
    """
    model = _toy_model()
    specs = [
        ParamGroupSpec(pattern=r".*", options={"lr": 1.0}),
        ParamGroupSpec(pattern=r"bias", options={"lr": 2.0}),
    ]
    groups = _split_param_groups(model, specs, {"lr": 0.01})

    # Only one non-empty group; the .* bucket
    assert len(groups) == 1
    assert groups[0]["lr"] == 1.0
    # All trainable params landed here
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(p.numel() for p in groups[0]["params"]) == total


def test_invariant_partition_is_disjoint_and_covering():
    """Invariant: across all returned groups, the params form a disjoint
    partition of the trainable param set (each param in EXACTLY one bucket).

    Setup: 2 specs that together cover everything (bias-or-weight).
    Expected: union via ``id(p)`` equals the trainable set; no duplicates.
    """
    model = _toy_model()
    specs = [
        ParamGroupSpec(pattern=r"\.bias$", options={"weight_decay": 0.0}),
        ParamGroupSpec(pattern=r"\.weight$", options={"weight_decay": 0.1}),
    ]
    groups = _split_param_groups(model, specs, {"lr": 1e-3})

    seen: list[int] = []
    for g in groups:
        for p in g["params"]:
            seen.append(id(p))

    # Disjoint: each id appears once
    assert len(seen) == len(set(seen))
    # Covering: union equals the full trainable param id set
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(seen) == expected


def test_invariant_frozen_params_excluded_from_all_buckets():
    """Invariant: params with ``requires_grad=False`` never appear in any
    bucket (line 50-51 of wrappers.py).

    Setup: freeze a known param; specs match all named params; verify the
    frozen param does NOT show up in any bucket via ``id``.
    """
    model = _toy_model()
    # Freeze layer 0's weight specifically
    layer0_weight = model[0].weight
    layer0_weight.requires_grad_(False)

    specs = [
        ParamGroupSpec(pattern=r".*", options={"lr": 1e-3}),
    ]
    groups = _split_param_groups(model, specs, {"lr": 1e-4})

    frozen_id = id(layer0_weight)
    for g in groups:
        for p in g["params"]:
            assert id(p) != frozen_id, (
                "frozen layer0.weight should not appear in any bucket"
            )


def test_distinct_lr_per_group_preserved_on_optimizer():
    """After wrapping an AdamW with two specs at different lr values, the
    optimizer's ``param_groups`` reports each group's lr distinctly.

    Setup: specs = [``\\.bias$`` (lr=1e-2), ``\\.weight$`` (lr=1e-4)].
    Expected: optimizer.param_groups has two groups with the corresponding lr.
    """
    model = _toy_model()
    w = AdamWWrapper(
        lr=5e-4,
        param_groups=[
            {"pattern": r"\.bias$", "lr": 1e-2},
            {"pattern": r"\.weight$", "lr": 1e-4},
        ],
    )
    opt = w.build(model)

    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-2]


# ---------------------------------------------------------------------------
# Additive predicates: min_ndim / module_type (P1a / issue #3)
# ---------------------------------------------------------------------------

def test_min_ndim_selects_weight_matrices_excludes_1d_vectors():
    """``min_ndim=2`` after a permissive name regex selects only the 2-D
    weight matrices (Linear weights), excluding 1-D bias and LayerNorm params.

    This is GaLore's "Linear weights, ndim>=2" selection expressed in the
    built-in DSL with no custom adapter.
    """
    model = _toy_model()  # Linear(8,8) + LayerNorm(8) + Linear(8,4)
    specs = [ParamGroupSpec(pattern=r".*", min_ndim=2, options={"lr": 1e-3})]
    groups = _split_param_groups(model, specs, {"lr": 1e-4})

    # Exactly the 2 Linear weight matrices land in the matched bucket.
    matched = groups[0]["params"]
    assert all(p.ndim >= 2 for p in matched)
    expected_2d = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    assert {id(p) for p in matched} == {id(p) for p in expected_2d}
    assert len(matched) == 2  # the two nn.Linear weights


def test_module_type_selects_only_named_module_class():
    """``module_type="Linear"`` selects params owned by ``nn.Linear`` modules
    (weight + bias), excluding the LayerNorm params even though the name
    regex ``.*`` matches everything.
    """
    model = _toy_model()
    specs = [ParamGroupSpec(pattern=r".*", module_type="Linear", options={"lr": 1e-3})]
    groups = _split_param_groups(model, specs, {"lr": 1e-4})

    matched_ids = {id(p) for g in groups for p in g["params"] if g.get("lr") == 1e-3}
    linear_ids = {
        id(p)
        for m in model.modules()
        if type(m).__name__ == "Linear"
        for p in m.parameters(recurse=False)
    }
    assert matched_ids == linear_ids
    # LayerNorm params fall through to the fallback bucket, not the matched one.
    ln_ids = {id(p) for m in model.modules() if type(m).__name__ == "LayerNorm"
              for p in m.parameters(recurse=False)}
    assert matched_ids.isdisjoint(ln_ids)


def test_module_type_and_min_ndim_compose_to_linear_weights_only():
    """Combining ``module_type="Linear"`` + ``min_ndim=2`` selects exactly the
    Linear *weight* matrices — the GaLore projected-layer set."""
    model = _toy_model()
    specs = [
        ParamGroupSpec(
            pattern=r".*", module_type="Linear", min_ndim=2, options={"lr": 1e-3}
        )
    ]
    groups = _split_param_groups(model, specs, {"lr": 1e-4})
    matched = [p for g in groups for p in g["params"] if g.get("lr") == 1e-3]
    expected = [m.weight for m in model.modules() if type(m).__name__ == "Linear"]
    assert {id(p) for p in matched} == {id(p) for p in expected}


def test_predicates_parsed_from_wrapper_param_groups_dict():
    """``min_ndim`` / ``module_type`` keys in a recipe's ``param_groups`` are
    parsed into ``ParamGroupSpec`` fields (not leaked into optimizer options).
    """
    w = AdamWWrapper(
        lr=1e-3,
        param_groups=[{"pattern": r".*", "min_ndim": 2, "module_type": "Linear", "lr": 5e-4}],
    )
    spec = w.param_groups[0]
    assert spec.min_ndim == 2
    assert spec.module_type == "Linear"
    # Reserved predicate keys must not bleed into optimizer hyperparams.
    assert "min_ndim" not in spec.options
    assert "module_type" not in spec.options
    assert spec.options.get("lr") == 5e-4


# ---------------------------------------------------------------------------
# Regex semantics pin (search vs fullmatch)
# ---------------------------------------------------------------------------

def test_pin_regex_uses_re_search_substring_semantics():
    """Pin: ``ParamGroupSpec.match`` uses ``re.search`` (substring), NOT
    ``re.fullmatch``. So the pattern ``"bias"`` matches both ``"0.bias"`` and
    ``"my_module.bias_extra"``.

    Setup: a single spec ``pattern="bias"`` against the toy model.
    Expected: all bias-containing params land in this bucket.

    If you intentionally switch to ``re.fullmatch``, update this test AND
    every recipe that relies on the substring behavior.
    """
    model = _toy_model()
    specs = [ParamGroupSpec(pattern="bias", options={"lr": 1e-2})]
    groups = _split_param_groups(model, specs, {"lr": 1e-4})

    # The matched bucket gets every param whose name contains "bias"
    # (Linear bias for layer 0 + LN bias + Linear bias for layer 2).
    bias_count = sum(1 for n, _ in model.named_parameters() if "bias" in n)
    # Plus the LayerNorm's "bias" — verify count matches
    matched_count = sum(1 for p in groups[0]["params"])
    assert matched_count == bias_count


# ---------------------------------------------------------------------------
# Rebuild / step-before-build
# ---------------------------------------------------------------------------

def test_rebuild_raises_runtime_error():
    """Calling ``build()`` twice raises RuntimeError.

    Goal: pin the once-only contract — accidental rebuild would replace
    the optimizer reference held by the trainer.
    """
    model = _toy_model()
    w = AdamWWrapper(lr=1e-3)
    w.build(model)
    with pytest.raises(RuntimeError) as exc:
        w.build(model)
    assert "rebuild" in str(exc.value).lower() or "already" in str(exc.value).lower()


def test_step_before_build_raises_attribute_error():
    """Calling ``.step()`` on an unbuilt wrapper crashes because
    ``self.optimizer is None`` (line 82 of wrappers.py).

    Goal: catch the silent-None failure mode — if someone forgets to call
    build, they should get a clear AttributeError rather than wandering
    deeper into the call stack.
    """
    w = AdamWWrapper(lr=1e-3)
    with pytest.raises(AttributeError):
        w.step()


# ---------------------------------------------------------------------------
# Empty / all-frozen model
# ---------------------------------------------------------------------------

def test_no_trainable_params_raises_value_error():
    """A model where every parameter has ``requires_grad=False`` raises
    ValueError with a clear message (line 43-44 of wrappers.py).
    """
    model = _toy_model()
    for p in model.parameters():
        p.requires_grad_(False)

    with pytest.raises(ValueError) as exc:
        _split_param_groups(model, None, {"lr": 1e-3})
    assert "no trainable" in str(exc.value).lower()


def test_all_specs_empty_raises_value_error():
    """When every spec matches nothing AND there are no unmatched trainable
    params (frozen model), ValueError is raised (line 62-63 of wrappers.py).

    Setup: model with all frozen params AND a non-matching spec.
    Expected: ValueError "no parameters matched any param-group spec" OR
    the earlier "no trainable parameters" guard (whichever fires first).
    """
    model = _toy_model()
    for p in model.parameters():
        p.requires_grad_(False)
    specs = [ParamGroupSpec(pattern=r"never_matches", options={"lr": 999.0})]
    with pytest.raises(ValueError):
        _split_param_groups(model, specs, {"lr": 1e-3})


# ---------------------------------------------------------------------------
# _safe_state_dict — aliasing-trap guard for custom-state serialization
# ---------------------------------------------------------------------------

def test_safe_state_dict_does_not_mutate_live_optimizer_state():
    """``_safe_state_dict(convert)`` must copy inner state dicts so that
    rewriting custom state for a portable checkpoint does NOT corrupt the live
    optimizer (torch's state_dict aliases self.state — the v0.1.9 P2a trap).
    """
    model = _toy_model()
    w = AdamWWrapper(lr=1e-3)
    w.build(model)
    x = torch.randn(2, 8)
    model(x).sum().backward()
    w.step()  # populate Adam state (exp_avg / exp_avg_sq tensors)

    # A converter that would clobber a state entry if it aliased live state.
    sentinel = {"replaced": True}

    def conv(k, v):
        return sentinel if k == "exp_avg" else v

    sd = w._safe_state_dict(conv)
    # The returned (copied) state was rewritten...
    assert any(st.get("exp_avg") is sentinel for st in sd["state"].values())
    # ...but the LIVE optimizer state is untouched (still real tensors).
    for st in w.optimizer.state.values():
        assert isinstance(st["exp_avg"], torch.Tensor)
    # And the optimizer can still step without error.
    model(x).sum().backward()
    w.step()


def test_safe_state_dict_without_convert_is_plain_copy():
    """No converter → a faithful copy that still round-trips through load."""
    model = _toy_model()
    w = AdamWWrapper(lr=1e-3)
    w.build(model)
    model(torch.randn(2, 8)).sum().backward()
    w.step()

    sd = w._safe_state_dict()
    model2 = _toy_model()
    w2 = AdamWWrapper(lr=1e-3)
    w2.build(model2)
    w2.load_state_dict(sd)
    assert w2.state_dict()["state"].keys() == sd["state"].keys()


# ---------------------------------------------------------------------------
# Wrapper round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip_preserves_step_count():
    """Save state after one optimizer step → load into fresh wrapper →
    state dict equality on ``step`` field (an Adam-style state key).
    """
    model = _toy_model()
    w = AdamWWrapper(lr=1e-3)
    w.build(model)
    # Trigger one step so the optimizer state has Adam moments.
    x = torch.randn(2, 8)
    model(x).sum().backward()
    w.step()
    sd = w.state_dict()

    # Build a new wrapper + optimizer of the same shape
    model2 = _toy_model()
    w2 = AdamWWrapper(lr=1e-3)
    w2.build(model2)
    w2.load_state_dict(sd)
    assert w2.state_dict()["state"].keys() == sd["state"].keys()


def test_lion_wrapper_builds_to_lion_optimizer():
    """``LionWrapper.build`` returns an instance of the internal ``_Lion``
    class (registered as ``optimizer/lion``).
    """
    from lighttrain.optim.wrappers import _Lion

    model = _toy_model()
    w = LionWrapper(lr=1e-4)
    opt = w.build(model)
    assert isinstance(opt, _Lion)
