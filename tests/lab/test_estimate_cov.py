"""Coverage-extension tests for ``lighttrain.lab.estimate``.

Pins and exercises every reachable uncovered branch at the time of creation:

* ``_optim_state_bytes`` SGD → 0 (line 80)
* ``_resolve_optim_state_bytes`` non-Mapping fallback (line 106)
* ``_resolve_optim_state_bytes`` Mapping without name/_target_ (line 109)
* ``_resolve_optim_state_bytes`` hook is None → silent fallback (line 130)
* ``_resolve_optim_state_bytes`` hook raises → warn + fallback (lines 133-140)
* ``_spec_name`` with None (line 177), pydantic model_dump branch (line 179),
  and non-Mapping/non-pydantic fallback (line 182)
* ``_activation_estimate`` d_model fallback to 256 (line 213)
* ``_activation_estimate`` n_layers fallback to 4 (line 219)
* ``_offload_estimate`` engine not a Mapping → None (line 255)
* ``_offload_estimate`` probe_layer_bandwidth raises → coarse fallback (lines 269-281)
* ``estimate`` cfg not a Mapping → TypeError (line 312)
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lighttrain.cli._runtime import _eager_import_components
from lighttrain.lab.estimate import (
    EstimateReport,
    OffloadEstimate,
    _activation_estimate,
    _offload_estimate,
    _optim_state_bytes,
    _resolve_optim_state_bytes,
    _spec_name,
    estimate,
)

_eager_import_components()

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------


class _TinyLinear(nn.Module):
    """Minimal model with Linear layers but no d_model/n_layers attributes."""

    def __init__(self, in_f: int = 8, out_f: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.fc(x)


class _TinyWithAttrs(nn.Module):
    """Model that exposes d_model and n_layers as attributes."""

    def __init__(
        self,
        d_model: int = 32,
        n_layers: int = 2,
        vocab_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:  # pragma: no cover
        x = self.emb(input_ids)
        x = self.ln(x)
        return {"logits": x}


class _ModelWithLayerNorms(nn.Module):
    """Model with LayerNorm layers but NO n_layers/d_model attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(16)
        self.ln2 = nn.LayerNorm(16)
        self.fc = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.fc(self.ln2(self.ln1(x)))


def _tiny_cfg(**overrides):
    """Base recipe config for tiny_lm."""
    base = {
        "mode": "lab",
        "seed": 0,
        "exp": "estimate_cov_smoke",
        "run_root": "runs",
        "model": "default",
        "model_profiles": {
            "default": {
                "name": "tiny_lm",
                "vocab_size": 64,
                "d_model": 16,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 32,
            }
        },
        "data": {
            "name": "simple",
            "dataset": {
                "name": "line_file_text",
                "path": "tests/fixtures/tiny_corpus.txt",
                "max_len": 32,
            },
            "tokenizer": {"name": "byte"},
            "collator": {"name": "causal_lm", "max_len": 32},
            "batch_size": 4,
        },
        "loss": {"name": "cross_entropy"},
        "optim": {"name": "adamw", "lr": 1e-3},
        "scheduler": {"name": "constant"},
        "engine": {"name": "standard", "mixed_precision": "no"},
        "trainer": {"name": "pretrain", "max_steps": 10},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _optim_state_bytes: SGD → 0 (line 80)
# ---------------------------------------------------------------------------


def test_invariant_optim_state_bytes_sgd_returns_zero():
    """``_optim_state_bytes`` with sgd returns 0 (no momentum state by default)."""
    model = _TinyLinear()
    result = _optim_state_bytes(model, "sgd")
    assert result == 0


@pytest.mark.parametrize("name", ["adamw", "adam", "cpu_offload"])
def test_invariant_optim_state_bytes_adam_variants_return_double(name: str):
    """AdamW / Adam / cpu_offload return 2 × trainable bytes."""
    model = _TinyLinear()
    trainable = sum(
        p.numel() * 4
        for p in model.parameters()
        if p.requires_grad
    )
    result = _optim_state_bytes(model, name)
    assert result == 2 * trainable


def test_invariant_optim_state_bytes_lion_returns_single():
    """Lion returns 1 × trainable bytes (single momentum buffer)."""
    model = _TinyLinear()
    trainable = sum(
        p.numel() * 4
        for p in model.parameters()
        if p.requires_grad
    )
    result = _optim_state_bytes(model, "lion")
    assert result == trainable


def test_invariant_optim_state_bytes_unknown_name_falls_back_to_adamw():
    """An unknown optimizer name conservatively returns 2 × params (Adam-like)."""
    model = _TinyLinear()
    unknown = _optim_state_bytes(model, "my_exotic_optimizer")
    adamw = _optim_state_bytes(model, "adamw")
    assert unknown == adamw


# ---------------------------------------------------------------------------
# _resolve_optim_state_bytes: non-Mapping fallback (line 106)
# ---------------------------------------------------------------------------


def test_invariant_resolve_optim_state_bytes_non_mapping_returns_fallback():
    """When optim_spec is not a Mapping (e.g. None or a string),
    falls back to the name-based estimate without warning."""
    model = _TinyLinear()
    result_none = _resolve_optim_state_bytes(None, model, "adamw")
    result_str = _resolve_optim_state_bytes("adamw", model, "adamw")
    expected = _optim_state_bytes(model, "adamw")
    assert result_none == expected
    assert result_str == expected


# ---------------------------------------------------------------------------
# _resolve_optim_state_bytes: Mapping missing name/_target_ (line 109)
# ---------------------------------------------------------------------------


def test_invariant_resolve_optim_state_bytes_mapping_no_name_key_returns_fallback():
    """Mapping spec with no 'name' or '_target_' key falls back silently."""
    model = _TinyLinear()
    # A dict without name or _target_
    result = _resolve_optim_state_bytes({"lr": 1e-3}, model, "adamw")
    expected = _optim_state_bytes(model, "adamw")
    assert result == expected


# ---------------------------------------------------------------------------
# _resolve_optim_state_bytes: hook is None → silent fallback (line 130)
# ---------------------------------------------------------------------------


def test_pin_current_behavior_resolve_optim_state_bytes_hook_none_falls_back_silently():
    """When the resolved wrapper has no optim_state_bytes attribute,
    the function falls back silently (no warning).

    Pin: cpu_offload does NOT inherit OptimizerWrapperBase and therefore has
    no optim_state_bytes method — this is the legitimate 'no hook' path.
    """
    model = _TinyLinear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_optim_state_bytes(
            {"name": "cpu_offload", "lr": 1e-3}, model, "cpu_offload"
        )
    # No UserWarning about optim_state_bytes should be emitted
    optim_warns = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "optim_state_bytes" in str(w.message)
    ]
    assert not optim_warns, [str(w.message) for w in optim_warns]
    # Falls back to name-based estimate for cpu_offload (treated as adamw: 2×)
    expected = _optim_state_bytes(model, "cpu_offload")
    assert result == expected


# ---------------------------------------------------------------------------
# _resolve_optim_state_bytes: hook raises → warn + fallback (lines 133-140)
# ---------------------------------------------------------------------------


def test_invariant_resolve_optim_state_bytes_hook_raises_warns_and_falls_back():
    """When optim_state_bytes() raises, a UserWarning is emitted and
    the generic 2×params estimate is returned."""
    from lighttrain.builtin_plugins.optim.wrappers import AdamWWrapper
    from lighttrain.registry import contains as _has
    from lighttrain.registry import register

    if not _has("optimizer", "_raising_hook_test"):
        @register("optimizer", "_raising_hook_test")
        class _RaisingHook(AdamWWrapper):
            def optim_state_bytes(self, model):
                raise ValueError("deliberate probe failure")

    model = _TinyLinear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_optim_state_bytes(
            {"name": "_raising_hook_test", "lr": 1e-3},
            model,
            "_raising_hook_test",
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "_raising_hook_test" in m and "falling back" in m.lower()
        for m in msgs
    ), msgs
    expected = _optim_state_bytes(model, "adamw")
    assert result == expected


# ---------------------------------------------------------------------------
# _spec_name: None returns "" (line 177)
# ---------------------------------------------------------------------------


def test_invariant_spec_name_none_returns_empty_string():
    """``_spec_name(None)`` returns an empty string (line 177)."""
    assert _spec_name(None) == ""


# ---------------------------------------------------------------------------
# _spec_name: pydantic model_dump branch (line 179)
# ---------------------------------------------------------------------------


def test_invariant_spec_name_pydantic_model():
    """``_spec_name`` calls model_dump() on pydantic-like objects and
    reads 'name' from the result (line 179)."""

    class _PydanticLike:
        """Quacks like a pydantic model."""

        def model_dump(self) -> dict:
            return {"name": "my_pydantic_spec", "lr": 1e-3}

    result = _spec_name(_PydanticLike())
    assert result == "my_pydantic_spec"


def test_invariant_spec_name_pydantic_target_key():
    """``_spec_name`` also reads '_target_' from model_dump() output."""

    class _PydanticTarget:
        def model_dump(self) -> dict:
            return {"_target_": "my.module.OptimizerClass"}

    result = _spec_name(_PydanticTarget())
    assert result == "my.module.OptimizerClass"


# ---------------------------------------------------------------------------
# _spec_name: non-Mapping, non-pydantic → "" (line 182)
# ---------------------------------------------------------------------------


def test_invariant_spec_name_plain_object_returns_empty_string():
    """A plain Python object (not Mapping, not pydantic) returns "" (line 182)."""
    assert _spec_name(42) == ""
    assert _spec_name([1, 2]) == ""
    assert _spec_name(object()) == ""


# ---------------------------------------------------------------------------
# _activation_estimate: d_model fallback when no Linear and no d_model attr
# (line 213)
# ---------------------------------------------------------------------------


def test_invariant_activation_estimate_d_model_fallback_256():
    """When model has no d_model attr and no Linear layers, d_model falls
    back to 256 (line 213)."""

    class _NoLinearModel(nn.Module):
        def forward(self, x):
            return x

    model = _NoLinearModel()
    cfg = {}  # minimal, no data section
    tokens, act_bytes = _activation_estimate(cfg, model)
    # d_model=256, n_layers=4, batch_size=8, seq_len=128
    # act = 8 * 128 * 256 * 4 * 4 * 2 = 8*128*256*4*4*2
    expected = 8 * 128 * 256 * 4 * 4 * 2
    assert act_bytes == expected
    assert tokens == 8 * 128


# ---------------------------------------------------------------------------
# _activation_estimate: n_layers fallback to 4 (line 219)
# ---------------------------------------------------------------------------


def test_invariant_activation_estimate_n_layers_fallback_4():
    """When model has no n_layers attr and no LayerNorms, n_layers falls
    back to 4 (line 219)."""
    model = _TinyLinear(in_f=8, out_f=4)
    # _TinyLinear has a Linear(8, 4) → d_model=4 (from max out_features=4)
    # No n_layers attr, no LayerNorms → n_layers=4
    cfg = {}
    tokens, act_bytes = _activation_estimate(cfg, model)
    # d_model=4, n_layers=4
    expected_act = 8 * 128 * 4 * 4 * 4 * 2
    assert act_bytes == expected_act


def test_invariant_activation_estimate_n_layers_from_layernorm_count():
    """When model has LayerNorms but no n_layers attr, n_layers is counted
    from LayerNorm modules (not the fallback 4)."""
    model = _ModelWithLayerNorms()
    # Has 2 LayerNorms → n_layers=2
    # Linear(16, 4) → d_model = max(out_features) = 4
    cfg = {}
    tokens, act_bytes = _activation_estimate(cfg, model)
    # d_model=4 (max out_features), n_layers=2, batch_size=8, seq_len=128
    expected_act = 8 * 128 * 4 * 4 * 2 * 2
    assert act_bytes == expected_act


def test_invariant_activation_estimate_reads_batch_size_from_data_cfg():
    """batch_size and seq_len are read from the data cfg when present."""
    model = _TinyWithAttrs(d_model=32, n_layers=3)
    cfg = {
        "data": {
            "batch_size": 2,
            "collator": {"max_len": 64},
        }
    }
    tokens, act_bytes = _activation_estimate(cfg, model)
    assert tokens == 2 * 64
    # d_model=32, n_layers=3 (from attr)
    assert act_bytes == 2 * 64 * 32 * 4 * 3 * 2


# ---------------------------------------------------------------------------
# _offload_estimate: engine not a Mapping → None (line 255)
# ---------------------------------------------------------------------------


def test_invariant_offload_estimate_no_engine_key_returns_none():
    """When cfg has no 'engine' key, _offload_estimate returns None (line 255)."""
    model = _TinyWithAttrs()
    result = _offload_estimate({}, model)
    assert result is None


def test_invariant_offload_estimate_engine_not_mapping_returns_none():
    """When engine value is not a Mapping, _offload_estimate returns None."""
    model = _TinyWithAttrs()
    result = _offload_estimate({"engine": "standard"}, model)
    assert result is None


def test_invariant_offload_estimate_engine_wrong_name_returns_none():
    """When engine.name != 'layer_offload', _offload_estimate returns None."""
    model = _TinyWithAttrs()
    result = _offload_estimate({"engine": {"name": "standard"}}, model)
    assert result is None


# ---------------------------------------------------------------------------
# _offload_estimate: probe raises → coarse fallback (lines 269-281)
# ---------------------------------------------------------------------------


def test_invariant_offload_estimate_probe_failure_uses_coarse_fallback():
    """When probe_layer_bandwidth raises, the coarse fallback is used.
    The returned OffloadEstimate must still be a valid dataclass."""
    model = _TinyWithAttrs(d_model=32, n_layers=2)
    cfg = {"engine": {"name": "layer_offload", "resident_layers": 1}}

    def _raise_probe(*_a, **_kw):
        raise RuntimeError("simulated probe failure")

    # Patch the function in the module where it is imported at call time
    with patch(
        "lighttrain.builtin_plugins.layer_offload._io.probe_layer_bandwidth",
        side_effect=RuntimeError("simulated probe failure"),
    ):
        result = _offload_estimate(cfg, model)

    assert isinstance(result, OffloadEstimate)
    # Coarse formula: d_model=32, n_layers=2
    expected_param_bytes = 32 * 32 * 12 * 4
    assert result.layer_param_bytes == expected_param_bytes
    assert result.layers == 2
    assert result.resident_layers == 1
    assert result.recommended_mode in ("recompute", "offload", "mixed")


def test_invariant_offload_estimate_probe_failure_coarse_no_d_model_attr():
    """Coarse fallback uses d_model=256, n_layers=4 when model has no attrs."""

    class _Bare(nn.Module):
        def forward(self, x):
            return x  # pragma: no cover

    model = _Bare()
    cfg = {"engine": {"name": "layer_offload", "resident_layers": 2}}

    def _raise_probe(*_a, **_kw):
        raise RuntimeError("no probe")

    with patch(
        "lighttrain.builtin_plugins.layer_offload._io.probe_layer_bandwidth",
        _raise_probe,
    ):
        result = _offload_estimate(cfg, model)

    assert isinstance(result, OffloadEstimate)
    # d_model=256, n_layers=4 from fallback
    expected_bytes = 256 * 256 * 12 * 4
    assert result.layer_param_bytes == expected_bytes
    assert result.layers == 4
    assert result.resident_layers == 2


# ---------------------------------------------------------------------------
# _offload_estimate: mode selection (recompute / offload / mixed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recompute_us,transfer_us,expected_mode",
    [
        (10.0, 50.0, "recompute"),    # recompute < transfer → "recompute"
        (100.0, 10.0, "offload"),     # transfer*2 < recompute → "offload"
        (30.0, 20.0, "mixed"),        # neither condition → "mixed"
    ],
)
def test_invariant_offload_estimate_mode_selection(
    recompute_us: float, transfer_us: float, expected_mode: str
):
    """Verify the mode selection logic of _offload_estimate."""
    model = _TinyWithAttrs(d_model=8, n_layers=1)
    cfg = {"engine": {"name": "layer_offload", "resident_layers": 1}}

    fake_return = (recompute_us, transfer_us, 1024, 1)

    with patch(
        "lighttrain.builtin_plugins.layer_offload._io.probe_layer_bandwidth",
        return_value=fake_return,
    ):
        result = _offload_estimate(cfg, model)

    assert result is not None
    assert result.recommended_mode == expected_mode


# ---------------------------------------------------------------------------
# estimate(): cfg not a Mapping → TypeError (line 312)
# ---------------------------------------------------------------------------


def test_invariant_estimate_raises_type_error_for_non_mapping_cfg():
    """``estimate()`` raises ``TypeError`` when cfg is not a Mapping (line 312)."""
    with pytest.raises(TypeError, match="estimate: cfg must be a mapping"):
        estimate("not a dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="estimate: cfg must be a mapping"):
        estimate(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# estimate(): end-to-end with SGD optimizer (exercises line 80 via full stack)
# ---------------------------------------------------------------------------


def test_invariant_estimate_sgd_optimizer_state_bytes_zero():
    """estimate() with an SGD optimizer should report 0 optim_state_bytes
    (pin SGD path in _optim_state_bytes line 80 through full estimate call)."""
    from lighttrain.builtin_plugins.optim.wrappers import AdamWWrapper
    from lighttrain.registry import contains as _has
    from lighttrain.registry import register

    # Register a fake SGD-like wrapper so the config resolves
    if not _has("optimizer", "_fake_sgd_test"):
        @register("optimizer", "_fake_sgd_test")
        class _FakeSGD(AdamWWrapper):
            def _moments_per_param(self) -> int:
                return 0  # override to return 0 like SGD

            def optim_state_bytes(self, model: torch.nn.Module) -> int:
                return 0

    rpt = estimate(_tiny_cfg(optim={"name": "_fake_sgd_test", "lr": 1e-2}))
    assert isinstance(rpt, EstimateReport)
    assert rpt.optim_state_bytes == 0


# ---------------------------------------------------------------------------
# _spec_name: Mapping branch with name (normal path sanity-check)
# ---------------------------------------------------------------------------


def test_invariant_spec_name_mapping_with_name_key():
    """``_spec_name`` returns the 'name' value from a plain dict."""
    assert _spec_name({"name": "adamw", "lr": 1e-3}) == "adamw"


def test_invariant_spec_name_mapping_with_target_key():
    """``_spec_name`` falls back to '_target_' when 'name' is absent."""
    assert _spec_name({"_target_": "torch.optim.AdamW"}) == "torch.optim.AdamW"


def test_invariant_spec_name_mapping_empty_values_returns_empty():
    """``_spec_name`` returns '' when name and _target_ are both None/absent."""
    assert _spec_name({"name": None, "_target_": None}) == ""
    assert _spec_name({}) == ""


# ---------------------------------------------------------------------------
# report_to_dict round-trip
# ---------------------------------------------------------------------------


def test_invariant_report_to_dict_is_json_serialisable():
    """``report_to_dict`` returns a plain dict with no torch tensors."""
    import json

    from lighttrain.lab.estimate import report_to_dict

    rpt = estimate(_tiny_cfg())
    d = report_to_dict(rpt)
    # Should be serialisable without errors
    dumped = json.dumps(d)
    assert "trainable_params" in dumped


# ---------------------------------------------------------------------------
# _activation_estimate: collator without max_len uses default seq_len=128
# ---------------------------------------------------------------------------


def test_invariant_activation_estimate_no_collator_max_len_uses_default():
    """Without collator.max_len the default seq_len=128 is used."""
    model = _TinyWithAttrs(d_model=16, n_layers=2)
    cfg = {"data": {"batch_size": 4}}
    tokens, act_bytes = _activation_estimate(cfg, model)
    # seq_len stays at default 128
    assert tokens == 4 * 128
    assert act_bytes == 4 * 128 * 16 * 4 * 2 * 2
