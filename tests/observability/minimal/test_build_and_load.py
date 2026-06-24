"""Adversarial tests for ``lighttrain.observability.minimal``.

The minimal module is the "repro" backbone — it must reconstruct a model
from a tiny spec WITHOUT pulling in OmegaConf / Pydantic / Trainer. These
tests pin the public surface:

* ``build_minimal_model`` via short name (registry) AND via ``_target_``.
* ``build_minimal_model`` from a JSON file on disk.
* Missing ``name`` and ``_target_`` raises ValueError.
* ``_target_`` colon-form vs rpartition-form both work.
* ``load_state`` round-trip with safetensors AND .pt.
* ``load_state`` strict=False tolerates missing keys.
* ``dump_spec`` produces JSON-safe values for every supported type
  (int / float / str / bool / None / list / dict / nested / custom obj).
* ``_jsonable`` falls back to ``str(...)`` for non-JSON types.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from lighttrain.observability.minimal import (
    _import_target,
    _jsonable,
    build_minimal_model,
    dump_spec,
    load_state,
)
from lighttrain.registry import register

# ---------------------------------------------------------------------------
# build_minimal_model — short name path
# ---------------------------------------------------------------------------

def test_build_via_short_name_uses_registry(clean_registry):
    """Closed form: ``{"name": "minimal_linear", "params": {"in_features": 4,
    "out_features": 2}}`` constructs a Linear(4, 2)-equivalent.
    """
    @register("model", "minimal_linear_test")
    class _MyLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.lin = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.lin(x)

    model = build_minimal_model({
        "name": "minimal_linear_test",
        "params": {"in_features": 4, "out_features": 2},
    })
    assert isinstance(model, _MyLinear)
    assert model.lin.in_features == 4
    assert model.lin.out_features == 2


def test_build_via_target_dotted_path():
    """``_target_`` dotted path constructs the class via importlib."""
    model = build_minimal_model({
        "_target_": "torch.nn.Linear",
        "params": {"in_features": 8, "out_features": 4},
    })
    assert isinstance(model, nn.Linear)
    assert model.in_features == 8


def test_build_via_target_colon_form():
    """``pkg.module:Class`` colon form is also supported (line 95-96 of
    source).
    """
    model = build_minimal_model({
        "_target_": "torch.nn:Linear",
        "params": {"in_features": 2, "out_features": 1},
    })
    assert isinstance(model, nn.Linear)


def test_build_missing_name_and_target_raises_value_error():
    """A spec with NEITHER ``name`` nor ``_target_`` raises ValueError
    (line 49-50 of source).
    """
    with pytest.raises(ValueError, match="missing 'name'"):
        build_minimal_model({"params": {"x": 1}})


def test_build_empty_dict_raises_value_error():
    """An empty spec also raises."""
    with pytest.raises(ValueError):
        build_minimal_model({})


def test_build_from_json_file_on_disk(tmp_path, clean_registry):
    """Spec can be a path; the file is parsed via ``json.loads`` (line 88-91)."""
    @register("model", "minimal_test_from_json")
    class _Tiny(nn.Module):
        def __init__(self, dim: int = 4) -> None:
            super().__init__()
            self.dim = dim

    spec_path = tmp_path / "model_spec.json"
    spec_path.write_text(json.dumps({
        "name": "minimal_test_from_json",
        "params": {"dim": 7},
    }))
    model = build_minimal_model(spec_path)
    assert model.dim == 7


def test_build_from_missing_json_file_raises_file_not_found_error(tmp_path):
    """Spec path that doesn't exist raises FileNotFoundError (line 89-90)."""
    with pytest.raises(FileNotFoundError):
        build_minimal_model(tmp_path / "no_such_spec.json")


# ---------------------------------------------------------------------------
# load_state — safetensors and .pt
# ---------------------------------------------------------------------------

def test_load_state_safetensors_round_trip(tmp_path):
    """Save model state to safetensors → load into a fresh model → tensors
    match via ``assert_close``.
    """
    torch.manual_seed(0)
    src = nn.Linear(4, 2)
    sd = src.state_dict()
    p = tmp_path / "state.safetensors"
    from safetensors.torch import save_file
    save_file(dict(sd), str(p))

    fresh = nn.Linear(4, 2)
    # Re-init fresh so its weights differ from src
    with torch.no_grad():
        fresh.weight.zero_()
        fresh.bias.zero_()
    load_state(fresh, p)

    for k in sd:
        torch.testing.assert_close(
            fresh.state_dict()[k], sd[k], atol=1e-5, rtol=1e-4
        )


def test_load_state_pt_round_trip(tmp_path):
    """Same round trip via torch.save .pt path (line 68-69 of source)."""
    torch.manual_seed(0)
    src = nn.Linear(4, 2)
    sd = src.state_dict()
    p = tmp_path / "state.pt"
    torch.save(dict(sd), str(p))

    fresh = nn.Linear(4, 2)
    with torch.no_grad():
        fresh.weight.zero_()
        fresh.bias.zero_()
    load_state(fresh, p)

    for k in sd:
        torch.testing.assert_close(fresh.state_dict()[k], sd[k], atol=1e-5, rtol=1e-4)


def test_pin_load_state_strict_false_tolerates_missing_keys(tmp_path):
    """Pin: ``strict=False`` is the default (line 60) so missing keys do
    not raise.

    Setup: save state with only `weight` (no bias); load into Linear(2,1).
    Expected: no exception; weight loaded; bias untouched.
    """
    src = nn.Linear(4, 2)
    sd = {"weight": src.weight.detach()}
    p = tmp_path / "partial.safetensors"
    from safetensors.torch import save_file
    save_file(sd, str(p))

    fresh = nn.Linear(4, 2)
    original_bias = fresh.bias.detach().clone()
    load_state(fresh, p)  # strict=False default → no raise
    torch.testing.assert_close(
        fresh.weight.detach(), src.weight.detach(), atol=1e-5, rtol=1e-4
    )
    # Bias untouched because it was not in the saved state.
    torch.testing.assert_close(fresh.bias.detach(), original_bias, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# dump_spec / _jsonable
# ---------------------------------------------------------------------------

def test_dump_spec_round_trip_primitives():
    """dump_spec preserves primitives unchanged.

    Closed form: ``dump_spec("m", {"a": 1, "b": 2.5, "c": "x", "d": True,
    "e": None})`` → params dict matches input exactly.
    """
    out = dump_spec("m", {"a": 1, "b": 2.5, "c": "x", "d": True, "e": None})
    assert out["name"] == "m"
    assert out["params"] == {"a": 1, "b": 2.5, "c": "x", "d": True, "e": None}


def test_dump_spec_round_trip_nested_dict():
    """Nested dict structures are preserved recursively."""
    out = dump_spec("m", {"nested": {"deep": {"value": 1}}})
    assert out["params"] == {"nested": {"deep": {"value": 1}}}


def test_dump_spec_tuple_becomes_list():
    """Tuples are coerced to lists (line 107-108 of source) — JSON has no tuples."""
    out = dump_spec("m", {"betas": (0.9, 0.99)})
    assert out["params"]["betas"] == [0.9, 0.99]
    assert isinstance(out["params"]["betas"], list)


def test_dump_spec_non_json_value_falls_back_to_str_repr():
    """Pin: arbitrary objects fall back to ``str(...)`` (line 111).

    Setup: pass an nn.Linear instance as a value.
    Expected: value is the repr of the Linear (not raise).
    """
    layer = nn.Linear(2, 2)
    out = dump_spec("m", {"layer": layer})
    assert isinstance(out["params"]["layer"], str)


def test_dump_spec_round_trip_through_json():
    """The produced spec is JSON-serializable — round-trips through
    ``json.dumps`` / ``loads``.
    """
    out = dump_spec("m", {
        "a": 1, "b": (1, 2, 3), "c": {"nested": True}, "d": [1.0, 2.0],
    })
    text = json.dumps(out)
    back = json.loads(text)
    assert back == out


# ---------------------------------------------------------------------------
# _jsonable parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected_type",
    [
        (1, int),
        (1.5, float),
        ("hello", str),
        (True, bool),
        (None, type(None)),
        ([1, 2], list),
        ((1, 2), list),         # tuple → list
        ({"a": 1}, dict),
        (object(), str),         # bare object → str(...) fallback
    ],
)
def test_invariant_jsonable_type_coercion_matrix(value, expected_type):
    """``_jsonable`` returns the expected output type for each input kind."""
    out = _jsonable(value)
    assert isinstance(out, expected_type)


def test_jsonable_dict_with_non_string_keys_coerces_keys_to_str():
    """Pin: dict keys are stringified (line 110: ``str(k)``)."""
    out = _jsonable({1: "a", 2.5: "b"})
    assert set(out.keys()) == {"1", "2.5"}


# ---------------------------------------------------------------------------
# _import_target
# ---------------------------------------------------------------------------

def test_import_target_colon_form():
    """``"torch.nn:Linear"`` resolves to ``torch.nn.Linear``."""
    cls = _import_target("torch.nn:Linear")
    assert cls is nn.Linear


def test_import_target_rpartition_form():
    """``"torch.nn.Linear"`` resolves via rpartition."""
    cls = _import_target("torch.nn.Linear")
    assert cls is nn.Linear


def test_import_target_invalid_path_raises():
    """A garbage path raises (ModuleNotFoundError or AttributeError).
    """
    with pytest.raises((ModuleNotFoundError, AttributeError, ImportError)):
        _import_target("no.such.module.path:Class")
