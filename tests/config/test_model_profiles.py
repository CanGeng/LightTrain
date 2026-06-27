"""v0.1.8 model_profiles selector — `select_model_spec` semantics.

These exercise the config-group selection that replaced the ambient `model:`
block (closes Issue #1 by construction): a recipe declares named profiles under
`model_profiles:` and selects one with a `model: <name>` string. A bare-dict
`model:` is rejected with a migration hint.
"""

from __future__ import annotations

import pytest

from lighttrain.config import ConfigResolveError
from lighttrain.config._resolver import select_model_spec


def test_string_selector_resolves_to_profile_dict():
    profiles = {"a": {"name": "tiny_lm", "d_model": 128}, "b": {"name": "mamba2_lm"}}
    spec = select_model_spec("a", profiles)
    assert spec == {"name": "tiny_lm", "d_model": 128}
    # Returned dict is a copy — mutating it must not corrupt the profiles.
    spec["d_model"] = 999
    assert profiles["a"]["d_model"] == 128  # type: ignore[index]


def test_unknown_profile_raises_with_available_list():
    with pytest.raises(ConfigResolveError) as ei:
        select_model_spec("nope", {"a": {"name": "tiny_lm"}, "b": {"name": "x"}})
    msg = str(ei.value)
    assert "nope" in msg
    assert "'a'" in msg and "'b'" in msg  # available list is surfaced


def test_string_selector_without_profiles_raises():
    with pytest.raises(ConfigResolveError) as ei:
        select_model_spec("transformer", None)
    assert "model_profiles" in str(ei.value)


def test_bare_dict_model_is_rejected_with_migration_hint():
    with pytest.raises(ConfigResolveError) as ei:
        select_model_spec({"name": "tiny_lm", "d_model": 128}, None)
    msg = str(ei.value)
    assert "removed in v0.1.8" in msg
    assert "--to-profiles" in msg  # discoverability: points at the migrate cmd


def test_none_selector_single_profile_autopicks():
    spec = select_model_spec(None, {"only": {"name": "tiny_lm", "d_model": 64}})
    assert spec == {"name": "tiny_lm", "d_model": 64}


def test_none_selector_multiple_profiles_requires_selector():
    with pytest.raises(ConfigResolveError) as ei:
        select_model_spec(None, {"a": {"name": "x"}, "b": {"name": "y"}})
    assert "selector" in str(ei.value).lower()


def test_nothing_declared_raises_missing_section():
    with pytest.raises(RuntimeError):
        select_model_spec(None, None)


def test_empty_profile_raises():
    with pytest.raises(ConfigResolveError):
        select_model_spec("a", {"a": {}})


def test_profiles_accept_pydantic_or_omegaconf_mappings():
    # _as_plain_dict must coerce a Mapping value (e.g. OmegaConf DictConfig
    # surfaces as a Mapping) into a plain dict spec.
    from collections import OrderedDict

    profiles = OrderedDict({"a": OrderedDict({"name": "tiny_lm", "d_model": 32})})
    spec = select_model_spec("a", profiles)
    assert spec == {"name": "tiny_lm", "d_model": 32}
    assert isinstance(spec, dict)
