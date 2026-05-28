"""Tests for lighttrain.config."""

from __future__ import annotations

import pytest

from lighttrain.config import (
    ComponentSpec,
    ConfigError,
    ConfigResolveError,
    ConfigSchemaError,
    RootConfig,
    dump_resolved,
    load_config,
    resolve,
)
from lighttrain.registry import register


def test_basic_yaml_loads(tmp_yaml):
    p = tmp_yaml("mode: lab\nseed: 7\n")
    cfg = load_config(p)
    assert isinstance(cfg, RootConfig)
    assert cfg.mode == "lab"
    assert cfg.seed == 7


def test_default_mode_is_lab(tmp_yaml):
    p = tmp_yaml("seed: 1\n")
    cfg = load_config(p)
    assert cfg.mode == "lab"


def test_missing_file_raises(tmp_yaml):
    with pytest.raises(ConfigError):
        load_config("/no/such/path.yaml")


def test_invalid_mode_raises(tmp_yaml):
    p = tmp_yaml("mode: chaos\n")
    with pytest.raises(ConfigSchemaError):
        load_config(p)


def test_seed_type_error(tmp_yaml):
    p = tmp_yaml("seed: not-a-number\n")
    with pytest.raises(ConfigSchemaError):
        load_config(p)


def test_overrides_basic(tmp_yaml):
    p = tmp_yaml("mode: lab\nseed: 1\n")
    cfg = load_config(p, overrides=["seed=42"])
    assert cfg.seed == 42


def test_overrides_dotted_path(tmp_yaml):
    p = tmp_yaml("model:\n  name: a\n  lr: 0.1\n")
    cfg = load_config(p, overrides=["model.lr=0.5"], validate=False)
    assert cfg.model.lr == 0.5


def test_overrides_force_add(tmp_yaml):
    p = tmp_yaml("mode: lab\n")
    cfg = load_config(p, overrides=["++run_dir=runs/x"])
    assert cfg.run_dir == "runs/x"


def test_overrides_delete(tmp_yaml):
    p = tmp_yaml("mode: lab\nrun_dir: runs/keep\n")
    cfg = load_config(p, overrides=["~run_dir"])
    assert cfg.run_dir is None


def test_overrides_yaml_value(tmp_yaml):
    p = tmp_yaml("mode: lab\n")
    cfg = load_config(p, overrides=["++user_modules=[a,b]"], validate=False)
    assert list(cfg.user_modules) == ["a", "b"]


def test_overrides_invalid_format(tmp_yaml):
    p = tmp_yaml("mode: lab\n")
    with pytest.raises(ConfigError):
        load_config(p, overrides=["malformed_no_eq"])


def test_defaults_compose(tmp_config_dir):
    base = tmp_config_dir / "base.yaml"
    base.write_text("mode: prod\nseed: 99\n", encoding="utf-8")
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [base]\nseed: 100\n", encoding="utf-8")
    cfg = load_config(leaf)
    assert cfg.mode == "prod"
    assert cfg.seed == 100  # leaf overrides base


def test_defaults_chain(tmp_config_dir):
    a = tmp_config_dir / "a.yaml"
    a.write_text("mode: lab\nseed: 1\n", encoding="utf-8")
    b = tmp_config_dir / "b.yaml"
    b.write_text("defaults: [a]\nseed: 2\n", encoding="utf-8")
    c = tmp_config_dir / "c.yaml"
    c.write_text("defaults: [b]\nseed: 3\n", encoding="utf-8")
    cfg = load_config(c)
    assert cfg.mode == "lab"
    assert cfg.seed == 3


def test_defaults_missing_raises(tmp_config_dir):
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [missing]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(leaf)


def test_interpolation_internal(tmp_yaml):
    p = tmp_yaml("base: 10\nderived: ${base}\n")
    cfg = load_config(p, validate=False)
    assert cfg.derived == 10


def test_env_interpolation(tmp_yaml, monkeypatch):
    monkeypatch.setenv("LT_TEST_VAR", "hello")
    p = tmp_yaml('greeting: ${oc.env:LT_TEST_VAR}\n')
    cfg = load_config(p, validate=False)
    assert cfg.greeting == "hello"


def test_dump_resolved_root(tmp_yaml):
    p = tmp_yaml("mode: prod\nseed: 5\n")
    cfg = load_config(p)
    text = dump_resolved(cfg)
    assert "mode: prod" in text
    assert "seed: 5" in text


def test_component_spec_xor_required():
    with pytest.raises(ValueError):
        ComponentSpec(name=None, _target_=None)
    with pytest.raises(ValueError):
        ComponentSpec(name="x", _target_="lighttrain.foo.Bar")


def test_component_spec_short_name_only():
    cs = ComponentSpec(name="adamw", params={"lr": 1e-4})
    assert cs.name == "adamw"
    assert cs.params == {"lr": 1e-4}


def test_component_spec_target_only():
    cs = ComponentSpec(_target_="lighttrain.optim.AdamW", params={"lr": 1e-4})
    assert cs.target == "lighttrain.optim.AdamW"


def test_resolve_short_name(clean_registry):
    class FakeOptim:
        def __init__(self, lr: float = 1e-3) -> None:
            self.lr = lr

    register("optimizer", "fake_opt", FakeOptim)
    obj = resolve({"name": "fake_opt", "lr": 0.5}, category="optimizer")
    assert isinstance(obj, FakeOptim)
    assert obj.lr == 0.5


def test_resolve_short_name_requires_category(clean_registry):
    with pytest.raises(ConfigResolveError):
        resolve({"name": "x"})


def test_resolve_target(clean_registry):
    obj = resolve({"_target_": "decimal.Decimal", "value": "3.14"}, instantiate=True)
    assert str(obj) == "3.14"


def test_resolve_target_factory_only():
    factory = resolve(
        {"_target_": "decimal.Decimal"}, instantiate=False
    )
    from decimal import Decimal

    assert factory is Decimal


def test_resolve_target_invalid():
    with pytest.raises(ConfigResolveError):
        resolve({"_target_": "no.such.module:Nope"})


def test_resolve_xor_enforced():
    with pytest.raises(ValueError):
        resolve({"name": "a", "_target_": "b.C"}, category="model")


def test_resolve_construction_failure(clean_registry):
    class Strict:
        def __init__(self, x: int) -> None:
            self.x = x

    register("model", "strict", Strict)
    with pytest.raises(ConfigResolveError):
        resolve({"name": "strict", "wrong_kwarg": 1}, category="model")


def test_user_modules_field(tmp_yaml):
    p = tmp_yaml("mode: lab\nuser_modules: [./a.py, ./b.py]\n")
    cfg = load_config(p)
    assert list(cfg.user_modules) == ["./a.py", "./b.py"]


def test_extra_top_level_keys_allowed(tmp_yaml):
    """RootConfig has extra='allow' so users can add experimental fields."""
    p = tmp_yaml("mode: lab\nmy_experimental_field: 42\n")
    cfg = load_config(p)
    assert cfg.my_experimental_field == 42  # type: ignore[attr-defined]
