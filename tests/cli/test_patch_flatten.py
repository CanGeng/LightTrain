"""Adversarial tests for ``lighttrain.cli._app._flatten_patch_to_overrides``.

The flatten helper turns a nested dict (parsed from
``--apply-degrade patch.yaml``) into ``++a.b=value`` OmegaConf overrides.

Coverage:

* Depth-5 nesting (legacy tests cover depth-1 only).
* None values must emit ``=null`` and NOT be silently dropped — otherwise
  the patch ``{lr: None}`` cannot delete a key by overwriting it.
* List values dispatch through ``yaml.safe_dump`` so OmegaConf re-parses them.
* Non-dict input returns an empty list (safety on malformed YAML root).
* Empty dict returns empty list.
* Round-trip through ``_apply_overrides`` (so we verify the produced strings
  are parseable by the override stack).
"""

from __future__ import annotations

from omegaconf import OmegaConf

from lighttrain.cli._app import _flatten_patch_to_overrides
from lighttrain.config._loader import _apply_overrides


def test_flatten_simple_nested_dict_to_dotted_keys():
    """One-level nested dict yields a single ``++a.b=value`` override.

    Input: ``{"a": {"b": 1}}``.
    Closed form: ``["++a.b=1"]`` exactly.
    """
    assert _flatten_patch_to_overrides({"a": {"b": 1}}) == ["++a.b=1"]


def test_flatten_deeply_nested_five_levels():
    """5-level nesting flattens to a single dotted override.

    Input: ``{a: {b: {c: {d: {e: 1}}}}}``.
    Closed form: ``["++a.b.c.d.e=1"]``.
    """
    patch = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    assert _flatten_patch_to_overrides(patch) == ["++a.b.c.d.e=1"]


def test_regression_CLI_FLATTEN_LIST_01_uses_flow_style_yaml():
    """Pre-fix bug: ``yaml.safe_dump([1, 2])`` defaults to block style
    (``"- 1\\n- 2"``); since ``_parse_override_value`` only routes to YAML
    parsing when the input starts with ``[ { ' "``, the block-form list was
    stored as the literal string ``"- 1\\n- 2"`` instead of a list. Fix:
    pass ``default_flow_style=True`` so safe_dump emits ``"[1, 2]"`` which
    survives round-trip through ``_parse_override_value``.

    Input: ``{"a": [1, 2]}``.
    Closed form: produced override starts with ``++a=[`` AND round-trip
    yields the original list ``[1, 2]``.

    Pre-fix bug: ``_flatten_patch_to_overrides`` emitted block-YAML for
    sequences, which the conservative override parser then preserved as a
    literal multi-line string (discovered while writing this suite — fixed
    inline in this PR: see _app.py line 62-72).
    """
    overrides = _flatten_patch_to_overrides({"a": [1, 2]})
    assert len(overrides) == 1
    # Flow-style YAML produces a value starting with `[` which is exactly
    # what _parse_override_value recognizes as a YAML container.
    assert overrides[0].startswith("++a=[")
    # Round-trip: the produced override must be re-parseable to the original.
    cfg = _apply_overrides(OmegaConf.create({}), overrides)
    assert list(cfg.a) == [1, 2]


def test_flatten_tuple_value_treated_as_list():
    """Tuples are coerced via ``list(v)`` before YAML dump (line 66).

    Input: ``{"a": (1, 2, 3)}``.
    Closed form: round-tripped value == ``[1, 2, 3]`` (tuples lose identity
    but preserve element order via YAML).
    """
    overrides = _flatten_patch_to_overrides({"a": (1, 2, 3)})
    cfg = _apply_overrides(OmegaConf.create({}), overrides)
    assert list(cfg.a) == [1, 2, 3]


def test_flatten_none_value_emits_literal_null():
    """None values become ``++key=null`` (line 69-70 of _app.py), NOT skipped.

    Goal: an apply-degrade patch with ``warmup: None`` must be able to override
    a non-null setting to null. If None were silently dropped, the patch
    would have no effect.

    Input: ``{"a": None}``.
    Closed form: produced override is ``["++a=null"]``; when applied, cfg.a is None.
    """
    overrides = _flatten_patch_to_overrides({"a": None})
    assert overrides == ["++a=null"]
    cfg = _apply_overrides(OmegaConf.create({"a": "original"}), overrides)
    assert cfg.a is None


def test_flatten_empty_dict_yields_empty_list():
    """Empty patch produces no overrides.

    Input: ``{}``.
    Closed form: ``[]``.
    """
    assert _flatten_patch_to_overrides({}) == []


def test_flatten_non_dict_input_yields_empty_list():
    """Defensive: non-dict input (e.g., YAML file with top-level scalar)
    produces empty list, not exception (line 56-57 of _app.py).

    Input: scalar ``42`` and list ``[1, 2]``.
    Closed form: empty list for both — safety on malformed patches.
    """
    assert _flatten_patch_to_overrides(42) == []
    assert _flatten_patch_to_overrides([1, 2]) == []
    assert _flatten_patch_to_overrides(None) == []


def test_flatten_mixed_types_preserved_via_round_trip():
    """Mixed scalar / nested / list types round-trip cleanly.

    Input: ``{"a": 1, "b": {"c": "str", "d": True}, "e": [1, 2, 3]}``.
    Closed form: applying produced overrides reconstructs the original dict
    when read via OmegaConf containers.
    """
    patch = {"a": 1, "b": {"c": "str", "d": True}, "e": [1, 2, 3]}
    overrides = _flatten_patch_to_overrides(patch)
    cfg = _apply_overrides(OmegaConf.create({}), overrides)
    assert cfg.a == 1
    assert cfg.b.c == "str"
    assert cfg.b.d is True
    assert list(cfg.e) == [1, 2, 3]


def test_flatten_bool_value_preserved_through_round_trip():
    """Bool values stay bool through the round-trip (``True`` → ``"True"``
    → ``_parse_override_value`` → ``True``).

    Input: ``{"flag": True}``.
    Closed form: cfg.flag is True (not the string "True" or int 1).
    """
    overrides = _flatten_patch_to_overrides({"flag": True})
    cfg = _apply_overrides(OmegaConf.create({}), overrides)
    assert cfg.flag is True


def test_flatten_string_value_with_colon_round_trips_as_string():
    """A value like ``/tmp/foo`` must round-trip to the literal string, not
    a YAML container. This pins the cross-module contract between the patch
    flattener and ``_parse_override_value`` (regression CFG_PARSE_01).

    Input: ``{"path": "/tmp/foo"}``.
    Closed form: cfg.path == "/tmp/foo" (string).
    """
    overrides = _flatten_patch_to_overrides({"path": "/tmp/foo"})
    cfg = _apply_overrides(OmegaConf.create({}), overrides)
    assert cfg.path == "/tmp/foo"


def test_flatten_prefix_carried_through_recursion():
    """The internal ``prefix`` accumulator joins keys correctly across recursion.

    Input: directly invoke with a prefix argument.
    Closed form: ``prefix + "." + key`` for each leaf.
    """
    out = _flatten_patch_to_overrides({"a": 1, "b": {"c": 2}}, prefix="root")
    assert sorted(out) == sorted(["++root.a=1", "++root.b.c=2"])
