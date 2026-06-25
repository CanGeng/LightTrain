"""Adversarial tests for ``lighttrain.config._loader._parse_override_value`` /
``_apply_overrides``.

Coverage layered on top of the flat ``tests/test_config.py`` smoke checks:

* Parametrized magic-bool / colon-path / comment-char regressions (8 cases each
  vs the legacy 1 case each).
* Force-add depth-5 nesting (legacy only goes depth-1).
* Unicode keys / values.
* Whitespace-only and empty value boundaries.
* Tilde delete: regression pin tied to the v0.1.4 changelog entry.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from lighttrain.config import ConfigError, load_config
from lighttrain.config._loader import _apply_overrides, _parse_override_value

# ---------------------------------------------------------------------------
# _parse_override_value — direct unit tests of the scalar coercer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5", 5),
        ("-3", -3),
        ("0", 0),
        ("5.5", 5.5),
        ("-3.14", -3.14),
        ("1e-4", 1e-4),
        ("0.0", 0.0),
    ],
)
def test_parse_override_int_and_float(raw, expected):
    """Numeric literals must be coerced to int/float.

    Goal: confirm the ``int(s)`` / ``float(s)`` try-cast fast-path runs before
    YAML safe_load is reached. Otherwise YAML would still coerce these, but the
    code path matters for stability of edge cases (1e-4 etc.).

    Closed form: literal Python ``int``/``float`` value.
    """
    got = _parse_override_value(raw)
    assert type(got) is type(expected)
    assert got == expected


@pytest.mark.parametrize(
    "raw",
    ["on", "off", "yes", "no", "Yes", "On", "OFF", "Off", "YES", "NO"],
)
def test_regression_CFG_PARSE_01_magic_bool_stays_string(raw):
    """Pre-fix bug: YAML 1.1 ``safe_load`` mapped on/off/yes/no to bool, so
    ``mode=on`` became ``True``. Fix: ``_parse_override_value`` short-circuits
    these by not invoking YAML for tokens that do not start with ``[ { ' "``.

    Input: each of the 10 YAML 1.1 magic-bool spellings.
    Closed form: the input string itself (no coercion).

    Pre-fix bug: YAML 1.1 magic strings (on/off/yes/no/...) silently coerced
    to bool, breaking ``mode=on`` etc. (see docs/changelog/v0.1.4:
    "[override 值解析脆弱]").
    """
    assert _parse_override_value(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "/tmp/foo",
        "/var/log/x.log",
        "C:/Users/x",
        "gs://bucket/key",
        "s3://bucket/key",
        "http://example.com:8080/path",
        "redis://localhost:6379/0",
    ],
)
def test_regression_CFG_PARSE_01_colon_path_stays_string(raw):
    """Pre-fix bug: ``yaml.safe_load("/tmp/foo")`` returns ``{"/tmp/foo": None}``
    in some YAML flavors; ``s3://x`` was similarly mis-parsed. Fix: skip YAML
    for inputs not starting with ``[ { ' "``.

    Input: paths/URLs with colons and slashes — all things a YAML container
    parser could misinterpret.
    Closed form: the input string itself.

    Pre-fix bug: colon-bearing paths and URLs were parsed as YAML containers
    (see docs/changelog/v0.1.4: "[override 值解析脆弱]").
    """
    assert _parse_override_value(raw) == raw


@pytest.mark.parametrize(
    "raw",
    ["#alpha", "#1", "# comment-like", "###"],
)
def test_regression_CFG_PARSE_01_comment_char_stays_string(raw):
    """Pre-fix bug: ``yaml.safe_load("#alpha")`` returns ``None`` because ``#``
    starts a YAML comment. Fix: skip YAML for ``#``-prefixed values.

    Input: strings beginning with one or more ``#``.
    Closed form: the input string itself.

    Pre-fix bug: ``#``-prefixed values collapsed to None (see
    docs/changelog/v0.1.4: "[override 值解析脆弱]").
    """
    assert _parse_override_value(raw) == raw


@pytest.mark.parametrize("raw", ["null", "None", "~"])
def test_parse_override_explicit_null(raw):
    """Three canonical spellings of None are all accepted.

    Goal: hand-verify the explicit-literal branch (line 80 of _loader.py).
    Closed form: Python ``None``.
    """
    assert _parse_override_value(raw) is None


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("false", False), ("True", True), ("False", False)],
)
def test_parse_override_bool_canonical_only(raw, expected):
    """Only the four canonical bool spellings get coerced.

    Goal: distinguish accepted bool literals from rejected YAML-1.1 magic
    spellings (covered by test_regression_CFG_PARSE_01_magic_bool_stays_string).
    Closed form: Python ``True``/``False``.
    """
    got = _parse_override_value(raw)
    assert got is expected


def test_parse_override_yaml_container_list():
    """Container-prefixed values dispatch to YAML.

    Input: ``[a, b, c]`` — YAML list syntax.
    Closed form: Python list ``["a", "b", "c"]``.
    """
    assert _parse_override_value("[a, b, c]") == ["a", "b", "c"]


def test_parse_override_yaml_container_dict():
    """Container-prefixed values dispatch to YAML for dicts too.

    Input: ``{a: 1, b: 2}`` — YAML flow-mapping syntax.
    Closed form: Python dict ``{"a": 1, "b": 2}``.
    """
    assert _parse_override_value("{a: 1, b: 2}") == {"a": 1, "b": 2}


def test_parse_override_quoted_string_via_yaml():
    """Quoted strings are unquoted by the YAML branch.

    Input: ``"hello"`` — single-token quoted YAML string.
    Closed form: ``hello`` (no quotes).
    """
    assert _parse_override_value('"hello"') == "hello"


def test_parse_override_empty_string_preserved():
    """Empty string returns empty string, not None.

    Input: literal empty string (early-return branch in _parse_override_value).
    Closed form: ``""``.
    """
    assert _parse_override_value("") == ""


def test_parse_override_whitespace_only_value_round_trips():
    """A whitespace-only value preserves the original (after the empty-string
    fast path is bypassed by the leading whitespace).

    Input: ``"   "`` — three spaces. After ``.strip()`` it becomes empty, no
    explicit literal matches, int/float try-cast fails, no container prefix
    matches, so the literal-string fallback (line 102 of _loader.py) returns
    the original ``val`` unchanged.
    Closed form: ``"   "``.
    """
    assert _parse_override_value("   ") == "   "


# ---------------------------------------------------------------------------
# _apply_overrides — set / force-add / delete behavior on real cfgs
# ---------------------------------------------------------------------------

def _empty_cfg():
    return OmegaConf.create({})


def test_apply_override_force_add_creates_intermediates_depth_five():
    """Force-add (``++``) creates missing intermediate dicts at depth 5.

    Goal: assert ``OmegaConf.update(..., force_add=True)`` chain. Legacy tests
    only go depth-1 (``++run_dir=...``).
    Input: empty cfg, override ``++a.b.c.d.e=1``.
    Closed form: ``cfg.a.b.c.d.e == 1`` and the chain ``a.b.c.d`` is a dict.
    """
    cfg = _apply_overrides(_empty_cfg(), ["++a.b.c.d.e=1"])
    assert cfg.a.b.c.d.e == 1
    assert isinstance(cfg.a.b.c.d, type(cfg.a.b.c))  # both DictConfig


def test_regression_CFG_SET_01_plain_override_missing_key_raises():
    """Regression (v0.1.10, breaking): a plain ``a.b=c`` override against a key
    that does not exist is now rejected instead of silently creating it.

    Pre-fix bug: on an unstructured DictConfig, OmegaConf's ``force_add=False``
    still created missing intermediates, so a typo'd override (e.g.
    ``train.max_steps=3`` against a recipe whose key is ``trainer.max_steps``)
    was silently accepted and then dropped by ``extra='allow'`` Pydantic
    validation — the training step cap never took effect, with no error.

    Input: empty cfg, override ``mode.nested=foo`` (no ``++`` prefix).
    Closed form: a ``ConfigError`` is raised AND its message both names the key
    and suggests the ``++`` add form.

    To deliberately add a new key, use the ``++`` prefix (see
    ``test_apply_override_force_add_creates_intermediates_depth_five``).
    """
    with pytest.raises(ConfigError) as exc_info:
        _apply_overrides(_empty_cfg(), ["mode.nested=foo"])
    msg = str(exc_info.value)
    assert "mode.nested" in msg
    assert "++mode.nested=foo" in msg


def test_invariant_plain_override_existing_key_still_sets():
    """Invariant: the existence guard must not regress the common case — a plain
    override that targets an *existing* leaf still sets it (no ``++`` needed)."""
    cfg = OmegaConf.create({"trainer": {"max_steps": 2000}})
    out = _apply_overrides(cfg, ["trainer.max_steps=3"])
    assert out.trainer.max_steps == 3


def test_invariant_plain_override_existing_none_leaf_still_sets():
    """Invariant: a leaf that exists with value ``None`` counts as existing and
    is settable without ``++`` (the guard walks structurally, not by value)."""
    cfg = OmegaConf.create({"run_dir": None})
    out = _apply_overrides(cfg, ["run_dir=runs/x"])
    assert out.run_dir == "runs/x"


def test_regression_CFG_TILDE_01_missing_intermediate_raises_configerror():
    """Pre-fix bug: ``~missing.nested.key`` raised raw ``KeyError`` from
    OmegaConf when the intermediate node did not exist; the fix catches
    KeyError/AttributeError and re-raises ConfigError with a helpful message.

    Input: empty cfg, override ``~missing.nested.key``.
    Closed form: a ``ConfigError`` is raised AND its message mentions the
    failing key.

    Pre-fix bug: ``~`` delete with missing intermediate raised bare KeyError
    instead of ConfigError (see docs/changelog/v0.1.4:
    "[override `~key` 删除路径异常]").
    """
    with pytest.raises(ConfigError) as exc_info:
        _apply_overrides(_empty_cfg(), ["~missing.nested.key"])
    # the message should mention the override, not be an opaque KeyError str
    assert "missing.nested.key" in str(exc_info.value) or "~" in str(exc_info.value)


def test_apply_override_tilde_deletes_existing_key(tmp_yaml):
    """``~key`` removes an existing top-level key end-to-end through load_config;
    the validated RootConfig then reports it as absent (None).

    Input: YAML with ``run_dir: runs/keep``, override ``~run_dir``.
    Closed form: ``cfg.run_dir is None`` after the delete.
    """
    p = tmp_yaml("mode: lab\nrun_dir: runs/keep\n")
    cfg = load_config(p, overrides=["~run_dir"])
    assert cfg.run_dir is None


def test_invariant_tilde_missing_leaf_is_noop():
    """Invariant: when intermediates exist but the leaf is absent, ``~`` is a
    silent noop ("ensure absent" semantics).

    Input: cfg with ``model={"name": "a"}``, override ``~model.missing_leaf``.
    Closed form: cfg unchanged, no exception raised, ``cfg.model.name == "a"``.
    """
    cfg = OmegaConf.create({"model": {"name": "a"}})
    out = _apply_overrides(cfg, ["~model.missing_leaf"])
    assert out.model.name == "a"


def test_apply_override_empty_key_after_tilde_raises():
    """Whitespace-only key after ``~`` is rejected.

    Input: override ``"~ "`` — leading tilde, trailing whitespace.
    Closed form: ``ConfigError`` (line 113 of _loader.py).
    """
    with pytest.raises(ConfigError):
        _apply_overrides(_empty_cfg(), ["~ "])


def test_apply_override_no_equals_raises():
    """Override without ``=`` is malformed.

    Input: ``"malformed_no_eq"`` (legacy file covers this too; we re-pin via the
    direct-call form so we exercise ``_apply_overrides`` without going through
    ``load_config``).
    Closed form: ``ConfigError`` containing the literal "=".
    """
    with pytest.raises(ConfigError) as exc:
        _apply_overrides(_empty_cfg(), ["malformed_no_eq"])
    assert "=" in str(exc.value)


def test_apply_override_empty_key_after_equals_raises():
    """``=value`` is rejected because the key is empty.

    Input: override ``"=foo"``.
    Closed form: ``ConfigError`` (line 134-135 of _loader.py).
    """
    with pytest.raises(ConfigError):
        _apply_overrides(_empty_cfg(), ["=foo"])


def test_apply_override_empty_value_after_equals_preserved():
    """``key=`` (empty value after equals sign) preserves the empty string.

    Input: empty cfg, override ``++name=``.
    Closed form: ``cfg.name == ""``.
    """
    cfg = _apply_overrides(_empty_cfg(), ["++name="])
    assert cfg.name == ""


def test_apply_override_idempotent_double_set():
    """Applying the same override twice yields the same final cfg, no error.

    Goal: pin the idempotency of repeated overrides — protects against any
    future change that might raise on "duplicate set".
    Input: ``++mode=lab`` applied twice in one override list.
    Closed form: ``cfg.mode == "lab"`` after both passes.
    """
    cfg = _apply_overrides(_empty_cfg(), ["++mode=lab", "++mode=lab"])
    assert cfg.mode == "lab"


def test_apply_override_unicode_key_and_value_preserved():
    """Unicode keys and values round-trip exactly through the override parser.

    Input: ``++名前=測試`` — Japanese + Traditional Chinese tokens.
    Closed form: ``cfg["名前"] == "測試"`` (no NFC/NFD normalization).
    """
    cfg = _apply_overrides(_empty_cfg(), ["++名前=測試"])
    assert cfg["名前"] == "測試"


def test_apply_overrides_through_load_config_round_trip(tmp_yaml):
    """End-to-end: ``load_config`` with multiple overrides applies them in
    order and returns a validated RootConfig.

    Input: trivial YAML, overrides ``[++mode=lab, ++seed=99, ++custom=x]``.
    Closed form: ``cfg.mode=="lab"``, ``cfg.seed==99``, ``cfg.custom=="x"``.
    """
    p = tmp_yaml("seed: 1\n")
    cfg = load_config(p, overrides=["++mode=lab", "++seed=99", "++custom=x"])
    assert cfg.mode == "lab"
    assert cfg.seed == 99
    assert cfg.custom == "x"
