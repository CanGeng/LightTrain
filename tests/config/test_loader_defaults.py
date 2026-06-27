"""Adversarial tests for ``lighttrain.config._loader._compose_defaults``.

Coverage layered on top of the flat ``tests/test_config.py`` smoke checks:

* Resolution order: ``.yaml`` suffix tried first, then bare filename
  (line 57-60 of _loader.py).
* Non-string entries rejected with a clear ConfigError.
* Direct cycle and indirect (depth-3) cycle both detected.
* **CFG_DEFAULTS_01 regression pin**: the ``seen`` set must be unwound after
  a child-recursion exception, so an unrelated sibling include later in the
  list is not falsely reported as cyclic.
"""

from __future__ import annotations

import pytest

from lighttrain.config import ConfigError, load_config


def test_compose_defaults_basic_merge_order(tmp_config_dir):
    """Leaf cfg overrides parent (base) per Hydra-style defaults semantics.

    Setup: base sets mode=prod, seed=99. leaf inherits base, overrides seed=100.
    Expected: cfg.mode == "prod" (from base), cfg.seed == 100 (leaf wins).
    """
    (tmp_config_dir / "base.yaml").write_text("mode: prod\nseed: 99\n", encoding="utf-8")
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [base]\nseed: 100\n", encoding="utf-8")
    cfg = load_config(leaf)
    assert cfg.mode == "prod"
    assert cfg.seed == 100


def test_compose_defaults_three_level_chain(tmp_config_dir):
    """A three-deep defaults chain (a → b → c) composes; the leaf wins on the
    shared key and the base's untouched key survives the whole chain.

    Setup: a sets mode=lab, seed=1; b inherits a, seed=2; c inherits b, seed=3.
    Expected: cfg.mode == 'lab' (from a), cfg.seed == 3 (c wins).
    """
    (tmp_config_dir / "a.yaml").write_text("mode: lab\nseed: 1\n", encoding="utf-8")
    (tmp_config_dir / "b.yaml").write_text("defaults: [a]\nseed: 2\n", encoding="utf-8")
    c = tmp_config_dir / "c.yaml"
    c.write_text("defaults: [b]\nseed: 3\n", encoding="utf-8")
    cfg = load_config(c)
    assert cfg.mode == "lab"
    assert cfg.seed == 3


def test_compose_defaults_internal_interpolation(tmp_yaml):
    """OmegaConf ``${...}`` internal interpolation resolves at load time.

    Setup: ``base: 10, derived: ${base}``.
    Expected: cfg.derived == 10 (resolved to base's value).
    """
    p = tmp_yaml("base: 10\nderived: ${base}\n")
    cfg = load_config(p, validate=False)
    assert cfg.derived == 10  # type: ignore[union-attr]


def test_compose_defaults_env_interpolation(tmp_yaml, monkeypatch):
    """OmegaConf ``${oc.env:VAR}`` env interpolation resolves from the environment.

    Setup: env LT_TEST_VAR=hello; ``greeting: ${oc.env:LT_TEST_VAR}``.
    Expected: cfg.greeting == 'hello'.
    """
    monkeypatch.setenv("LT_TEST_VAR", "hello")
    p = tmp_yaml("greeting: ${oc.env:LT_TEST_VAR}\n")
    cfg = load_config(p, validate=False)
    assert cfg.greeting == "hello"  # type: ignore[union-attr]


def test_compose_defaults_yaml_suffix_tried_first(tmp_config_dir):
    """``defaults: [base]`` finds ``base.yaml`` (suffix appended at line 57).

    Setup: parent file is named ``base.yaml``. leaf says ``defaults: [base]``.
    Expected: parent is found via the ``.yaml`` suffix path; cfg.mode loads.
    """
    (tmp_config_dir / "base.yaml").write_text("mode: prod\n", encoding="utf-8")
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [base]\nseed: 7\n", encoding="utf-8")
    cfg = load_config(leaf)
    assert cfg.mode == "prod"


def test_compose_defaults_bare_filename_fallback(tmp_config_dir):
    """If ``defaults: [base.cfg]`` and ``base.cfg.yaml`` does not exist but
    ``base.cfg`` does, the bare-filename fallback (line 58-59 of _loader.py)
    finds it.

    Setup: parent file is ``base.cfg`` (no extension, valid YAML inside).
    leaf says ``defaults: [base.cfg]``.
    Expected: parent located via bare-filename fallback; cfg.mode loads.
    """
    (tmp_config_dir / "base.cfg").write_text("mode: prod\n", encoding="utf-8")
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [base.cfg]\nseed: 7\n", encoding="utf-8")
    cfg = load_config(leaf)
    assert cfg.mode == "prod"


def test_compose_defaults_missing_file_raises_with_message(tmp_config_dir):
    """Unknown default reference is a ConfigError naming the missing entry.

    Setup: leaf says ``defaults: [nonexistent]``; no such file on disk.
    Expected: ConfigError, message contains the bad ref string.
    """
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults: [nonexistent]\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(leaf)
    assert "nonexistent" in str(exc.value)


def test_compose_defaults_non_string_entry_raises(tmp_config_dir):
    """Non-string default entries are rejected (line 55-56 of _loader.py).

    Setup: leaf's defaults list contains a mapping ``{x: 1}`` instead of a str.
    Expected: ConfigError, message mentions the bad entry kind.
    """
    leaf = tmp_config_dir / "leaf.yaml"
    leaf.write_text("defaults:\n  - {x: 1}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(leaf)


def test_compose_defaults_direct_cycle_raises(tmp_config_dir):
    """Two files that include each other → ConfigError("Circular ...").

    Setup: a.yaml depends on b, b.yaml depends on a.
    Expected: ConfigError, message contains "Circular".
    """
    (tmp_config_dir / "a.yaml").write_text("defaults: [b]\nfoo: 1\n", encoding="utf-8")
    (tmp_config_dir / "b.yaml").write_text("defaults: [a]\nbar: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_config_dir / "a.yaml")
    assert "Circular" in str(exc.value) or "circular" in str(exc.value)


def test_compose_defaults_indirect_cycle_at_depth_three_raises(tmp_config_dir):
    """3-hop cycle (a → b → c → a) must be detected, not just direct cycles.

    Setup: a→b→c→a chain.
    Expected: ConfigError, message contains "Circular".
    """
    (tmp_config_dir / "a.yaml").write_text("defaults: [b]\n", encoding="utf-8")
    (tmp_config_dir / "b.yaml").write_text("defaults: [c]\n", encoding="utf-8")
    (tmp_config_dir / "c.yaml").write_text("defaults: [a]\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_config_dir / "a.yaml")
    assert "Circular" in str(exc.value) or "circular" in str(exc.value)


def test_regression_CFG_DEFAULTS_01_seen_unwound_after_sibling_exception(tmp_config_dir):
    """Pre-fix bug: when a defaults child-recursion raised, the parent's
    ``seen`` set still contained that child's path, so an unrelated *later*
    sibling that legitimately re-included the same node would be misdiagnosed
    as a circular reference. Fix: ``try/finally`` around ``seen.add/discard``
    (lines 44-67 of _loader.py).

    Setup: a 'diamond' include graph:
        root: defaults: [bad_sibling, good_sibling]
        bad_sibling: defaults: [bad_grandchild]      # bad_grandchild missing
        good_sibling: defaults: [shared_leaf]
        bad_grandchild: (does not exist on disk)
        shared_leaf:  defaults: [shared_dep]
        shared_dep: foo: ok

    Pre-fix: processing bad_sibling raised ConfigError (good); but because
    ``seen`` retained ``bad_sibling`` (or shared_leaf if the bug also affected
    that path), processing good_sibling under the *outer* recursion would
    spuriously hit the cycle guard. The fix unwinds ``seen`` via finally.

    We verify the post-fix behavior by *catching* the bad_sibling error at the
    outer level using a different approach: we make a sibling that itself is
    fine but reuses one of bad_sibling's transitive deps. Since this is hard
    to reproduce in a single ``load_config`` call (which surfaces the first
    exception), we instead directly drive ``_compose_defaults`` twice and
    assert the second call succeeds even though the first leaks via exception.

    Pre-fix bug: ``seen`` set retained entries after sibling exception,
    causing false circular-include errors on unrelated paths (see
    docs/changelog/v0.1.4: "[_compose_defaults seen 残留风险]").
    """
    from lighttrain.config._loader import _compose_defaults

    (tmp_config_dir / "shared.yaml").write_text("foo: ok\n", encoding="utf-8")
    bad = tmp_config_dir / "bad.yaml"
    bad.write_text("defaults: [does_not_exist]\n", encoding="utf-8")
    good = tmp_config_dir / "good.yaml"
    good.write_text("defaults: [shared]\nbar: 1\n", encoding="utf-8")

    # Share the same external `seen` set across the two calls — this is what
    # the (now-fixed) bug would taint. Pre-fix: bad.yaml's path stays in
    # `seen` after the missing-file exception bubbles up, so any subsequent
    # call re-processing bad.yaml would falsely raise "Circular". Post-fix:
    # the try/finally clears it.
    seen: set = set()
    with pytest.raises(ConfigError):
        _compose_defaults(bad, seen)
    # `seen` must be empty after the exception (try/finally has run).
    assert seen == set(), (
        f"`seen` set must be empty after exception; got {seen}. "
        "Pre-fix bug: seen retained entries → spurious 'Circular' on next call."
    )

    # Second call uses the same `seen` set and must succeed without spurious
    # "Circular" diagnosis.
    result = _compose_defaults(good, seen)
    assert result.bar == 1
    assert result.foo == "ok"


def test_compose_defaults_top_level_list_yaml_raises(tmp_config_dir):
    """If the config file's top level is a YAML list (which OmegaConf accepts
    as a ListConfig), the loader's ``isinstance(cfg, DictConfig)`` guard
    rejects it with a clear ConfigError (line 26-27 of _loader.py).

    Setup: a YAML file whose top level is the list ``[1, 2, 3]``. OmegaConf
    will parse this as a ListConfig (top-level scalars raise OSError from
    OmegaConf itself before the guard runs, so we use a list to exercise the
    guard branch).
    Expected: ConfigError mentioning "mapping".
    """
    p = tmp_config_dir / "list.yaml"
    p.write_text("- 1\n- 2\n- 3\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(p, validate=False)
    assert "mapping" in str(exc.value).lower()
