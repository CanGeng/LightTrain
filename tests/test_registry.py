"""Tests for lighttrain.registry."""

from __future__ import annotations

import pytest

from lighttrain.registry import (
    KNOWN_CATEGORIES,
    Registry,
    NotRegisteredError,
    RegistryConflictError,
    UnknownCategoryError,
    categories,
    contains,
    get,
    list_entries,
    register,
    register_category,
    unregister,
)


def test_known_categories_includes_core_eight():
    core_eight = {
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "dataset",
        "processor",
        "collator",
        "sampler",
    }
    assert core_eight.issubset(set(KNOWN_CATEGORIES))


def test_categories_is_sorted_snapshot():
    cats = categories()
    assert cats == sorted(cats)
    assert "callback" in cats
    assert "prep_node" in cats


def test_decorator_registers(clean_registry):
    @register("model", "test_tiny_lm")
    class TinyLM:
        pass

    assert get("model", "test_tiny_lm") is TinyLM
    assert "test_tiny_lm" in list_entries("model")
    assert contains("model", "test_tiny_lm")


def test_function_form_registers(clean_registry):
    class Foo:
        pass

    register("loss", "ce_v2", Foo)
    assert get("loss", "ce_v2") is Foo


def test_decorator_and_function_form_equivalent(clean_registry):
    class A:
        pass

    class B:
        pass

    register("loss", "x", A)
    register("loss", "y")(B)
    assert get("loss", "x") is A
    assert get("loss", "y") is B


def test_duplicate_raises_conflict(clean_registry):
    register("model", "dup", lambda: None)
    with pytest.raises(RegistryConflictError):
        register("model", "dup", lambda: None)


def test_force_overrides(clean_registry):
    class A:
        pass

    class B:
        pass

    register("model", "ovr", A)
    register("model", "ovr", B, force=True)
    assert get("model", "ovr") is B


def test_unknown_category_register_raises(clean_registry):
    with pytest.raises(UnknownCategoryError):
        register("not_a_category", "x", lambda: None)


def test_unknown_category_get_raises():
    with pytest.raises(UnknownCategoryError):
        get("nope", "x")


def test_register_category_then_use(clean_registry):
    register_category("my_plugin_category")
    assert "my_plugin_category" in categories()

    @register("my_plugin_category", "thing")
    class Thing:
        pass

    assert get("my_plugin_category", "thing") is Thing


def test_register_category_idempotent(clean_registry):
    register_category("p1")
    register_category("p1")
    assert categories().count("p1") == 1


def test_not_registered_raises():
    with pytest.raises(NotRegisteredError):
        get("model", "definitely_missing_xyz")


def test_categories_isolate_same_name(clean_registry):
    class M:
        pass

    class L:
        pass

    register("model", "shared", M)
    register("loss", "shared", L)
    assert get("model", "shared") is M
    assert get("loss", "shared") is L


def test_unregister_removes(clean_registry):
    register("model", "tmp", lambda: None)
    assert contains("model", "tmp")
    unregister("model", "tmp")
    assert not contains("model", "tmp")


def test_unregister_missing_raises(clean_registry):
    with pytest.raises(NotRegisteredError):
        unregister("model", "never_added")


def test_list_entries_empty_initially(clean_registry):
    # After snapshot+restore, the category should be empty for our test entries.
    assert "this_should_not_exist" not in list_entries("model")


def test_list_entries_unknown_category_raises():
    with pytest.raises(UnknownCategoryError):
        list_entries("xxx")


def test_isolated_registry_instance():
    """Constructing a fresh Registry doesn't touch the global one."""
    r = Registry()
    r.register("model", "iso", object)
    assert r.get("model", "iso") is object
    with pytest.raises(NotRegisteredError):
        get("model", "iso")  # global registry is untouched


def test_clear_resets_category(clean_registry):
    reg = clean_registry
    reg.register("model", "a", object)
    reg.register("model", "b", object)
    reg.clear("model")
    assert reg.list("model") == []


def test_clear_all(clean_registry):
    reg = clean_registry
    reg.register("model", "a", object)
    reg.register("loss", "b", object)
    reg.clear()
    assert reg.list("model") == []
    assert reg.list("loss") == []


def test_snapshot_restore_round_trip(clean_registry):
    reg = clean_registry
    reg.register("model", "snap", object)
    snap = reg.snapshot()
    reg.unregister("model", "snap")
    assert not reg.contains("model", "snap")
    reg.restore(snap)
    assert reg.contains("model", "snap")


# ---------------------------------------------------------------------------
# Content-identity idempotency (ISSUE-2 / root cause B): registering the same
# logical component twice — e.g. one file imported under two module identities,
# or the exact same object — is a no-op, not a conflict. Genuinely different
# definitions with the same name still raise.
# ---------------------------------------------------------------------------

def _load_twice(tmp_path, body: str, name: str):
    """Load one .py file under two distinct module identities (as the
    user_modules path-import and a _target_ dotted-import would)."""
    import importlib.util

    f = tmp_path / "lt_idem_src.py"
    f.write_text(body)
    mods = []
    for modname in ("lt_idem_A", "lt_idem_B"):
        spec = importlib.util.spec_from_file_location(modname, f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods.append(m)
    return mods


def test_same_file_two_module_identities_is_noop(clean_registry, tmp_path):
    body = (
        "from lighttrain.registry import register\n"
        "@register('prep_node', '_idem_node')\n"
        "class IdemNode:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    # No RegistryConflictError despite two @register executions on two distinct
    # class objects produced from the same physical file.
    _load_twice(tmp_path, body, "_idem_node")
    assert contains("prep_node", "_idem_node")


def test_same_object_reregistered_is_noop(clean_registry):
    class Foo:
        def run(self):  # gives the class a code object to fingerprint
            return 1

    register("model", "_idem_obj", Foo)
    # Re-registering the identical object must not raise.
    register("model", "_idem_obj", Foo)
    assert get("model", "_idem_obj") is Foo


def test_different_class_same_name_still_raises(clean_registry):
    class A:
        def run(self):
            return 1

    class B:  # different qualname + line → genuine conflict
        def run(self):
            return 2

    register("model", "_idem_conflict", A)
    with pytest.raises(RegistryConflictError):
        register("model", "_idem_conflict", B)


def test_force_still_overrides_after_idempotency(clean_registry):
    class A:
        def run(self):
            return 1

    class B:
        def run(self):
            return 2

    register("model", "_idem_force", A)
    register("model", "_idem_force", B, force=True)
    assert get("model", "_idem_force") is B
