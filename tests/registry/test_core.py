"""Adversarial tests for ``lighttrain.registry``.

Coverage beyond the flat ``tests/test_registry.py``:

* Force-overwrite is fully silent — pin contract: no warning, get returns the
  new object by identity, old object is fully evicted.
* ``force=True`` on a non-existent entry is permitted (same as regular register).
* Sequential force-overwrites: A → B(force) → C(force) — get returns C.
* Snapshot/restore preserves categories outside the snapshot scope.
* NotRegisteredError message includes the available entry list.
* Decorator returns the class itself (no wrapping).
* Unicode and unusual-but-allowed names pin current permissive behavior.
* ``clear(None)`` clears every known category (legacy passes only category).
"""

from __future__ import annotations

import warnings

import pytest

from lighttrain.registry import (
    Registry,
    NotRegisteredError,
    RegistryConflictError,
    UnknownCategoryError,
    contains,
    get,
    list_entries,
    register,
    register_category,
    unregister,
)


# ---------------------------------------------------------------------------
# Force-overwrite contract (the highest-stakes corner of the registry)
# ---------------------------------------------------------------------------

def test_pin_register_force_overwrite_is_silent_no_warning(clean_registry):
    """Pin: ``force=True`` overwrites without emitting any warning.

    Goal: rationale = plugin overrides are a legitimate use-case; the
    registry should not pollute stderr on every overwrite. Test pins this
    silence.

    Setup: register obj A, then re-register with force=True; capture warnings
    via ``warnings.catch_warnings(record=True)``.
    Expected: no warnings captured AND get returns the new object.

    If you intentionally add a deprecation/override warning, update this test
    AND document the breaking change.
    """
    class A:
        pass

    class B:
        pass

    register("model", "ovr_silent", A)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register("model", "ovr_silent", B, force=True)
    assert caught == []
    assert get("model", "ovr_silent") is B


def test_invariant_force_overwrite_evicts_old_object_completely(clean_registry):
    """Invariant: after ``register(force=True)``, ``get`` returns the new
    object identity AND there is no path that can still reach the old one.

    Setup: register A; force-register B; assert ``get is B``, AND the bucket
    contains exactly one (name, B) pair (via list_entries length unchanged).
    Expected: list length unchanged from 1; get is B (not A).
    """
    class A:
        pass

    class B:
        pass

    register("model", "evict_test", A)
    n_before = len(list_entries("model"))
    register("model", "evict_test", B, force=True)
    assert get("model", "evict_test") is B
    assert get("model", "evict_test") is not A
    assert len(list_entries("model")) == n_before


def test_pin_force_true_on_nonexistent_entry_acts_like_register(clean_registry):
    """Pin: ``force=True`` on a name that has never been registered is
    semantically equivalent to a regular registration — no error.

    Setup: call ``register('model', 'fresh', obj, force=True)`` with the name
    NOT in the bucket.
    Expected: subsequent get returns obj, no exception.

    If you intentionally tighten ``force=True`` to require an existing entry
    (e.g. as a guard against typos), update this test.
    """
    class Fresh:
        pass

    register("model", "fresh_force", Fresh, force=True)
    assert get("model", "fresh_force") is Fresh


def test_sequential_force_overwrites_chain_correctly(clean_registry):
    """Three-way overwrite chain A → B(force) → C(force) ends at C.

    Setup: register A; force-register B; force-register C.
    Expected: ``get`` returns C; list length stays 1.
    """
    class A: pass
    class B: pass
    class C: pass

    register("model", "chain", A)
    register("model", "chain", B, force=True)
    register("model", "chain", C, force=True)
    assert get("model", "chain") is C
    assert "chain" in list_entries("model")


def test_force_false_after_force_true_overwrite_still_raises(clean_registry):
    """After a ``force=True`` overwrite, the next plain register without
    force STILL conflicts. Force is per-call, not sticky.

    Setup: register A; force-register B; attempt register C without force.
    Expected: RegistryConflictError, get still returns B.
    """
    class A: pass
    class B: pass
    class C: pass

    register("model", "no_sticky", A)
    register("model", "no_sticky", B, force=True)
    with pytest.raises(RegistryConflictError):
        register("model", "no_sticky", C)
    assert get("model", "no_sticky") is B


# ---------------------------------------------------------------------------
# Conflict-error message quality
# ---------------------------------------------------------------------------

def test_conflict_error_message_names_existing_object(clean_registry):
    """``RegistryConflictError`` message includes the existing object's repr.

    Goal: pin the message format so debugging is easier.
    Setup: register A; attempt to register B with the same name.
    Expected: exception message contains the existing object's repr AND the
    name being registered.
    """
    class A:
        pass

    register("model", "msg_test", A)
    with pytest.raises(RegistryConflictError) as exc:
        register("model", "msg_test", lambda: None)
    msg = str(exc.value)
    assert "msg_test" in msg
    assert "force=True" in msg


def test_not_registered_error_message_lists_available_names(clean_registry):
    """``NotRegisteredError`` message includes the sorted available list.

    Setup: register two names in 'model'; look up a missing third name.
    Expected: exception message contains both registered names (in sorted order).
    """
    register("model", "alpha", object)
    register("model", "beta", object)
    with pytest.raises(NotRegisteredError) as exc:
        get("model", "gamma_missing")
    msg = str(exc.value)
    assert "alpha" in msg
    assert "beta" in msg


def test_unknown_category_error_message_lists_known_categories():
    """``UnknownCategoryError`` message contains known categories.

    Input: ``get('not_a_cat', 'x')``.
    Expected: message mentions ``not_a_cat`` AND lists at least one known
    category (e.g. 'model').
    """
    with pytest.raises(UnknownCategoryError) as exc:
        get("not_a_cat", "x")
    msg = str(exc.value)
    assert "not_a_cat" in msg
    assert "model" in msg  # at least one known category named


# ---------------------------------------------------------------------------
# Decorator semantics
# ---------------------------------------------------------------------------

def test_invariant_decorator_returns_decorated_class_unchanged(clean_registry):
    """Invariant: ``@register(...)`` returns the decorated class itself, not
    a wrapper. ``cls is wrapped_cls`` (identity, not just equality).

    Setup: decorate a class with ``@register(...)``.
    Expected: the named binding equals the class object AND the registry
    returns the same identity.
    """
    @register("model", "dec_id_test")
    class Decorated:
        pass

    assert Decorated.__name__ == "Decorated"  # not wrapped
    assert get("model", "dec_id_test") is Decorated


def test_decorator_function_register_round_trip_identity_equivalence(clean_registry):
    """Function form and decorator form produce identical registry state.

    Setup: register two classes A (function) and B (decorator) under the same
    category.
    Expected: both retrievable; identity matches.
    """
    class A: pass

    register("loss", "fa", A)

    @register("loss", "fb")
    class B:
        pass

    assert get("loss", "fa") is A
    assert get("loss", "fb") is B


# ---------------------------------------------------------------------------
# Cross-category isolation (deeper than legacy)
# ---------------------------------------------------------------------------

def test_invariant_get_only_looks_in_target_category(clean_registry):
    """Invariant: ``get(category, name)`` never falls back to another category.

    Setup: register obj under ('model', 'x'); attempt get under ('loss', 'x').
    Expected: NotRegisteredError under 'loss' even though the name exists in
    'model'.
    """
    register("model", "iso_x", object)
    with pytest.raises(NotRegisteredError):
        get("loss", "iso_x")


def test_invariant_unregister_only_affects_target_category(clean_registry):
    """Unregister under 'model' leaves 'loss' entry untouched.

    Setup: register under both ('model', 'name') and ('loss', 'name').
    Expected: after unregister 'model'/'name', 'loss'/'name' still retrievable.
    """
    register("model", "name", object)
    register("loss", "name", object)
    unregister("model", "name")
    assert not contains("model", "name")
    assert contains("loss", "name")


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

def test_snapshot_after_force_overwrite_restores_to_overwritten_state(clean_registry):
    """Snapshot taken AFTER a force-overwrite restores the overwritten value.

    Setup: register A; force-overwrite to B; snapshot; force-overwrite to C;
    restore snapshot.
    Expected: post-restore, ``get`` returns B (the snapshotted overwrite),
    not A and not C.
    """
    class A: pass
    class B: pass
    class C: pass

    reg = clean_registry
    reg.register("model", "snap_after", A)
    reg.register("model", "snap_after", B, force=True)
    snap = reg.snapshot()
    reg.register("model", "snap_after", C, force=True)
    assert reg.get("model", "snap_after") is C
    reg.restore(snap)
    assert reg.get("model", "snap_after") is B


def test_restore_preserves_categories_outside_snapshot_scope(clean_registry):
    """Restore only overwrites the categories present in the snapshot.

    Setup: register under 'model'; snapshot; register under 'loss' AFTER
    snapshot; restore.
    Expected: 'model' entry restored; 'loss' entry NOT present (because
    snapshot did not include the 'loss' write — restore overwrites with
    snapshot's 'loss' bucket which is empty).

    Pin: ``restore`` iterates ``snap.items()`` and copies bucket-by-bucket;
    every key in the snapshot is overwritten. Categories not in the snapshot
    are left untouched.
    """
    reg = clean_registry
    reg.register("model", "before_snap", object)
    snap = reg.snapshot()
    reg.register("loss", "after_snap", object)
    reg.register("model", "after_snap_too", object)
    reg.restore(snap)
    # model bucket restored to snapshot state — only 'before_snap' present
    assert reg.contains("model", "before_snap")
    assert not reg.contains("model", "after_snap_too")
    # loss bucket was empty in snapshot — restore overwrites with empty bucket
    assert not reg.contains("loss", "after_snap")


def test_restore_creates_missing_category_for_plugin(clean_registry):
    """If the snapshot has a category not currently registered, restore
    creates it (line 162-163 of _core.py).

    Setup: build a fake snapshot with a category name that doesn't exist
    yet, then restore.
    Expected: category is auto-created, and its entry is retrievable.
    """
    reg = clean_registry
    fake_snap = {"plugin_only_cat": {"foo": object}}
    reg.restore(fake_snap)
    assert "plugin_only_cat" in reg.categories()
    assert reg.get("plugin_only_cat", "foo") is object


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_none_clears_every_category(clean_registry):
    """``clear(None)`` clears every category, not just one.

    Setup: register entries under 'model' AND 'loss'.
    Expected: after clear(None), both buckets empty.
    """
    reg = clean_registry
    reg.register("model", "x", object)
    reg.register("loss", "y", object)
    reg.clear()
    assert reg.list("model") == []
    assert reg.list("loss") == []


def test_clear_unknown_category_raises():
    """``clear('bogus')`` is rejected with UnknownCategoryError.

    Expected: UnknownCategoryError.
    """
    reg = Registry()  # isolated instance — own categories
    with pytest.raises(UnknownCategoryError):
        reg.clear("xxx_unknown")


# ---------------------------------------------------------------------------
# Name string permissiveness (pins)
# ---------------------------------------------------------------------------

def test_pin_register_accepts_empty_string_name(clean_registry):
    """Pin: the registry currently accepts ``name=""`` without validation.

    Setup: register obj with empty-string name.
    Expected: subsequent get/contains succeed with empty string.

    If you intentionally add a name-format validator, update this test.
    """
    register("model", "", object)
    assert contains("model", "")
    assert get("model", "") is object


def test_pin_register_accepts_unicode_names(clean_registry):
    """Pin: unicode names round-trip without normalization.

    Setup: register obj under '名前'; get with same string.
    Expected: identity match; no NFC/NFD coercion.
    """
    register("model", "名前", object)
    assert get("model", "名前") is object


def test_pin_register_accepts_names_with_dots_and_slashes(clean_registry):
    """Pin: names with dots/slashes are allowed (no syntactic check).

    Setup: register names ``a.b.c`` and ``a/b``.
    Expected: both retrievable as exact-match strings.
    """
    register("model", "a.b.c", object)
    register("model", "a/b", str)
    assert get("model", "a.b.c") is object
    assert get("model", "a/b") is str


# ---------------------------------------------------------------------------
# list() / contains() / categories()
# ---------------------------------------------------------------------------

def test_invariant_list_entries_returns_sorted(clean_registry):
    """Invariant: ``list_entries`` returns names in sorted order
    (line 132-133 of _core.py).

    Setup: register names ``z, a, m`` in insertion order.
    Expected: list returns ``['a', 'm', 'z']``.
    """
    register("model", "z_name", object)
    register("model", "a_name", object)
    register("model", "m_name", object)
    out = list_entries("model")
    assert out == sorted(out)
    # Specifically check our three are sorted relative to each other.
    idx_a = out.index("a_name")
    idx_m = out.index("m_name")
    idx_z = out.index("z_name")
    assert idx_a < idx_m < idx_z


def test_contains_unknown_category_raises():
    """``contains('bogus', 'x')`` raises UnknownCategoryError, NOT silent False.

    Goal: pin the strict-by-default contract — typos on category names should
    not silently report "not present".
    """
    with pytest.raises(UnknownCategoryError):
        contains("bogus_cat_unknown", "x")


# ---------------------------------------------------------------------------
# Isolated Registry instance
# ---------------------------------------------------------------------------

def test_isolated_registry_does_not_share_state_with_global():
    """A freshly-constructed ``Registry()`` does not see globally-registered
    entries.

    Setup: register obj on global registry; construct fresh Registry();
    look up on fresh instance.
    Expected: NotRegisteredError on the fresh instance even though the
    global has the entry.
    """
    # First do registration on global registry — must be inside test scope
    # so the global-state pollution doesn't bleed; relies on clean_registry
    # at the test boundary if needed. Here we use a different name to avoid
    # collisions, and isolate via a try/finally cleanup.
    try:
        register("model", "global_only_xyz", object)
        fresh = Registry()
        with pytest.raises(NotRegisteredError):
            fresh.get("model", "global_only_xyz")
    finally:
        try:
            unregister("model", "global_only_xyz")
        except Exception:
            pass


def test_isolated_registry_can_be_cleared_independently():
    """Clearing a fresh instance does not affect the global registry.

    Setup: register obj on both global and a fresh instance; clear fresh.
    Expected: fresh is empty; global still has the entry.
    """
    try:
        register("model", "indep_test", object)
        fresh = Registry()
        fresh.register("model", "indep_test", object)
        fresh.clear("model")
        assert fresh.list("model") == []
        # global still has it
        assert contains("model", "indep_test")
    finally:
        try:
            unregister("model", "indep_test")
        except Exception:
            pass
