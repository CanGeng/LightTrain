"""Coverage-completion tests for ``lighttrain.observability.lineage.migration``.

Pins every previously-uncovered branch in the module (88% → ~100%):

* Line 60  — duplicate ``@migrate`` registration raises ``ValueError``.
* Lines 88-89 — BFS multi-hop path (intermediate ``seen.add`` + ``queue.append``).
* Line 109 — ``migrate_payload`` with unknown ``schema_kind`` (no SCHEMA_VERSION entry).
* Line 115 — ``migrate_payload`` payload already at target (short-circuit return).
* Line 122 — migration fn that omits ``schema_version``; framework patches it.
* Line 142 — ``migrate_file`` on non-existent path raises ``FileNotFoundError``.
* Lines 151-154 — ``migrate_file`` with unknown suffix: JSON-first fallback path.
* Lines 153-154 — ``migrate_file`` with unknown suffix: YAML fallback (JSON parse fails).
* Line 156 — ``migrate_file`` whose top-level parse is not a dict raises ``SchemaMigrationError``.
* Line 159 — ``migrate_file`` when payload already at target version (early return).
* Lines 199-200 — ``lineage_store`` failure is swallowed; migration still succeeds.
* Line 302 — ``rewrite_model_to_profiles_file`` on non-existent path raises ``FileNotFoundError``.
* Line 306 — ``rewrite_model_to_profiles_file`` with ``in_place=False`` returns ``changed`` without writing.
* Lines 336-338 — ``_migrate_checkpoint_manifest_03_to_04`` function body.

Suspected bugs: none found.
"""

from __future__ import annotations

import json
import logging

import pytest
import yaml

from lighttrain.observability.lineage.migration import (
    SchemaMigrationError,
    find_path,
    migrate,
    migrate_file,
    migrate_payload,
    registered_migrations,
    rewrite_model_to_profiles_file,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_BLOCK_YAML = """\
mode: lab
seed: 7

model:
  name: tiny_lm
  d_model: 128

data:
  name: simple
"""

_ALREADY_PROFILED_YAML = """\
model: default
model_profiles:
  default:
    name: tiny_lm
"""


class _BrokenStore:
    """Stub LineageStore whose upsert_node always raises."""

    def upsert_node(self, **kw):
        raise RuntimeError("simulated DB hiccup")

    def add_edge(self, *a, **kw):  # pragma: no cover — never reached from broken upsert
        pass


# ---------------------------------------------------------------------------
# @migrate decorator
# ---------------------------------------------------------------------------


def test_invariant_duplicate_migrate_registration_raises():
    """Registering the same (kind, from_, to_) key twice must raise ``ValueError``."""
    unique = "_test_cov_dup_xyz"

    @migrate(unique, from_="0.1", to_="0.2")
    def _first(old):
        return {**old, "schema_version": "0.2"}

    with pytest.raises(ValueError, match="duplicate migration registration"):
        @migrate(unique, from_="0.1", to_="0.2")
        def _second(old):
            return old


# ---------------------------------------------------------------------------
# find_path — multi-hop BFS (lines 88-89)
# ---------------------------------------------------------------------------


def test_invariant_find_path_multi_hop_bfs():
    """BFS must traverse intermediate nodes (lines 88-89: seen.add + queue.append)."""
    kind = "_test_cov_multihop_abc"

    @migrate(kind, from_="1.0", to_="1.1")
    def _h1(old):
        return {**old, "schema_version": "1.1"}

    @migrate(kind, from_="1.1", to_="1.2")
    def _h2(old):
        return {**old, "schema_version": "1.2"}

    path = find_path(kind, "1.0", "1.2")
    assert path == [("1.0", "1.1"), ("1.1", "1.2")]


def test_invariant_find_path_multi_hop_three_steps():
    """A three-hop chain is correctly resolved via BFS."""
    kind = "_test_cov_threehop_def"

    @migrate(kind, from_="a", to_="b")
    def _a_b(old):
        return {**old, "schema_version": "b"}

    @migrate(kind, from_="b", to_="c")
    def _b_c(old):
        return {**old, "schema_version": "c"}

    @migrate(kind, from_="c", to_="d")
    def _c_d(old):
        return {**old, "schema_version": "d"}

    path = find_path(kind, "a", "d")
    assert path == [("a", "b"), ("b", "c"), ("c", "d")]


# ---------------------------------------------------------------------------
# migrate_payload
# ---------------------------------------------------------------------------


def test_invariant_migrate_payload_unknown_schema_kind_raises(
):
    """``migrate_payload`` with an unknown ``schema_kind`` and no ``target`` must raise (line 109)."""
    with pytest.raises(SchemaMigrationError, match="no CURRENT schema_version"):
        migrate_payload({"schema_version": "0.1"}, schema_kind="_nonexistent_xyz_kind_999")


def test_invariant_migrate_payload_already_at_target_returns_copy():
    """``migrate_payload`` when payload is already at target returns a dict copy (line 115)."""
    payload = {"schema_version": "0.4", "mode": "lab"}
    result = migrate_payload(payload, schema_kind="config")
    assert result == payload
    assert result is not payload  # must be a fresh dict, not the same object


def test_invariant_migrate_payload_with_explicit_matching_target():
    """Explicit ``target`` matching current version also triggers the early-return path."""
    payload = {"schema_version": "0.4"}
    result = migrate_payload(payload, schema_kind="config", target="0.4")
    assert result["schema_version"] == "0.4"
    assert result is not payload


def test_pin_current_behavior_missing_schema_version_in_fn_output():
    """Pin: when a migration fn omits ``schema_version`` in its return dict,
    ``migrate_payload`` patches it to the declared ``to_`` version (line 122).

    This is a documented safety net — the behavior is intentional but worth pinning
    in case the safety net is ever inadvertently removed.
    """
    kind = "_test_cov_forget_sv_ghi"

    @migrate(kind, from_="0.1", to_="0.2")
    def _forgets_sv(old):
        new = dict(old)
        new["extra"] = "patched"
        # deliberately does NOT set new["schema_version"]
        return new

    result = migrate_payload({"schema_version": "0.1"}, schema_kind=kind, target="0.2")
    assert result["schema_version"] == "0.2", (
        "framework should auto-patch missing schema_version to the declared to_ version"
    )
    assert result["extra"] == "patched"


# ---------------------------------------------------------------------------
# migrate_file — error paths
# ---------------------------------------------------------------------------


def test_invariant_migrate_file_raises_on_missing_file(tmp_path):
    """``migrate_file`` on a non-existent path must raise ``FileNotFoundError`` (line 142)."""
    with pytest.raises(FileNotFoundError):
        migrate_file(tmp_path / "ghost.yaml", schema_kind="config")


def test_invariant_migrate_file_raises_on_non_dict_payload(tmp_path):
    """``migrate_file`` whose parsed content is not a dict must raise ``SchemaMigrationError`` (line 156)."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(SchemaMigrationError, match="did not parse to a mapping"):
        migrate_file(p, schema_kind="config")


def test_invariant_migrate_file_raises_on_non_dict_yaml_payload(tmp_path):
    """``migrate_file`` whose YAML content is a bare list must raise ``SchemaMigrationError``."""
    p = tmp_path / "bad.yaml"
    p.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(SchemaMigrationError, match="did not parse to a mapping"):
        migrate_file(p, schema_kind="config")


def test_invariant_migrate_file_early_return_when_already_current(tmp_path):
    """``migrate_file`` returns immediately without backup when payload is already current (line 159)."""
    p = tmp_path / "current.json"
    p.write_text(json.dumps({"schema_version": "0.4", "backend": "x"}), encoding="utf-8")
    result = migrate_file(p, schema_kind="artifact_header")
    assert result["schema_version"] == "0.4"
    bak = p.with_suffix(".json.pre-migration-bak")
    assert not bak.exists(), "no backup should be written when payload was already current"


def test_invariant_migrate_file_early_return_yaml(tmp_path):
    """``migrate_file`` YAML variant: no write/backup when already at target."""
    p = tmp_path / "current.yaml"
    p.write_text(yaml.safe_dump({"schema_version": "0.4", "mode": "lab"}), encoding="utf-8")
    original_mtime = p.stat().st_mtime
    result = migrate_file(p, schema_kind="config")
    assert result["schema_version"] == "0.4"
    assert p.stat().st_mtime == original_mtime, "file should not be touched"


# ---------------------------------------------------------------------------
# migrate_file — unknown-suffix fallback paths (lines 151-154)
# ---------------------------------------------------------------------------


def test_invariant_migrate_file_unknown_suffix_json_first_fallback(tmp_path):
    """Unknown extension: JSON tried first (lines 151-152); succeeds when content is valid JSON."""
    p = tmp_path / "manifest.dat"
    p.write_text(json.dumps({"schema_version": "0.3", "backend": "x"}), encoding="utf-8")
    result = migrate_file(p, schema_kind="artifact_header")
    assert result["schema_version"] == "0.4"
    assert result["framework_version"].startswith("torch:")


def test_invariant_migrate_file_unknown_suffix_yaml_fallback(tmp_path):
    """Unknown extension: when JSON parse fails, YAML fallback (lines 153-154) is attempted."""
    p = tmp_path / "manifest.dat"
    # YAML that is not valid JSON (contains a bare key)
    p.write_text("schema_version: '0.3'\nfiles:\n  - ckpt.pt\n", encoding="utf-8")
    result = migrate_file(p, schema_kind="checkpoint_manifest")
    assert result["schema_version"] == "0.4"


def test_invariant_migrate_file_yml_extension_parsed_as_yaml(tmp_path):
    """Files with ``.yml`` extension must be parsed as YAML (not via JSON fallback)."""
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump({"schema_version": "0.3"}), encoding="utf-8")
    result = migrate_file(p, schema_kind="checkpoint_manifest")
    assert result["schema_version"] == "0.4"


# ---------------------------------------------------------------------------
# migrate_file — backup + in_place flags
# ---------------------------------------------------------------------------


def test_invariant_migrate_file_no_backup_flag(tmp_path):
    """``backup=False`` skips backup creation even when migration is needed."""
    p = tmp_path / "hdr.json"
    p.write_text(json.dumps({"schema_version": "0.3", "backend": "x"}), encoding="utf-8")
    migrate_file(p, schema_kind="artifact_header", backup=False)
    bak = p.with_suffix(".json.pre-migration-bak")
    assert not bak.exists()


def test_invariant_migrate_file_in_place_false_does_not_write(tmp_path):
    """``in_place=False`` returns the migrated payload without modifying the file."""
    original = json.dumps({"schema_version": "0.3", "backend": "x"})
    p = tmp_path / "hdr.json"
    p.write_text(original, encoding="utf-8")
    result = migrate_file(p, schema_kind="artifact_header", in_place=False)
    assert result["schema_version"] == "0.4"
    assert p.read_text(encoding="utf-8") == original, "file must remain unchanged with in_place=False"
    bak = p.with_suffix(".json.pre-migration-bak")
    assert not bak.exists()


# ---------------------------------------------------------------------------
# migrate_file — lineage store failure is soft (lines 199-200)
# ---------------------------------------------------------------------------


def test_invariant_lineage_store_failure_swallowed_migration_succeeds(tmp_path, caplog):
    """A broken lineage store must not block the migration (lines 199-200).

    The warning is emitted and the migrated payload is returned.
    """
    p = tmp_path / "hdr.json"
    p.write_text(json.dumps({"schema_version": "0.3", "backend": "x"}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.lineage.migration"):
        result = migrate_file(p, schema_kind="artifact_header", lineage_store=_BrokenStore())

    assert result["schema_version"] == "0.4", "migration must still succeed despite lineage DB failure"
    assert any("failed to record migration lineage" in r.message for r in caplog.records)


def test_invariant_lineage_add_edge_failure_swallowed(tmp_path, caplog):
    """Even if ``add_edge`` (not ``upsert_node``) fails, migration succeeds."""

    class _FailOnEdge:
        def upsert_node(self, **kw):
            return "fake-node-id"

        def add_edge(self, *a, **kw):
            raise RuntimeError("edge write failed")

    p = tmp_path / "hdr2.json"
    p.write_text(json.dumps({"schema_version": "0.3", "backend": "x"}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.lineage.migration"):
        result = migrate_file(p, schema_kind="artifact_header", lineage_store=_FailOnEdge())
    assert result["schema_version"] == "0.4"
    assert any("failed to record migration lineage" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# rewrite_model_to_profiles_file — error and no-op paths
# ---------------------------------------------------------------------------


def test_invariant_rewrite_raises_on_missing_file(tmp_path):
    """``rewrite_model_to_profiles_file`` on missing path raises ``FileNotFoundError`` (line 302)."""
    with pytest.raises(FileNotFoundError):
        rewrite_model_to_profiles_file(tmp_path / "ghost.yaml")


def test_invariant_rewrite_in_place_false_returns_changed_without_writing(tmp_yaml):
    """``in_place=False`` returns ``True`` but leaves the file untouched (line 306)."""
    p = tmp_yaml(_BLOCK_YAML)
    before = p.read_text(encoding="utf-8")
    changed = rewrite_model_to_profiles_file(p, in_place=False)
    assert changed is True
    assert p.read_text(encoding="utf-8") == before, "file must not be modified when in_place=False"


def test_invariant_rewrite_no_change_returns_false_without_writing(tmp_yaml):
    """When the YAML is already in profiles form, ``changed=False`` is returned (line 306)."""
    p = tmp_yaml(_ALREADY_PROFILED_YAML)
    changed = rewrite_model_to_profiles_file(p, in_place=False)
    assert changed is False


# ---------------------------------------------------------------------------
# Seed migrations: checkpoint_manifest (lines 334-338)
# ---------------------------------------------------------------------------


def test_invariant_checkpoint_manifest_migration_bumps_version():
    """``_migrate_checkpoint_manifest_03_to_04`` updates schema_version (lines 334-338)."""
    result = migrate_payload(
        {"schema_version": "0.3", "files": ["model.pt", "optimizer.pt"]},
        schema_kind="checkpoint_manifest",
    )
    assert result["schema_version"] == "0.4"
    assert result["files"] == ["model.pt", "optimizer.pt"], "payload keys must be preserved"


def test_invariant_checkpoint_manifest_migration_via_file(tmp_path):
    """End-to-end file migration for checkpoint_manifest schema."""
    p = tmp_path / "manifest.json"
    original = {"schema_version": "0.3", "files": ["model.pt"]}
    p.write_text(json.dumps(original), encoding="utf-8")
    result = migrate_file(p, schema_kind="checkpoint_manifest")
    assert result["schema_version"] == "0.4"
    bak = p.with_suffix(".json.pre-migration-bak")
    assert bak.exists(), "backup must be written for checkpoint_manifest migration"


# ---------------------------------------------------------------------------
# registered_migrations correctness
# ---------------------------------------------------------------------------


def test_invariant_registered_migrations_returns_copy():
    """``registered_migrations()`` returns an independent copy, not the live registry."""
    snap1 = registered_migrations()
    kind = "_test_cov_reg_copy_jkl"

    @migrate(kind, from_="x", to_="y")
    def _dummy(old):
        return old

    snap2 = registered_migrations()
    assert (kind, "x", "y") not in snap1
    assert (kind, "x", "y") in snap2


# ---------------------------------------------------------------------------
# Parametrised: migrate_payload with explicit target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_kind,extra_key,extra_val", [
    ("config", "mode", "lab"),
    ("artifact_header", "framework_version", "torch:unknown"),
    ("checkpoint_manifest", "files", []),
])
def test_invariant_seed_migrations_explicit_target(schema_kind, extra_key, extra_val):
    """Each seed migration 0.3→0.4 produces the expected extra key when ``target`` is explicit."""
    payload = {"schema_version": "0.3"}
    result = migrate_payload(payload, schema_kind=schema_kind, target="0.4")
    assert result["schema_version"] == "0.4"
    if schema_kind != "checkpoint_manifest":
        # checkpoint_manifest migration only sets schema_version (no extra required keys)
        assert extra_key in result
