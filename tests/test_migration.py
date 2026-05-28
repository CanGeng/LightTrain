"""Schema migration registry + file driver — DESIGN §12.6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lighttrain.lineage import LineageStore
from lighttrain.lineage.migration import (
    SchemaMigrationError,
    find_path,
    migrate_file,
    migrate_payload,
    registered_migrations,
)


def test_registered_migrations_include_seed_set():
    keys = registered_migrations()
    assert ("config", "0.3", "0.4") in keys
    assert ("artifact_header", "0.3", "0.4") in keys
    assert ("checkpoint_manifest", "0.3", "0.4") in keys


def test_find_path_identity_and_one_hop():
    assert find_path("config", "0.4", "0.4") == []
    assert find_path("config", "0.3", "0.4") == [("0.3", "0.4")]


def test_find_path_unknown_raises():
    with pytest.raises(SchemaMigrationError):
        find_path("config", "0.1", "0.4")


def test_migrate_payload_renames_ema_start_and_sets_mode():
    old = {"schema_version": "0.3", "ema": {"start": 100, "decay": 0.99}}
    new = migrate_payload(old, schema_kind="config")
    assert new["schema_version"] == "0.4"
    assert "start" not in new["ema"]
    assert new["ema"]["start_step"] == 100
    assert new["mode"] == "lab"


def test_migrate_file_writes_backup_and_lineage_edge(tmp_path):
    """F4 acceptance — DESIGN §25.3."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "0.3", "mode": "prod"}),
                    encoding="utf-8")
    store = LineageStore(tmp_path / "l.sqlite")
    migrated = migrate_file(path, schema_kind="config", lineage_store=store)
    assert migrated["schema_version"] == "0.4"
    assert (path.with_suffix(".yaml.pre-migration-bak")).exists()

    # Lineage edge written.
    nodes = list(store.iter_nodes(kind="config"))
    assert len(nodes) == 2
    edges = store.edges_from(nodes[0]["id"]) + store.edges_to(nodes[1]["id"])
    assert any(e["kind"] == "migrated_from" for e in edges)


def test_migrate_artifact_header_adds_framework_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "0.3", "backend": "x"}),
                    encoding="utf-8")
    new = migrate_file(path, schema_kind="artifact_header")
    assert new["framework_version"].startswith("torch:")
    assert new["schema_version"] == "0.4"
