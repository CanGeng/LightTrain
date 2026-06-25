"""Edge-case tests for ``lighttrain.lab.decision_record``.

Pins the JSONL-backed decision log end to end:

* persistence round-trip (add → reload from disk);
* ``_load`` skips blank / malformed / wrong-schema lines;
* ``add`` ids increment; ``accept`` / ``deprecate`` / ``_set_status`` (incl.
  non-matching id no-op);
* ``render_markdown`` empty + every optional field + unknown-status icon;
* ``write_markdown`` default ``.md`` path and explicit path; ``__len__``.
"""

from __future__ import annotations

from lighttrain.lab.decision_record import DecisionEntry, DecisionRecord


def test_add_assigns_incrementing_ids_and_persists(tmp_path):
    dr = DecisionRecord(tmp_path / "decisions.jsonl")
    assert dr.add("first") == 0
    assert dr.add("second") == 1
    assert len(dr) == 2
    # A fresh instance over the same file reloads both entries.
    assert len(DecisionRecord(tmp_path / "decisions.jsonl")) == 2


def test_load_skips_blank_malformed_and_wrong_schema_lines(tmp_path):
    p = tmp_path / "decisions.jsonl"
    good = '{"id": 0, "title": "ok"}'
    p.write_text("\n".join(["", good, "{not json", '{"no_id_field": true}']) + "\n", encoding="utf-8")
    dr = DecisionRecord(p)
    # Only the well-formed, schema-valid line survives.
    assert len(dr) == 1


def test_accept_sets_status(tmp_path):
    dr = DecisionRecord(tmp_path / "d.jsonl")
    did = dr.add("x", status="proposed")
    dr.accept(did)
    assert DecisionRecord(tmp_path / "d.jsonl")._entries[0].status == "accepted"


def test_deprecate_sets_status_and_superseded_by(tmp_path):
    dr = DecisionRecord(tmp_path / "d.jsonl")
    a = dr.add("old")
    b = dr.add("new")
    dr.deprecate(a, superseded_by=b)
    reloaded = DecisionRecord(tmp_path / "d.jsonl")._entries[0]
    assert reloaded.status == "deprecated"
    assert reloaded.superseded_by == b


def test_set_status_nonexistent_id_is_noop(tmp_path):
    dr = DecisionRecord(tmp_path / "d.jsonl")
    dr.add("x")
    dr.accept(999)  # no entry with this id → silent no-op (still saves)
    assert dr._entries[0].status == "proposed"


def test_render_markdown_empty():
    dr = DecisionRecord.__new__(DecisionRecord)  # avoid disk
    dr._entries = []
    md = dr.render_markdown()
    assert "_No decisions recorded yet._" in md


def test_render_markdown_includes_all_optional_fields_and_icons():
    dr = DecisionRecord.__new__(DecisionRecord)
    dr._entries = [
        DecisionEntry(
            id=0, title="T", context="ctx", decision="dec",
            consequences="con", status="accepted", superseded_by=1,
        ),
        DecisionEntry(id=1, title="U", status="mystery"),  # unknown status → ❓
    ]
    md = dr.render_markdown()
    assert "## DR-000: T ✅" in md
    assert "**Context:** ctx" in md
    assert "**Decision:** dec" in md
    assert "**Consequences:** con" in md
    assert "_Superseded by DR-001._" in md
    assert "❓" in md  # unknown-status fallback icon


def test_write_markdown_default_path_uses_md_suffix(tmp_path):
    dr = DecisionRecord(tmp_path / "decisions.jsonl")
    dr.add("x")
    out = dr.write_markdown()
    assert out == tmp_path / "decisions.md"
    assert out.read_text(encoding="utf-8").startswith("# Decision record")


def test_write_markdown_explicit_path(tmp_path):
    dr = DecisionRecord(tmp_path / "d.jsonl")
    dr.add("x")
    out = dr.write_markdown(tmp_path / "custom.md")
    assert out == tmp_path / "custom.md"
    assert out.exists()
