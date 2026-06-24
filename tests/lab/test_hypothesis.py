"""Edge-case tests for ``lighttrain.lab.hypothesis``.

Pins the JSONL-backed hypothesis log:

* persistence round-trip; ``_load`` skips blank / malformed / wrong-schema;
* ``add`` (id increment, ``run_dir`` stringified); ``update`` (outcome sets
  ``resolved_ts``, run_dir update, non-matching id no-op);
* ``render_markdown`` empty + resolved/pending status + run/outcome/tags;
* ``write_markdown`` default/explicit paths; ``__len__`` / ``__iter__``.
"""

from __future__ import annotations

from pathlib import Path

from lighttrain.lab.hypothesis import HypothesisEntry, HypothesisLog


def test_add_increments_ids_stringifies_run_dir_and_persists(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    assert log.add("h0", run_dir=Path("/runs/a")) == 0
    assert log.add("h1") == 1
    assert log._entries[0].run_dir == str(Path("/runs/a"))
    assert log._entries[1].run_dir is None
    # reload
    assert len(HypothesisLog(tmp_path / "h.jsonl")) == 2


def test_load_skips_blank_malformed_and_wrong_schema(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text(
        "\n".join(["", '{"id": 0, "hypothesis": "ok"}', "nope", '{"bad": 1}']) + "\n",
        encoding="utf-8",
    )
    assert len(HypothesisLog(p)) == 1


def test_update_sets_outcome_and_resolved_ts(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    hid = log.add("h")
    log.update(hid, outcome="confirmed")
    e = HypothesisLog(tmp_path / "h.jsonl")._entries[0]
    assert e.outcome == "confirmed"
    assert e.resolved_ts is not None


def test_update_can_set_run_dir_only(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    hid = log.add("h")
    log.update(hid, run_dir="/runs/b")
    e = log._entries[0]
    assert e.run_dir == "/runs/b"
    assert e.outcome is None  # no outcome → resolved_ts stays None
    assert e.resolved_ts is None


def test_update_nonexistent_id_is_noop(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    log.add("h")
    log.update(42, outcome="x")  # no match → silent no-op
    assert log._entries[0].outcome is None


def test_render_markdown_empty():
    log = HypothesisLog.__new__(HypothesisLog)
    log._entries = []
    md = log.render_markdown()
    assert "_No hypotheses logged yet._" in md


def test_render_markdown_resolved_and_pending_with_fields():
    log = HypothesisLog.__new__(HypothesisLog)
    log._entries = [
        HypothesisEntry(id=0, hypothesis="resolved one", run_dir="/r/a",
                        outcome="it worked", tags=["lr", "rwkv"]),
        HypothesisEntry(id=1, hypothesis="pending one"),
    ]
    md = log.render_markdown()
    assert "✅ resolved" in md
    assert "⏳ pending" in md
    assert "**Run:** `/r/a`" in md
    assert "**Outcome:** it worked" in md
    assert "**Tags:** lr, rwkv" in md


def test_write_markdown_default_and_explicit_paths(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    log.add("h")
    default_out = log.write_markdown()
    assert default_out == tmp_path / "h.md"
    custom_out = log.write_markdown(tmp_path / "c.md")
    assert custom_out == tmp_path / "c.md" and custom_out.exists()


def test_len_and_iter(tmp_path):
    log = HypothesisLog(tmp_path / "h.jsonl")
    log.add("a")
    log.add("b")
    assert len(log) == 2
    assert [e.hypothesis for e in log] == ["a", "b"]
