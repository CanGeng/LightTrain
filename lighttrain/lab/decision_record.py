"""Decision records.

Opt-in structured log of architectural and algorithmic decisions.  Modelled
after Architecture Decision Records (ADR) but intentionally lightweight — a
single JSONL file per project, not one file per decision.

Usage::

    from lighttrain.lab.decision_record import DecisionRecord

    dr = DecisionRecord(Path("decisions.jsonl"))
    dr.add(
        title="Use AdamW over Lion for RWKV pre-training",
        context="Lion diverged at step 500 on the RWKV experiment; ...",
        decision="Switch back to AdamW with lr=3e-4",
        status="accepted",
    )
    dr.write_markdown(Path("decisions.md"))
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DecisionEntry:
    id: int
    title: str
    context: str = ""
    decision: str = ""
    consequences: str = ""
    status: str = "proposed"   # proposed | accepted | deprecated | superseded
    added_ts: float = field(default_factory=time.time)
    superseded_by: int | None = None
    tags: list[str] = field(default_factory=list)


class DecisionRecord:
    """Append-only log of project decisions backed by a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: list[DecisionEntry] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._entries.append(DecisionEntry(**d))
                except (json.JSONDecodeError, TypeError):
                    continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(asdict(e)) + "\n")

    def add(
        self,
        title: str,
        *,
        context: str = "",
        decision: str = "",
        consequences: str = "",
        status: str = "proposed",
        tags: list[str] | None = None,
    ) -> int:
        """Append a new decision record; returns its id."""
        entry = DecisionEntry(
            id=len(self._entries),
            title=title,
            context=context,
            decision=decision,
            consequences=consequences,
            status=status,
            tags=list(tags or []),
        )
        self._entries.append(entry)
        self._save()
        return entry.id

    def accept(self, decision_id: int) -> None:
        self._set_status(decision_id, "accepted")

    def deprecate(self, decision_id: int, *, superseded_by: int | None = None) -> None:
        for e in self._entries:
            if e.id == decision_id:
                e.status = "deprecated"
                e.superseded_by = superseded_by
                break
        self._save()

    def _set_status(self, decision_id: int, status: str) -> None:
        for e in self._entries:
            if e.id == decision_id:
                e.status = status
                break
        self._save()

    def render_markdown(self) -> str:
        lines = ["# Decision record", ""]
        if not self._entries:
            lines.append("_No decisions recorded yet._")
            return "\n".join(lines)
        _STATUS_ICON = {
            "proposed": "🔵",
            "accepted": "✅",
            "deprecated": "🚫",
            "superseded": "⬆️",
        }
        for e in self._entries:
            icon = _STATUS_ICON.get(e.status, "❓")
            lines.append(f"## DR-{e.id:03d}: {e.title} {icon}")
            lines.append(f"**Status:** {e.status}")
            if e.context:
                lines.append(f"\n**Context:** {e.context}")
            if e.decision:
                lines.append(f"\n**Decision:** {e.decision}")
            if e.consequences:
                lines.append(f"\n**Consequences:** {e.consequences}")
            if e.superseded_by is not None:
                lines.append(f"\n_Superseded by DR-{e.superseded_by:03d}._")
            lines.append("")
        return "\n".join(lines)

    def write_markdown(self, out_path: Path | None = None) -> Path:
        if out_path is None:
            out_path = self.path.with_suffix(".md")
        out_path = Path(out_path)
        out_path.write_text(self.render_markdown(), encoding="utf-8")
        return out_path

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["DecisionRecord", "DecisionEntry"]
