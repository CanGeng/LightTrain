"""Anti-drift lock (#9): model DECLARATION parsing must live in ONE place.

The v0.1.8 / Step-4 bug class was each consumer growing its own parser for the
``model:`` declaration. After unification, the two declaration-parsing forms —
the ``select_model_spec`` primitive and bare ``cfg.model`` / ``cfg.models`` reads
— must appear only in ``config/_models.py`` and ``config/_resolver.py``. A new
command that grows its own parser fails *here*, at lint time, instead of shipping
latent and only blowing up on a specific recipe months later.

Deliberately NOT flagged (legitimate, verified against the tree):
* ``_resolve(category="model")`` — consuming an already-normalised spec.
* ``self.model`` / ``ctx.model`` / ``trainer.model`` — runtime model objects.
* ``lighttrain.models`` — module paths.
* ``spec.get("model")`` — e.g. dynamic_producer checking a ``$self`` ref.
* ``cfg.model_profiles`` / ``models_cfg`` — not the bare declaration field.
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "lighttrain"
ALLOWLIST = {Path("config/_models.py"), Path("config/_resolver.py")}

# The unique declaration→spec primitive.
_SELECT = re.compile(r"\bselect_model_spec\b")
# ``cfg``-anchored reads of the bare ``model`` / ``models`` declaration field.
# ``(?!_)`` + the ``\b`` keep ``cfg.model_profiles`` / ``cfg.model_dump`` out.
_CFG_ATTR = re.compile(r"\bcfg\.models?\b(?!_)")
_CFG_GET = re.compile(r"""\bcfg\.get\(\s*['"]models?['"]""")
_CFG_GETATTR = re.compile(r"""\bgetattr\(\s*cfg\s*,\s*['"]models?['"]""")
_PATTERNS = (_SELECT, _CFG_ATTR, _CFG_GET, _CFG_GETATTR)


def _offenders() -> list[str]:
    out: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        if path.relative_to(PKG) in ALLOWLIST:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments
            if any(p.search(code) for p in _PATTERNS):
                out.append(f"{path.relative_to(PKG)}:{lineno}: {line.strip()}")
    return out


def test_no_rogue_model_declaration_parser():
    offenders = _offenders()
    assert not offenders, (
        "model declaration parsing found outside config/_models.py + "
        "config/_resolver.py — route it through normalize_model_set instead:\n"
        + "\n".join(offenders)
    )
