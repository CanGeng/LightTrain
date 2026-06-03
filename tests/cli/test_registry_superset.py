"""Mechanism-swap safety net (#5): auto-discovery must not LOSE any registration
the old hand-maintained ``_eager_import_components`` list produced.

``tests/cli/fixtures/registry_baseline.json`` was dumped from the pre-refactor
code (the hand list). The replacement (``config._components.import_all_components``,
curated packages walked recursively) must be a per-category SUPERSET of it:
adding components is fine, *losing* one is the dangerous silent-omission failure.

**Why a subprocess**: run in-process and ``import lighttrain`` / sibling test
modules pull register-bearing packages in transitively, masking whether
``import_all_components`` *alone* covers them. A clean subprocess (only
``import_all_components`` + dump) is the only way to prove the curated list is
self-sufficient and doesn't lean on incidental transitive imports — exactly the
fragility this refactor removes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).parent / "fixtures" / "registry_baseline.json"

_DUMP = textwrap.dedent(
    """
    import json
    from lighttrain.config._components import import_all_components
    from lighttrain import categories, list_entries
    import_all_components()
    print("REGISTRY_DUMP::" + json.dumps(
        {c: sorted(list_entries(c)) for c in categories()}))
    """
)


def _registry_in_clean_process() -> dict[str, list[str]]:
    res = subprocess.run(
        [sys.executable, "-c", _DUMP],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert res.returncode == 0, f"clean-process import_all_components failed:\n{res.stderr}"
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("REGISTRY_DUMP::")), None)
    assert line is not None, f"no registry dump in output:\n{res.stdout}\n{res.stderr}"
    return json.loads(line[len("REGISTRY_DUMP::"):])


def test_auto_discovery_loses_no_baseline_registration():
    live = _registry_in_clean_process()
    base: dict[str, list[str]] = json.loads(BASELINE.read_text(encoding="utf-8"))
    missing = {
        cat: sorted(set(names) - set(live.get(cat, [])))
        for cat, names in base.items()
        if set(names) - set(live.get(cat, []))
    }
    assert not missing, f"auto-discovery LOST registrations vs baseline: {missing}"


def test_aux_losses_now_registered():
    """#5b — the drift this fixes: info_nce / moe_balance (lighttrain/builtin_plugins/losses/aux.py)
    were never eagerly imported by the hand list; auto-discovery picks them up."""
    live = _registry_in_clean_process()
    assert {"info_nce", "moe_balance"} <= set(live.get("loss", []))
