"""Migration invariant: every in-tree model-bearing recipe is in profile form.

After the v0.1.8 `model:` → `model_profiles:` migration, no shipped recipe may
carry a bare-dict `model:` block. This parametrizes over all tracked recipes and
asserts the invariant via `select_model_spec` — which is the precise migration
boundary (it rejects bare dicts and bad selectors) and, unlike full model
construction, needs none of a recipe's optional deps. So it stays CI-safe while
still catching a recipe that slipped through un-migrated.

Recipes with neither `model:` nor `model_profiles:` are distributed/sweep
*overlays* (composed onto a base, never built standalone) and are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lighttrain.config import load_config
from lighttrain.config._resolver import select_model_spec

_ROOT = Path(__file__).resolve().parents[2]
# Distributed/sweep overlays now live alongside the rest under examples/references/recipes/ (they
# were moved out of the old distributed-recipes location back when the bundled
# extensions package was renamed to lighttrain.builtin_plugins).
_RECIPES = sorted((_ROOT / "examples" / "references" / "recipes").glob("*.yaml"))


def _ids(p: Path) -> str:
    return str(p.relative_to(_ROOT))


@pytest.mark.parametrize("recipe", _RECIPES, ids=[_ids(p) for p in _RECIPES])
def test_recipe_has_no_bare_model_block(recipe: Path):
    raw = yaml.safe_load(recipe.read_text(encoding="utf-8")) or {}

    # Overlays / sweep specs declare neither — they are not standalone-buildable.
    if "model" not in raw and "model_profiles" not in raw:
        pytest.skip("overlay/sweep recipe: no model section")

    # A raw `model:` that is still a mapping means the migration was missed.
    assert not isinstance(raw.get("model"), dict), (
        f"{_ids(recipe)} still has a bare-dict `model:` block; "
        "run `lighttrain migrate config <recipe> --to-profiles`"
    )

    # Static migration check: parse + validate only. `import_user_modules=False`
    # keeps this CI-safe (a recipe's user_modules may be optional/absent in CI);
    # select_model_spec needs the parsed model/model_profiles, not the imports.
    cfg = load_config(recipe, import_user_modules=False)
    spec = select_model_spec(cfg.model, cfg.model_profiles)
    assert isinstance(spec, dict) and spec
    assert spec.get("name") or spec.get("_target_"), spec


def test_inventory_nonempty():
    # Guard against the glob silently matching nothing (e.g. a path typo).
    assert len(_RECIPES) >= 20
