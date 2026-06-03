"""Model-build paths are unified onto ``config._models`` (``normalize_model_set``).

Closes the v0.1.8 / Step-4 drift: ``estimate`` / ``produce`` / ``export`` /
``dry-run --build`` each carried their own model-declaration parser and broke on
``models:`` sets (or even on the v0.1.8 ``model_profiles:`` string selector).
These pin every build path on BOTH declaration forms, plus the
desugar-transparency bit-check that proves the single-model result is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from lighttrain.cli._runtime import _build_model
from lighttrain.config import ConfigError, load_config
from lighttrain.config._models import build_primary_model, normalize_model_set

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "recipes" / "pretrain_causal.yaml"          # model: + model_profiles:
MODELS_SET = REPO / "recipes" / "online_distill_demo.yaml"    # explicit models: set

pytestmark = pytest.mark.skipif(
    not (PROFILES.exists() and MODELS_SET.exists()),
    reason="recipe fixtures missing",
)


@pytest.mark.parametrize("recipe", [PROFILES, MODELS_SET], ids=["model_profiles", "models_set"])
def test_cli_build_model_builds_on_both_forms(recipe):
    """#1 — cli ``_build_model`` (dry-run / produce-artifact path) builds on a
    ``model_profiles:`` recipe AND a ``models:`` set (the latter used to raise
    'recipe is missing model:/model_profiles:')."""
    model = _build_model(load_config(recipe))
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("recipe", [PROFILES, MODELS_SET], ids=["model_profiles", "models_set"])
def test_estimate_builds_on_both_forms(recipe):
    """#2 — public ``estimate()`` builds on both forms (the ``models:`` set used
    to raise via estimate's own removed ``_build_model``)."""
    from lighttrain.lab.estimate import estimate

    rpt = estimate(load_config(recipe))
    assert rpt.all_params > 0
    assert rpt.model_name


def test_build_is_bit_identical_to_direct_path():
    """#3 — desugar transparency (stronger than a param-count check): the lone
    ``model:``+``model_profiles:`` path through ``normalize_model_set`` builds a
    model bit-for-bit identical to the pre-refactor direct
    ``select_model_spec`` + ``resolve`` under a fixed seed."""
    from lighttrain.config._resolver import resolve, select_model_spec
    from lighttrain.utils.seed import seed_everything

    cfg = load_config(PROFILES)
    seed_everything(int(cfg.seed))
    m_unified, n_trainable = build_primary_model(cfg)
    seed_everything(int(cfg.seed))
    m_direct = resolve(select_model_spec(cfg.model, cfg.model_profiles), category="model")

    assert n_trainable == 1
    assert type(m_unified) is type(m_direct)
    sd_u, sd_d = m_unified.state_dict(), m_direct.state_dict()
    assert sd_u.keys() == sd_d.keys()
    for k in sd_u:
        assert torch.equal(sd_u[k], sd_d[k]), f"weight mismatch at {k}"


def test_dangling_optimizer_reference_raises(tmp_path):
    """#4 (F) — a trainable entry naming an optimizer absent from ``optimizers:``
    is reported by name, not a generic 'missing optimizer' at build time."""
    cfg = yaml.safe_load(MODELS_SET.read_text(encoding="utf-8"))
    cfg["models"]["student"]["optimizer"] = "ghost"
    recipe = tmp_path / "dangling.yaml"
    recipe.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"ghost.*not found"):
        normalize_model_set(load_config(recipe))


def test_primary_is_first_trainable_in_axis_b(tmp_path):
    """#6 — Axis-B: two TRAINABLE models of different sizes → ``build_primary_model``
    returns the FIRST trainable's model and reports the count (documents the
    'primary = first trainable' semantics export relies on)."""
    recipe = tmp_path / "axis_b.yaml"
    recipe.write_text(
        # sort_keys=False: keep ``gen`` before ``disc`` so "first trainable" is
        # unambiguous (primary == recipe declaration order).
        yaml.safe_dump(
            {
                "mode": "lab",
                "seed": 7,
                "exp": "axis_b",
                "run_root": str(tmp_path),
                "models": {
                    "gen": {
                        "spec": {"name": "tiny_lm", "vocab_size": 64, "d_model": 32,
                                 "n_layers": 2, "n_heads": 4, "max_seq_len": 32},
                        "trainable": True,
                        "optimizer": "main",
                    },
                    "disc": {
                        "spec": {"name": "tiny_lm", "vocab_size": 64, "d_model": 128,
                                 "n_layers": 2, "n_heads": 4, "max_seq_len": 32},
                        "trainable": True,
                        "optimizer": "main",
                    },
                },
                "optimizers": {"main": {"name": "adamw", "lr": 1.0e-3}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    model, n_trainable = build_primary_model(load_config(recipe))
    assert n_trainable == 2
    # first trainable is ``gen`` (d_model=32), not ``disc`` (d_model=128).
    assert int(model.d_model) == 32
