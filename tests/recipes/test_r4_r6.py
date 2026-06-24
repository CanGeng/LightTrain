"""Recipe smoke tests R4–R6 (M6) — CPU import + config parse only.

These tests verify that:
  1. The recipe YAML files parse without errors.
  2. The referenced trainer classes are importable and registered.
  3. The referenced loss classes are importable and registered.

Full training runs (heavy) are deliberately omitted because R4/R5/R6 require
external artifacts (reference logprobs) or multi-step rollout that would be
slow on CI CPU. Heavy equivalents belong in test_recipes_r4_r6_heavy.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_RECIPES = Path(__file__).parent.parent.parent / "recipes"


def _load(name: str) -> dict:
    with open(_RECIPES / name) as f:
        return yaml.safe_load(f)


# ---- R4 DPO offline -------------------------------------------------------

def test_r4_recipe_parses():
    cfg = _load("dpo_offline.yaml")
    # keystone step 2: single preference trainer; algorithm via loss: seam
    assert cfg["trainer"]["name"] == "preference"
    assert cfg["loss"]["name"] == "dpo"


def test_r4_preference_trainer_registered():
    from lighttrain.builtin_plugins.trainers._preference_base import PreferenceTrainer
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "preference") is PreferenceTrainer


# ---- R5 PPO online --------------------------------------------------------

def test_r5_recipe_parses():
    cfg = _load("ppo_online.yaml")
    assert cfg["trainer"]["name"] == "ppo"
    assert cfg["judge"]["name"] == "verifier"


def test_r5_ppo_trainer_registered():
    from lighttrain.builtin_plugins.trainers.ppo import PPOTrainer
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "ppo") is PPOTrainer


# ---- R6 GRPO --------------------------------------------------------------

def test_r6_recipe_parses():
    cfg = _load("grpo.yaml")
    assert cfg["trainer"]["name"] == "grpo"
    assert cfg["trainer"]["group_size"] == 4


def test_r6_grpo_trainer_registered():
    from lighttrain.builtin_plugins.trainers.grpo import GRPOTrainer
    from lighttrain.registry import get as resolve
    assert resolve("trainer", "grpo") is GRPOTrainer
