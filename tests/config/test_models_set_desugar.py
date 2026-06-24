"""Keystone step 4: models:/optimizers: set + sugar desugar.

- The explicit ``models:``/``optimizers:`` form must produce per-step losses
  IDENTICAL to the lone ``model:``/``model_profiles:``/``optim:`` sugar form
  (desugar transparency — the single-model path is bit-identical).
- Declaring both ``model:`` and ``models:`` is a conflict error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
RECIPE = REPO / "recipes" / "pretrain_causal.yaml"


def _losses_for(recipe_path: Path, run_root: Path) -> list[float]:
    from lighttrain.cli._runtime import setup_run_from_config

    overrides = [
        f"++run_root={run_root.as_posix()}",
        "++trainer.max_steps=5",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
        "++trainer.log_every=1",
        "++logger=[{name: jsonl}]",
    ]
    bundle = setup_run_from_config(recipe_path, overrides=overrides)
    bundle["trainer"].fit()
    bundle["logger"].close()
    jsonl = bundle["run_dir"] / "logs" / "metrics.jsonl"
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "scalar" and "loss" in rec:
            out.append(round(float(rec["loss"]), 8))
    return out


def _write_explicit_variant(tmp_path: Path) -> Path:
    """Rewrite the sugar recipe into the explicit models:/optimizers: form."""
    cfg = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    profiles = cfg.pop("model_profiles")
    selected = cfg.pop("model")
    optim = cfg.pop("optim")
    cfg["models"] = {
        "main": {"spec": dict(profiles[selected]), "trainable": True, "optimizer": "main"}
    }
    cfg["optimizers"] = {"main": optim}
    out = tmp_path / "explicit.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


@pytest.mark.skipif(not RECIPE.exists(), reason="pretrain_causal.yaml missing")
def test_explicit_models_equals_sugar(tmp_path):
    sugar = _losses_for(RECIPE, tmp_path / "sugar")
    explicit = _losses_for(_write_explicit_variant(tmp_path), tmp_path / "explicit")
    assert len(sugar) == 5 and sugar == explicit, (sugar, explicit)


@pytest.mark.skipif(not RECIPE.exists(), reason="pretrain_causal.yaml missing")
def test_model_and_models_conflict_raises(tmp_path):
    from lighttrain.cli._runtime import setup_run_from_config
    from lighttrain.config import ConfigError

    cfg = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    # keep model: AND add models: → conflict
    cfg["models"] = {
        "main": {"spec": dict(cfg["model_profiles"]["default"]), "trainable": True}
    }
    recipe = tmp_path / "conflict.yaml"
    recipe.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="both `model:` and `models:`"):
        setup_run_from_config(
            recipe, overrides=[f"++run_root={(tmp_path / 'r').as_posix()}"]
        )
