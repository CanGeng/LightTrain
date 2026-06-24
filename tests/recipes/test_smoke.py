"""Recipes smoke tests — config load + PrepGraph dry-run for core recipes (R1/R2/R14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lighttrain.cli._runtime import build_prep_runner
from lighttrain.config import load_config

REPO = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# R1 — causal LM pretrain (DESIGN §25.1 R1)
# ---------------------------------------------------------------------------

def test_pretrain_causal_config_loads():
    """R1 smoke: config parses cleanly and required keys are present."""
    cfg = load_config(REPO / "recipes" / "pretrain_causal.yaml")
    assert cfg is not None
    # Basic structure expected by the trainer
    assert hasattr(cfg, "model") or (isinstance(cfg, dict) and "model" in cfg)
    assert hasattr(cfg, "trainer") or (isinstance(cfg, dict) and "trainer" in cfg)


def test_sft_chat_dry_run():
    bundle = build_prep_runner(REPO / "recipes" / "sft_chat.yaml")
    plan = bundle["runner"].dry_run()
    names = [p.name for p in plan]
    assert "raw" in names
    assert "tokenized" in names
    assert "packed" in names
    assert "train_data" in names
    assert all(not p.hit for p in plan)


def test_vlm_sft_dry_run():
    bundle = build_prep_runner(REPO / "recipes" / "vlm_sft.yaml")
    plan = bundle["runner"].dry_run()
    names = [p.name for p in plan]
    assert "train_data" in names


def test_sft_chat_run_then_replay(tmp_path: Path):
    bundle = build_prep_runner(
        REPO / "recipes" / "sft_chat.yaml",
        store_root=tmp_path / "prep",
    )
    bundle["runner"].run()
    bundle2 = build_prep_runner(
        REPO / "recipes" / "sft_chat.yaml",
        store_root=tmp_path / "prep",
    )
    plan2 = bundle2["runner"].plan()
    assert all(entry.hit for entry in plan2)


@pytest.mark.heavy
def test_sft_chat_hf_dry_run():
    """HF tokenizer recipe — heavy because of the model download."""
    bundle = build_prep_runner(REPO / "recipes" / "sft_chat_hf.yaml")
    plan = bundle["runner"].dry_run()
    assert any(p.kind == "tokenize" for p in plan)
