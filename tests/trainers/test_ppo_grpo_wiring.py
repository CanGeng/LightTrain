"""PPO/GRPO runtime wiring (relocated from tests/test_ppo_grpo_wiring.py).

Unit layer:
  - GRPOTrainer / PPOTrainer accept val_every without TypeError.
  - _reward_fn adapter correctly decodes tensors and calls VerifierJudge.

Runtime integration layer (setup_run_from_config):
  - GRPO recipe: trainer.reward_fn is not None; grad_clip/accumulate filtered.
  - PPO recipe: same.
  - preference recipe: no grad_clip/accumulate leak to the trainer __init__.
  - Migration: removed per-algo preference trainers raise a clear error.
  - Negative: PPO/GRPO + pairwise_llm judge raises ConfigResolveError.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_RECIPES = Path(__file__).resolve().parents[2] / "examples" / "references" / "recipes"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _stub_engine():
    e = MagicMock()
    e.update_rule = MagicMock()
    return e


# ===========================================================================
# Unit layer — constructor compat
# ===========================================================================


def test_grpo_trainer_accepts_val_every():
    from lighttrain.builtin_plugins.trainers.grpo import GRPOTrainer

    trainer = GRPOTrainer(
        val_every=0, engine=_stub_engine(), data_module=MagicMock(),
        optimizer=MagicMock(), max_steps=1,
    )
    assert trainer is not None


def test_ppo_trainer_accepts_val_every():
    from lighttrain.builtin_plugins.trainers.ppo import PPOTrainer

    trainer = PPOTrainer(
        val_every=0, engine=_stub_engine(), data_module=MagicMock(),
        optimizer=MagicMock(), max_steps=1,
    )
    assert trainer is not None


def test_reward_fn_adapter_decodes_and_scores():
    """_reward_fn adapter: tensor decode → VerifierJudge.score() → list[float]."""
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
    from lighttrain.builtin_plugins.judges.judge import VerifierJudge

    tok = ByteTokenizer()
    judge = VerifierJudge(verify_pattern=r"\d+")

    def _reward_fn(prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> list[float]:
        prompts = [tok.decode(ids.tolist(), skip_special_tokens=True) for ids in prompt_ids]
        responses = [tok.decode(ids.tolist(), skip_special_tokens=True) for ids in response_ids]
        return judge.score(list(zip(prompts, responses, strict=False)))

    resp_with_digit = torch.tensor(list(b"hello 42\x00"), dtype=torch.long)
    resp_no_digit = torch.tensor(list(b"worldword"), dtype=torch.long)
    prompts = torch.zeros((2, 3), dtype=torch.long)
    responses = torch.stack([resp_with_digit, resp_no_digit])

    scores = _reward_fn(prompts, responses)
    assert len(scores) == 2
    assert scores[0] == 1.0, "response with digit must score 1.0"
    assert scores[1] == 0.0, "response without digit must score 0.0"


# ===========================================================================
# Runtime integration — setup_run_from_config
# ===========================================================================


@pytest.mark.skipif(not (_RECIPES / "grpo.yaml").exists(), reason="grpo.yaml missing")
def test_grpo_runtime_reward_fn_injected():
    from lighttrain.cli._runtime import setup_run_from_config

    bundle = setup_run_from_config(
        _RECIPES / "grpo.yaml",
        overrides=["++trainer.max_steps=1", "++trainer.ckpt_every=0"],
        mode="lab",
    )
    assert bundle["trainer"].reward_fn is not None


@pytest.mark.skipif(not (_RECIPES / "ppo_online.yaml").exists(), reason="ppo_online.yaml missing")
def test_ppo_runtime_reward_fn_injected():
    from lighttrain.cli._runtime import setup_run_from_config

    bundle = setup_run_from_config(
        _RECIPES / "ppo_online.yaml",
        overrides=["++trainer.max_steps=1", "++trainer.ckpt_every=0"],
        mode="lab",
    )
    assert bundle["trainer"].reward_fn is not None


def _write_minimal_preference_recipe(tmp_path, fixture, *, trainer_name: str) -> Path:
    """Self-contained minimal preference recipe (no artifact store) for hermetic
    runtime tests. ``trainer_name`` lets callers exercise the migration error."""
    import textwrap

    recipe = tmp_path / "pref_minimal.yaml"
    recipe.write_text(textwrap.dedent(f"""
        mode: lab
        seed: 42
        exp: test_pref_minimal
        run_root: {tmp_path / "runs"}

        model: default
        model_profiles:
          default:
            name: tiny_lm
            vocab_size: 260
            d_model: 32
            n_layers: 1
            n_heads: 2
            max_seq_len: 32

        data:
          name: simple
          dataset:
            name: preference_jsonl
            path: {fixture}
            max_len: 32
          tokenizer:
            name: byte
          collator:
            name: preference
            max_len: 32
            pad_id: 256
          sampler:
            name: sequential
          batch_size: 2
          num_workers: 0

        loss:
          name: dpo
          beta: 0.1

        optim:
          name: adamw
          lr: 1.0e-4
          betas: [0.9, 0.95]
          weight_decay: 0.01

        scheduler:
          name: warmup_cosine
          warmup_steps: 1
          total_steps: 1
          min_lr_ratio: 0.1

        engine:
          name: standard
          mixed_precision: "no"

        trainer:
          name: {trainer_name}
          max_steps: 1
          val_every: 0
          ckpt_every: 0
          log_every: 1
          grad_clip: 1.0
          accumulate: 1

        callbacks: []
        logger:
          - {{name: console, log_every: 1}}
    """).strip())
    return recipe


def test_preference_runtime_no_grad_clip_leak(tmp_path):
    """preference recipe: grad_clip/accumulate must not leak to the trainer
    __init__ (the runtime keeps them engine-only). Hermetic; no artifact store."""
    from lighttrain.cli._runtime import setup_run_from_config

    fixture = _FIXTURES / "tiny_preference.jsonl"
    if not fixture.exists():
        pytest.skip("tiny_preference.jsonl fixture not found")

    recipe = _write_minimal_preference_recipe(tmp_path, fixture, trainer_name="preference")
    bundle = setup_run_from_config(recipe, mode="lab")
    assert bundle["trainer"] is not None


def test_removed_preference_trainer_raises_migration_error(tmp_path):
    """Keystone step 2: `trainer: dpo` (and ipo/simpo/orpo/kto) must fail with a
    clear migration error pointing at `trainer: preference` + the `loss:` seam."""
    from lighttrain.cli._runtime import setup_run_from_config
    from lighttrain.config import ConfigError

    fixture = _FIXTURES / "tiny_preference.jsonl"
    if not fixture.exists():
        pytest.skip("tiny_preference.jsonl fixture not found")

    recipe = _write_minimal_preference_recipe(tmp_path, fixture, trainer_name="dpo")
    with pytest.raises(ConfigError, match="preference"):
        setup_run_from_config(recipe, mode="lab")


def test_pairwise_llm_judge_with_grpo_raises():
    """Non-VerifierJudge with PPO/GRPO must raise a clear error, not silently
    produce wrong rewards. Two failure modes are acceptable:
    (a) ConfigResolveError at judge-construction time,
    (b) ConfigResolveError / RuntimeError at reward_fn-injection time.
    Either way, the framework must refuse rather than silently break.
    """
    import tempfile
    import textwrap

    from lighttrain.cli._runtime import setup_run_from_config
    from lighttrain.config._exceptions import ConfigResolveError

    fixture = _FIXTURES / "tiny_corpus.txt"
    if not fixture.exists():
        pytest.skip("tiny_corpus.txt fixture not found")

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(textwrap.dedent(f"""
            mode: lab
            seed: 99
            exp: test_pairwise_grpo
            run_root: /tmp/test_pairwise_grpo_runs

            model: default
            model_profiles:
              default:
                name: tiny_lm
                vocab_size: 260
                d_model: 32
                n_layers: 1
                n_heads: 2
                max_seq_len: 32

            data:
              name: simple
              dataset:
                name: line_file_text
                path: {fixture}
                max_len: 32
              tokenizer:
                name: byte
              collator:
                name: causal_lm
                max_len: 32
              sampler:
                name: sequential
              batch_size: 2
              num_workers: 0

            judge:
              name: pairwise_llm
              judge_model_fn:
                _target_: builtins.str  # placeholder callable

            optim:
              name: adamw
              lr: 1.0e-4
              betas: [0.9, 0.95]
              weight_decay: 0.0

            scheduler:
              name: warmup_cosine
              warmup_steps: 1
              total_steps: 1
              min_lr_ratio: 0.0

            engine:
              name: standard
              mixed_precision: "no"

            trainer:
              name: grpo
              max_steps: 1
              ckpt_every: 0
              log_every: 1
              group_size: 2
              ppo_epochs: 1
              mini_batch_size: 2
              max_new_tokens: 8

            callbacks: []
            logger:
              - {{name: console, log_every: 1}}
        """).strip())
        recipe_path = f.name

    with pytest.raises((ConfigResolveError, RuntimeError)):
        setup_run_from_config(Path(recipe_path), mode="lab")
