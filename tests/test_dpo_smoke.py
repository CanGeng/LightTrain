"""DPO integration smoke — verifies aux.* → PreferenceCollator → DPOLoss chain.

Covers REVIEW_ROUND3 finding #3:
    collators.py:86 → _preference_base.py:266-275 → losses/preference.py:98.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch
import pytest

from lighttrain.data.core.collators import PreferenceCollator
from lighttrain.trainers.dpo import DPOTrainer


def _make_engine():
    """Minimal engine mock that satisfies DPOTrainer._step."""
    engine = MagicMock()
    engine.update_rule = MagicMock()
    return engine


def _make_sample(seq_len: int = 6, ref_chosen: float = -1.5, ref_rejected: float = -2.0):
    return {
        "chosen_input_ids": list(range(seq_len)),
        "chosen_labels": list(range(seq_len)),
        "rejected_input_ids": list(range(seq_len)),
        "rejected_labels": list(range(seq_len)),
        "aux.ref.chosen_logprobs": torch.tensor(ref_chosen),
        "aux.ref.rejected_logprobs": torch.tensor(ref_rejected),
    }


def test_collator_preserves_aux_keys():
    """PreferenceCollator must forward aux.ref.* keys to the batch."""
    collator = PreferenceCollator(pad_id=0, max_len=16)
    samples = [_make_sample(), _make_sample(ref_chosen=-0.8, ref_rejected=-1.2)]
    batch = collator(samples)

    assert "aux.ref.chosen_logprobs" in batch, "aux chosen key must survive collation"
    assert "aux.ref.rejected_logprobs" in batch, "aux rejected key must survive collation"
    assert batch["aux.ref.chosen_logprobs"].shape == (2,)
    assert batch["aux.ref.rejected_logprobs"].shape == (2,)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dpo_step_with_aux_logprobs():
    """Full chain: collator → _preference_base._step() → DPOLoss must not raise KeyError."""
    from lighttrain.models.adapters.tiny_lm import TinyCausalLM

    vocab_size = 64
    model = TinyCausalLM(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2,
                         max_seq_len=32, tie_weights=False)
    model.eval()

    collator = PreferenceCollator(pad_id=0, max_len=16)
    samples = [_make_sample(seq_len=4), _make_sample(seq_len=4, ref_chosen=-0.9, ref_rejected=-1.7)]
    batch = collator(samples)

    # Confirm aux keys are present before calling _step.
    assert "aux.ref.chosen_logprobs" in batch

    # Construct a minimal DPOTrainer (engine and data_module are mocked).
    trainer = DPOTrainer(
        beta=0.1,
        engine=_make_engine(),
        data_module=MagicMock(),
        optimizer=MagicMock(),
        model=model,
        device="cpu",
    )

    # _preference_step() reads aux.* and calls DPOLoss.
    loss_dict = trainer._preference_step(batch)

    assert "loss" in loss_dict, "_step must return a dict with 'loss'"
    loss_val = loss_dict["loss"]
    assert math.isfinite(float(loss_val)), f"DPO loss must be finite, got {loss_val}"
