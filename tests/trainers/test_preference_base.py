"""Adversarial tests for PreferenceTrainer (shared base for DPO/IPO/KTO/ORPO/SimPO/RM).

We exercise the abstract-ish base by instantiating DPOTrainer (concrete subclass)
to avoid duplicating preference-loss math here. The tests pin control-flow
contracts at the base class level:

  - fit() lifecycle: on_train_start → on_epoch_begin → on_train_batch_start
    → on_train_batch_end → on_train_end (strict order, partial)
  - STOP_TRAINING honored from both ctx.extras and on_train_batch_end
  - _step() clears stale loss_signal before _preference_step
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.callbacks.base import EventBus, Signal
from lighttrain.builtin_plugins.losses.preference import DPOLoss
from lighttrain.protocols import ModelOutput
from lighttrain.builtin_plugins.trainers._preference_base import PreferenceTrainer


class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


class _FakeEngine:
    pass


def _pref_batch(V: int = 16, T: int = 5, B: int = 2) -> dict:
    """Preference batch with chosen/rejected halves + ref log-probs."""
    return {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "chosen_labels": torch.randint(0, V, (B, T)),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_labels": torch.randint(0, V, (B, T)),
        "aux.ref.chosen_logprobs": torch.randn(B) - 1.0,
        "aux.ref.rejected_logprobs": torch.randn(B) - 2.0,
    }


class _PrefDM:
    """Pref DataModule returning a re-iterable list of batches."""

    def __init__(self, n: int = 3) -> None:
        self._batches = [_pref_batch() for _ in range(n)]

    def train_loader(self):
        return list(self._batches)


def _make_dpo(*, callbacks=None, max_steps: int = 1, model=None) -> PreferenceTrainer:
    if model is None:
        model = _TinyLM()
    trainer = PreferenceTrainer(
        engine=_FakeEngine(),
        data_module=_PrefDM(n=max_steps + 1),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model=model,
        callbacks=callbacks,
        max_steps=max_steps,
    )
    trainer.ctx.loss_fn = DPOLoss(beta=0.1)  # the loss: seam
    return trainer


# ===========================================================================
# Lifecycle order
# ===========================================================================


def test_preference_fit_lifecycle_strict_order():
    """Goal: pin temporal order of fit-level preference-trainer events.

    Construction: 1 step, recorder for all lifecycle events.

    Expected partial order (the recorder only catches events; trailing
    events from the inner step are emitted by RLUpdateRule which we test
    elsewhere):
      first 5 are [on_train_start, on_epoch_begin, on_train_batch_start,
       on_train_batch_end, on_train_end]
    """
    events: list[str] = []

    class _Rec:
        def on_train_start(self, **_): events.append("on_train_start")
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")
        def on_train_batch_start(self, **_): events.append("on_train_batch_start")
        def on_train_batch_end(self, **_): events.append("on_train_batch_end")
        def on_train_end(self, **_): events.append("on_train_end")

    trainer = _make_dpo(callbacks=[_Rec()], max_steps=1)
    trainer.fit()

    assert events[0] == "on_train_start"
    assert events[1] == "on_epoch_begin"
    assert "on_train_batch_start" in events
    assert "on_train_batch_end" in events
    assert events.index("on_train_batch_start") < events.index("on_train_batch_end")
    assert events[-1] == "on_train_end"


def test_preference_fit_honors_stop_signal_from_on_train_batch_end():
    """Goal: a callback returning STOP_TRAINING from on_train_batch_end
    breaks the fit loop after the current step (lines 196-197 in
    _preference_base.py).
    """

    class _Stopper:
        def on_train_batch_end(self, **_):
            return Signal.STOP_TRAINING

    trainer = _make_dpo(callbacks=[_Stopper()], max_steps=5)
    trainer.fit()

    assert trainer.ctx.step == 1


def test_preference_fit_honors_stop_signal_from_ctx_extras():
    """Goal: a callback that sets ``ctx.extras['loss_signal'] = STOP_TRAINING``
    inside on_loss_computed (during the RL step) must surface to the fit
    loop (lines 185-187 in _preference_base.py) and break after the
    current step.

    Construction: an on_loss_computed callback that returns STOP_TRAINING.
    The RLUpdateRule sets ctx.extras['loss_signal'] from the dispatched
    signal; the preference fit loop reads it.
    """

    class _Stopper:
        def on_loss_computed(self, **_):
            return Signal.STOP_TRAINING

    trainer = _make_dpo(callbacks=[_Stopper()], max_steps=5)
    trainer.fit()

    assert trainer.ctx.step == 1


# ===========================================================================
# loss_signal clearing
# ===========================================================================


def test_preference_step_clears_loss_signal_extras_per_call():
    """Goal: line 285 in _preference_base.py — ``_step`` pops loss_signal
    from ctx.extras before calling _preference_step.

    Catches a refactor that drops the pop and lets stale signals persist
    across iterations.
    """
    trainer = _make_dpo(max_steps=1)
    trainer.ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)

    trainer._step(_pref_batch())

    # Without callbacks returning a signal, loss_signal must not be re-added.
    assert "loss_signal" not in trainer.ctx.extras
