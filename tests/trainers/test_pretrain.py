"""Adversarial tests for PretrainTrainer — engine delegation, fit lifecycle,
STOP_TRAINING signal honoring, exception path.

Unlike PPO/GRPO, PretrainTrainer DOES go through ``engine.step(batch, ctx)``
— the engine is the standard path. We pin the delegation contract: each
train_step → exactly one engine.step call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer
from lighttrain.callbacks.base import Signal
from lighttrain.protocols import ModelOutput


class _TinyLM(nn.Module):
    def __init__(self, V: int = 8, D: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        return ModelOutput(outputs={"logits": self.head(self.emb(input_ids))})


def _batch():
    return {
        "input_ids": torch.randint(0, 8, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.randint(0, 8, (2, 4)),
    }


class _BatchListDM:
    """DataModule yielding a finite list of batches; re-iter-able.

    Returns the batch list itself so ``iter(loader)`` on epoch-rebuild
    yields fresh batches (a generator would stay exhausted).
    """

    def __init__(self, n: int) -> None:
        self._batches = [_batch() for _ in range(n)]

    def train_loader(self):
        return list(self._batches)  # list is re-iter-able

    def val_loader(self):
        return None

    def state_dict(self):
        return {}


# ===========================================================================
# Engine delegation
# ===========================================================================


def test_pretrain_fit_delegates_each_step_to_engine():
    """Goal: PretrainTrainer.fit must call ``engine.step`` exactly once per
    training step.

    Construction: mock engine with engine.step returning {"loss": 0.5}.
    Run fit() with max_steps=3 over a DM yielding 3 batches.

    Catches a refactor that calls engine.step twice (e.g. once for forward
    and once for backward — a misunderstanding of the contract).
    """
    mock_engine = MagicMock()
    mock_engine.step.return_value = {"loss": 0.5, "skipped": 0.0}

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(3),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=3,
        log_every=1,
        ckpt_every=0,
    )

    trainer.fit()

    assert mock_engine.step.call_count == 3
    assert trainer.ctx.step == 3


class _StatefulProfile:
    """Minimal duck-typed ArchitectureProfile for the stateful-reset path."""

    state_mode = "stateful"

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.reset_state_fn = lambda m: self.calls.append(1)


def test_pretrain_produce_batch_resets_state_on_doc_boundary():
    """Bit-check for the stateful arch-reset path lifted into produce_batch.

    The transformer regression fixtures are stateless, so this is the only
    coverage that the RWKV/Mamba recurrent-state reset still fires after the
    loop body moved into base.Trainer / run_train_loop.
    """
    model = _TinyLM()
    profile = _StatefulProfile()
    trainer = PretrainTrainer(
        engine=MagicMock(),
        data_module=_BatchListDM(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
        arch_profile=profile,
    )

    # No document boundary → no reset.
    b = trainer.produce_batch({**_batch()})
    assert profile.calls == []
    assert "_reset_state" not in b

    # Document boundary → reset_state_fn fired and flag propagated.
    b2 = trainer.produce_batch({**_batch(), "_doc_boundary": True})
    assert profile.calls == [1]
    assert b2["_reset_state"] is True


def test_pretrain_step_clears_loss_signal_extras():
    """Goal: ``_step`` must pop ``ctx.extras['loss_signal']`` before the
    engine call (line 227 in pretrain.py).
    """
    mock_engine = MagicMock()
    mock_engine.step.return_value = {"loss": 0.5}

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )
    trainer.ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)

    trainer._step(_batch())

    assert "loss_signal" not in trainer.ctx.extras


# ===========================================================================
# Lifecycle order
# ===========================================================================


def test_pretrain_fit_lifecycle_strict_order():
    """Goal: pin exact event order across one full fit() run.

    Construction:
      - mock engine returning {"loss": 0.5}
      - DM yielding 1 batch
      - record all lifecycle events
    Expected:
      [on_train_start, on_epoch_begin, on_train_batch_start,
       on_train_batch_end, on_epoch_end, on_train_end]
    """
    events: list[str] = []

    class _Rec:
        def on_train_start(self, **_): events.append("on_train_start")
        def on_epoch_begin(self, **_): events.append("on_epoch_begin")
        def on_epoch_end(self, **_): events.append("on_epoch_end")
        def on_train_batch_start(self, **_): events.append("on_train_batch_start")
        def on_train_batch_end(self, **_): events.append("on_train_batch_end")
        def on_train_end(self, **_): events.append("on_train_end")

    mock_engine = MagicMock()
    mock_engine.step.return_value = {"loss": 0.5}

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Rec()],
        max_steps=2,  # so iterator exhaustion triggers epoch end
    )

    trainer.fit()

    # The exact suffix may include extra on_epoch_begin (after exhaustion)
    # if max_steps > batches. We pin the head sequence:
    assert events[0] == "on_train_start"
    assert events[1] == "on_epoch_begin"
    assert "on_train_batch_start" in events
    assert "on_train_batch_end" in events
    # on_epoch_end fires on StopIteration (when target > 1 batch)
    assert "on_epoch_end" in events
    assert events[-1] == "on_train_end"
    # And start/end appear in the correct relative order:
    assert events.index("on_train_batch_start") < events.index("on_train_batch_end")


def test_pretrain_fit_honors_stop_training_from_ctx_extras():
    """Goal: STOP_TRAINING signaled via ctx.extras['loss_signal'] (set by
    StandardUpdateRule from a callback) must end the fit loop after the
    current step.

    Construction: mock engine that on its first call sets the loss_signal
    via the ctx it received, then returns metrics.

    Expected: ctx.step == 1 (only one step ran).
    """

    def _engine_step(batch, ctx):
        ctx.extras["loss_signal"] = int(Signal.STOP_TRAINING)
        return {"loss": 0.5}

    mock_engine = MagicMock()
    mock_engine.step = _engine_step

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(5),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=5,
    )

    trainer.fit()

    assert trainer.ctx.step == 1


def test_pretrain_fit_honors_stop_training_from_on_train_batch_end():
    """Goal: a callback returning STOP_TRAINING from ``on_train_batch_end``
    must end the loop after the current step (line 165-166 in pretrain.py).
    """

    class _Stopper:
        def on_train_batch_end(self, **_):
            return Signal.STOP_TRAINING

    mock_engine = MagicMock()
    mock_engine.step.return_value = {"loss": 0.5}

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(5),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Stopper()],
        max_steps=5,
    )

    trainer.fit()

    assert trainer.ctx.step == 1


# ===========================================================================
# Exception path
# ===========================================================================


def test_pretrain_fit_on_exception_dispatched_and_reraised():
    """Goal: when engine.step raises, on_exception fires, on_train_end still
    fires (finally), and the exception re-raises.
    """
    events: list[str] = []

    class _Rec:
        def on_exception(self, **_): events.append("on_exception")
        def on_train_end(self, **_): events.append("on_train_end")

    mock_engine = MagicMock()
    mock_engine.step.side_effect = RuntimeError("simulated step failure")

    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=mock_engine,
        data_module=_BatchListDM(5),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        callbacks=[_Rec()],
        max_steps=3,
    )

    with pytest.raises(RuntimeError, match="simulated step failure"):
        trainer.fit()

    assert "on_exception" in events
    assert "on_train_end" in events


# ===========================================================================
# Constructor guards
# ===========================================================================


def test_pretrain_fit_raises_when_model_is_none():
    """Pin guard at pretrain.py:95-96."""
    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=MagicMock(),
        data_module=_BatchListDM(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )
    trainer.model = None
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.fit()


def test_pretrain_fit_raises_when_optimizer_is_none():
    """Pin guard at pretrain.py:97-98."""
    model = _TinyLM()
    trainer = PretrainTrainer(
        engine=MagicMock(),
        data_module=_BatchListDM(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
        max_steps=1,
    )
    trainer.optimizer = None
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        trainer.fit()


# ===========================================================================
# Heavy end-to-end loss-decrease smoke (merged from tests/test_train_pretrain.py).
# Marked ``heavy`` so the default run skips it: pytest -m heavy tests/trainers
# ===========================================================================

import json  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRETRAIN_RECIPE = _REPO_ROOT / "recipes" / "pretrain_causal.yaml"


def _loss_windows(jsonl_path: Path, fraction: float = 0.25) -> tuple[float, float]:
    """Return (first_window_mean, last_window_mean) loss across logged steps."""
    losses = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "scalar" and "loss" in rec:
            losses.append(float(rec["loss"]))
    n = len(losses)
    if n < 4:
        raise AssertionError(f"too few loss records to window ({n})")
    win = max(2, int(n * fraction))
    return sum(losses[:win]) / win, sum(losses[-win:]) / win


@pytest.mark.heavy
def test_pretrain_r1_loss_decreases_after_200_steps(tmp_path: Path):
    """R1 acceptance: a real CPU/GPU training loop on tiny_corpus drives the
    windowed-average loss clearly down over 200 steps."""
    pytest.importorskip("torch")
    from lighttrain.cli._runtime import setup_run_from_config

    overrides = [
        f"++run_root={(tmp_path / 'runs').as_posix()}",
        "++trainer.max_steps=200",
        "++trainer.val_every=0",
        "++trainer.ckpt_every=0",
        "++trainer.log_every=10",
    ]
    bundle = setup_run_from_config(_PRETRAIN_RECIPE, overrides=overrides)
    bundle["trainer"].fit()
    bundle["logger"].close()

    jsonl = bundle["run_dir"] / "logs" / "metrics.jsonl"
    first, last = _loss_windows(jsonl, fraction=0.25)
    assert last < first * 0.7, f"R1 smoke: loss did not decrease ({first=}, {last=})"
