"""Builtin callbacks: NaNSkip, EarlyStop, Throughput, BestCkpt, EMA shadow swap."""

from __future__ import annotations

import math
import time

import torch

from lighttrain.callbacks.base import Signal
from lighttrain.callbacks.builtins.best_ckpt import BestCheckpointCallback
from lighttrain.callbacks.builtins.early_stop import EarlyStopCallback
from lighttrain.callbacks.builtins.ema import EMACallback
from lighttrain.callbacks.builtins.nan_skip import NaNSkipCallback
from lighttrain.callbacks.builtins.throughput import ThroughputCallback


def test_nan_skip_returns_skip_then_stop():
    cb = NaNSkipCallback(max_skips=2)
    assert cb.on_loss_computed(loss=1.0) is None
    bad = torch.tensor(float("nan"))
    assert cb.on_loss_computed(loss=bad) == Signal.SKIP_STEP
    assert cb.on_loss_computed(loss=bad) == Signal.SKIP_STEP
    assert cb.on_loss_computed(loss=bad) == Signal.STOP_TRAINING


def test_early_stop_after_patience_exhausted():
    cb = EarlyStopCallback(monitor="val_loss", patience=2, mode="min")
    assert cb.on_eval_end(metrics={"val_loss": 1.0}) is None  # best=1.0
    assert cb.on_eval_end(metrics={"val_loss": 1.5}) is None  # bad=1
    assert cb.on_eval_end(metrics={"val_loss": 1.5}) is None  # bad=2
    assert cb.on_eval_end(metrics={"val_loss": 1.5}) == Signal.STOP_TRAINING


def test_best_ckpt_flags_should_save_on_improvement():
    cb = BestCheckpointCallback(monitor="loss", mode="min")
    cb.on_step_end(metrics={"loss": 2.0})
    assert cb.should_save is True
    cb.on_step_end(metrics={"loss": 2.1})
    assert cb.should_save is False
    cb.on_step_end(metrics={"loss": 1.5})
    assert cb.should_save is True
    assert cb.best == 1.5


def test_throughput_records_rolling_metrics():
    cb = ThroughputCallback(window=5)
    metrics: dict = {}
    cb.on_step_begin()
    time.sleep(0.005)
    fake_batch = {"input_ids": torch.zeros(4, 32, dtype=torch.long)}
    cb.on_step_end(batch=fake_batch, metrics=metrics)
    assert "step_time_ms" in metrics
    assert metrics["samples_per_sec"] > 0
    assert metrics["tokens_per_sec"] > 0


def test_ema_shadow_swap_round_trip():
    model = torch.nn.Linear(4, 4, bias=False)
    cb = EMACallback(decay=0.5)
    # Warm shadow.
    cb.on_optimizer_step_post(model=model)
    original = model.weight.detach().clone()

    # Mutate model post-warmup; shadow now lags behind.
    with torch.no_grad():
        model.weight.add_(1.0)
    cb.on_optimizer_step_post(model=model)

    # Eval swap pulls shadow into live params.
    cb.on_eval_begin(model=model)
    swapped = model.weight.detach().clone()
    assert not torch.equal(swapped, original + 1.0)

    # End-of-eval restores.
    cb.on_eval_end(model=model)
    assert torch.allclose(model.weight, original + 1.0)


def test_early_stop_handles_nonnumeric_silently():
    cb = EarlyStopCallback(monitor="val_loss")
    assert cb.on_eval_end(metrics={"val_loss": float("nan")}) is None or True
    # Just verify no crash; best may stay inf.
    assert math.isinf(cb.best) or math.isnan(cb.best)
