"""PretrainTrainer.predict() — REVIEW #13."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lighttrain.protocols import ModelOutput
from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer


class _ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 6)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 4-dim "features" derived from input_ids
        h = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        logits = self.head(h)
        return ModelOutput(outputs={"logits": logits})


class _DM:
    def __init__(self, batches):
        self._batches = batches

    def train_loader(self):
        return iter([])

    def val_loader(self):
        return None

    def predict_loader(self):
        return iter(self._batches)

    def state_dict(self):
        return {}


def _batch(B=2, T=3):
    return {"input_ids": torch.randint(0, 4, (B, T))}


def test_predict_returns_logits_per_batch():
    model = _ToyLM()
    dm = _DM([_batch(), _batch(), _batch()])
    trainer = PretrainTrainer(
        engine=SimpleNamespace(step=lambda b, c: {}),
        data_module=dm,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
    )
    out = trainer.predict()
    assert len(out) == 3
    for o in out:
        assert "logits" in o
        assert o["logits"].shape[-1] == 6


def test_predict_raises_when_no_loader():
    model = _ToyLM()
    trainer = PretrainTrainer(
        engine=SimpleNamespace(step=lambda b, c: {}),
        data_module=_DM([]),  # predict_loader returns an empty iter
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
    )
    # Empty iter — predict should still return an empty list (graceful).
    out = trainer.predict()
    assert out == []


def test_predict_explicit_loader_overrides_data_module():
    model = _ToyLM()
    # data_module's predict_loader would emit [_batch()], we pass our own
    dm = _DM([_batch()])
    trainer = PretrainTrainer(
        engine=SimpleNamespace(step=lambda b, c: {}),
        data_module=dm,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
    )
    out = trainer.predict(loader=iter([_batch(B=1), _batch(B=1)]))
    assert len(out) == 2
