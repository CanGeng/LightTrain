"""Tests for PretrainTrainer.predict() (relocated from tests/test_predict.py).

predict() runs forward over a predict_loader and collects per-batch outputs;
an explicit loader argument overrides the data_module's predict_loader.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer
from lighttrain.protocols import ModelOutput


class _ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 6)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return ModelOutput(outputs={"logits": self.head(h)})


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


def _make_trainer(dm) -> PretrainTrainer:
    model = _ToyLM()
    return PretrainTrainer(
        engine=SimpleNamespace(step=lambda b, c: {}),
        data_module=dm,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        model=model,
    )


def test_predict_returns_logits_per_batch():
    trainer = _make_trainer(_DM([_batch(), _batch(), _batch()]))
    out = trainer.predict()
    assert len(out) == 3
    for o in out:
        assert "logits" in o
        assert o["logits"].shape[-1] == 6


def test_predict_returns_empty_list_when_loader_empty():
    """An empty predict_loader yields an empty result list (graceful, no raise)."""
    out = _make_trainer(_DM([])).predict()
    assert out == []


def test_predict_explicit_loader_overrides_data_module():
    trainer = _make_trainer(_DM([_batch()]))  # dm would emit 1 batch
    out = trainer.predict(loader=iter([_batch(B=1), _batch(B=1)]))
    assert len(out) == 2
