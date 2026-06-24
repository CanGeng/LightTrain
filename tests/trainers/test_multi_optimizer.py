"""Multi-model / multi-optimizer joint update (relocated from
tests/test_axis_b_multi_optimizer.py).

A minimal dual-model paradigm (`dual_lm`) — the smallest thing that exercises
the multi-optimizer machinery, not a real GAN. It overrides ``_step`` to run a
forward+loss on EACH model and drive EACH model's own optimizer via
``apply_update`` (per-model MicroState). The runtime builds both models and both
optimizers from a ``models:``/``optimizers:`` recipe; the test asserts both
models' parameters actually change (joint update happened).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml

from lighttrain.engine.update_rules._primitives import MicroState, apply_update
from lighttrain.protocols import ModelOutput, StepOutput
from lighttrain.registry import register
from lighttrain.trainers.base import Trainer

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "tests" / "fixtures" / "tiny_corpus.txt"


@register("trainer", "dual_lm", force=True)
class _DualLMTrainer(Trainer):
    """Two trainable LMs, each with its own optimizer; one joint step trains
    both (each on next-token CE over the same batch)."""

    def __init__(self, *, grad_clip: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grad_clip = float(grad_clip)
        self._micros = {name: MicroState() for name in self.models}

    def _ce(self, model: Any, input_ids: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids)
        logits = out.outputs["logits"] if isinstance(out, ModelOutput) else out["logits"]
        V = logits.size(-1)
        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_labels = input_ids[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_labels)

    def _step(self, batch: dict[str, Any]) -> StepOutput:
        input_ids = batch["input_ids"].to(self.device)
        metrics: dict[str, Any] = {}
        total = 0.0
        for name in ("actor", "critic"):
            model = self.models[name]
            optimizer = self.optimizers[name]
            loss = self._ce(model, input_ids)
            apply_update(
                loss=loss, model=model, optimizer=optimizer, ctx=self.ctx,
                micro_state=self._micros[name], scheduler=None,
                accelerator=None, grad_clip=self.grad_clip,
                accumulate_grad_batches=1, bus=self.bus,
            )
            metrics[f"loss_{name}"] = float(loss.detach())
            total += float(loss.detach())
        metrics["loss"] = total
        return StepOutput(loss=total, metrics=metrics)


def _recipe(tmp_path: Path) -> Path:
    prof = {
        "name": "tiny_lm", "vocab_size": 260, "d_model": 32,
        "n_layers": 1, "n_heads": 2, "max_seq_len": 32, "dropout": 0.0,
    }
    cfg = {
        "mode": "lab", "seed": 7, "exp": "axis_b", "run_root": str(tmp_path / "runs"),
        "model_profiles": {"a": dict(prof), "b": dict(prof)},
        "models": {
            "actor": {"spec": {"profile": "a"}, "trainable": True, "optimizer": "opt_actor"},
            "critic": {"spec": {"profile": "b"}, "trainable": True, "optimizer": "opt_critic"},
        },
        "optimizers": {
            "opt_actor": {"name": "adamw", "lr": 1.0e-3},
            "opt_critic": {"name": "lion", "lr": 1.0e-2},
        },
        "data": {
            "name": "simple",
            "dataset": {"name": "line_file_text", "path": str(CORPUS), "max_len": 16},
            "tokenizer": {"name": "byte"},
            "collator": {"name": "causal_lm", "max_len": 16},
            "sampler": {"name": "shuffle", "seed": 7},
            "batch_size": 4, "num_workers": 0,
        },
        "loss": {"name": "cross_entropy"},
        "engine": {"name": "standard", "mixed_precision": "no"},
        "trainer": {"name": "dual_lm", "max_steps": 3, "val_every": 0,
                    "ckpt_every": 0, "log_every": 1, "grad_clip": 1.0},
        "logger": [{"name": "jsonl"}],
    }
    p = tmp_path / "axis_b.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


@pytest.mark.skipif(not CORPUS.exists(), reason="tiny_corpus.txt missing")
def test_two_models_two_optimizers_update_jointly(tmp_path):
    from lighttrain.cli._runtime import setup_run_from_config

    bundle = setup_run_from_config(_recipe(tmp_path), mode="lab")
    trainer = bundle["trainer"]

    # Runtime built BOTH models and BOTH optimizers (per-entry pairing).
    assert set(trainer.models) == {"actor", "critic"}
    assert set(trainer.optimizers) == {"actor", "critic"}
    assert all(any(p.requires_grad for p in trainer.models[n].parameters())
               for n in ("actor", "critic"))

    # Distinct optimizer instances with per-entry specs honoured (different
    # types AND different lr) — each trainable model got its OWN optimizer.
    def inner(o):
        return getattr(o, "optimizer", o)

    oa, oc = inner(trainer.optimizers["actor"]), inner(trainer.optimizers["critic"])
    assert oa is not oc
    assert type(oa).__name__ != type(oc).__name__
    assert oa.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert oc.param_groups[0]["lr"] == pytest.approx(1.0e-2)

    def _snapshot(m):
        return torch.cat([p.detach().flatten() for p in m.parameters()]).clone()

    before = {n: _snapshot(trainer.models[n]) for n in ("actor", "critic")}
    metrics = trainer.fit()
    bundle["logger"].close()

    # BOTH models changed → joint multi-optimizer update happened.
    for n in ("actor", "critic"):
        after = _snapshot(trainer.models[n])
        assert not torch.allclose(before[n], after), f"{n} did not update"
    assert "loss_actor" in metrics and "loss_critic" in metrics
